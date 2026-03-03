"""
Local Expert-Iteration Trainer
================================

Single-round expert iteration for local models (e.g. Qwen3-8B) using:
- vLLM (or any BaseModel) for fast rollout generation
- TRL SFTTrainer for supervised fine-tuning
- Full CoT (<think>...</think>) included in training data

One "round" = generate rollouts → select best → fine-tune.
Chaining multiple rounds is handled by external scripts: pass the saved
checkpoint directory as `config.model_name_or_path` for the next round.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..configs.training import LocalTrainingConfig
from ..dataset_modules.base import BaseDataset, Rollout
from ..models.base import BaseModel
from ..storage import use_cache
from .trainer import Trainer, TrainingResult

logger = logging.getLogger(__name__)


class LocalTrainer(Trainer):
    """
    Expert-iteration trainer for local models via TRL SFTTrainer.

    Generation and training are kept as separate phases:
      1. Rollouts are generated using the `model` argument (e.g. a VLLMModel).
      2. The SFT model is loaded fresh from `config.model_name_or_path` so the
         two phases never compete for GPU memory.

    CoT handling: Qwen3's <think>...</think> blocks are part of the assistant
    turn in `rollout.messages`, so they are included in the training data
    automatically. No special parsing is needed.
    """

    def __init__(
        self,
        config: LocalTrainingConfig,
        dataset: BaseDataset,
        model: BaseModel,
        output_dir: str,
        selection_config=None,
        detection_config=None,
        evaluation_config=None,
    ):
        super().__init__(
            config,
            dataset,
            model,
            output_dir,
            selection_config,
            detection_config,
            evaluation_config,
        )
        # Narrow the type so IDE / type-checkers see LocalTrainingConfig fields.
        self.config: LocalTrainingConfig = config

    # ================================
    # MAIN TRAINING ENTRY POINT
    # ================================

    def train(self) -> TrainingResult:
        """Run one round of expert iteration: generate → select → fine-tune."""
        logger.info("Starting local expert-iteration training...")

        self.result.status = "running"
        self.result.started_at = datetime.now().isoformat()
        self._save_result()

        try:
            with use_cache():
                logger.info("=== PRE-TRAINING EVALUATION ===")
                self.evaluate_model_performance("pre_training")

                logger.info("=== TRAINING DATA GENERATION ===")
                training_samples = self.dataset.get_train_samples(
                    self.config.max_train_samples
                )
                sample_rollouts_list, pipeline_metrics = self._process_samples(
                    training_samples
                )

            # Collect the rollout selected for each sample.
            training_rollouts: List[Rollout] = [
                sr.selected_rollout
                for sr in sample_rollouts_list
                if sr.has_selection and sr.selected_rollout is not None
            ]

            # Expert-iteration filter: only train on examples that meet the bar.
            training_rollouts = self._filter_by_score(training_rollouts)

            if not training_rollouts:
                raise RuntimeError(
                    "No rollouts passed the score threshold "
                    f"(min_score_threshold={self.config.min_score_threshold}). "
                    "Cannot proceed with fine-tuning."
                )

            self.result.training_metrics["pipeline"] = pipeline_metrics
            self.result.training_metrics["n_training_rollouts"] = len(training_rollouts)
            self._save_result()

            # Release the generation model (vLLM engine, etc.) before loading
            # the HF training model. This is a no-op for API-backed models.
            logger.info("Unloading generation model to free GPU memory ...")
            self.model.unload()

            logger.info("=== FINE-TUNING ===")
            checkpoint_dir = self._run_sft(training_rollouts)

            self.result.status = "completed"
            self.result.model_ref = str(checkpoint_dir)
            self.result.completed_at = datetime.now().isoformat()
            self._save_result()

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.result.status = "failed"
            self.result.error = str(e)
            self.result.completed_at = datetime.now().isoformat()
            self._save_result()

        return self.result

    # ================================
    # DATA FORMATTING
    # ================================

    def _filter_by_score(self, rollouts: List[Rollout]) -> List[Rollout]:
        """Keep only rollouts whose evaluation score meets the configured threshold."""
        threshold = self.config.min_score_threshold
        if threshold <= 0.0:
            return rollouts

        filtered = [
            r
            for r in rollouts
            if r.evaluation_result is not None
            and r.evaluation_result.score >= threshold
        ]
        logger.info(
            f"Score filter (>= {threshold}): "
            f"{len(filtered)}/{len(rollouts)} rollouts kept"
        )
        return filtered

    def _build_training_dataset(self, rollouts: List[Rollout], tokenizer) -> "Dataset":
        """
        Convert rollouts to a HuggingFace Dataset for TRL SFTTrainer.

        The tokenizer's chat template is applied here so TRL receives a plain
        `text` column. `add_generation_prompt=False` ensures the final
        assistant turn (including any <think> CoT) is part of the training
        target rather than being treated as a prompt prefix.
        """
        from datasets import Dataset

        records = []
        for rollout in rollouts:
            text = tokenizer.apply_chat_template(
                rollout.messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            records.append({"text": text})

        return Dataset.from_list(records)

    # ================================
    # FINE-TUNING
    # ================================

    def _run_sft(self, rollouts: List[Rollout]) -> Path:
        """
        Fine-tune with TRL SFTTrainer.

        Loads the model and tokenizer from `config.model_name_or_path`
        independently of whichever model is used for generation, avoiding
        GPU memory conflicts with vLLM.

        Returns the path of the saved checkpoint directory.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        checkpoint_dir = Path(self.output_dir) / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading tokenizer from {self.config.model_name_or_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )

        logger.info("Building training dataset ...")
        dataset = self._build_training_dataset(rollouts, tokenizer)
        logger.info(f"Training dataset: {len(dataset)} examples")

        logger.info(f"Loading model from {self.config.model_name_or_path} ...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if self.config.use_lora:
            hf_model = self._apply_lora(hf_model)

        sft_config = SFTConfig(
            output_dir=str(checkpoint_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            bf16=True,
            logging_steps=10,
            # Save manually after training; no mid-run checkpoints by default.
            save_strategy="no",
            seed=self.config.training_seed or 42,
            dataset_text_field="text",
            max_length=self.config.max_seq_length,
        )

        trainer = SFTTrainer(
            model=hf_model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        logger.info("Running SFT ...")
        train_output = trainer.train()

        self.result.training_metrics["sft"] = {
            "train_loss": train_output.training_loss,
            "train_runtime_s": train_output.metrics.get("train_runtime"),
            "train_samples_per_second": train_output.metrics.get(
                "train_samples_per_second"
            ),
        }

        logger.info(f"Saving checkpoint to {checkpoint_dir} ...")
        trainer.save_model(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))

        return checkpoint_dir

    def _apply_lora(self, hf_model) -> Any:
        """Wrap model with PEFT LoRA adapters and log trainable parameter count."""
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            # None lets PEFT auto-detect all linear layers.
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        hf_model = get_peft_model(hf_model, lora_config)
        hf_model.print_trainable_parameters()
        return hf_model
