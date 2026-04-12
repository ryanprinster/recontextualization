"""
Local GRPO Trainer
==================

GRPO (Group Relative Policy Optimization) trainer for local models using
TRL's GRPOTrainer with code-execution rewards.

Unlike Expert Iteration (generate best → SFT), GRPO:
- Generates multiple completions per prompt
- Computes group-relative advantages from execution rewards
- Updates the policy with clipped policy-gradient loss
- Interleaves generation and optimization
"""

import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..configs.training import GRPOTrainingConfig
from ..dataset_modules.base import (
    BaseDataset,
    ProcessedSample,
    Rollout,
    Sample,
)
from ..models.base import BaseModel
from ..storage import use_cache
from .trainer import Trainer, TrainingResult

logger = logging.getLogger(__name__)


class LocalGRPOTrainer(Trainer):
    """
    GRPO trainer for local models via TRL's GRPOTrainer.

    Uses the existing dataset evaluation pipeline (code execution) as the
    reward function.  The vLLM generation model is unloaded before training
    to avoid GPU memory conflicts — TRL's GRPOTrainer handles its own
    generation with the HuggingFace model.
    """

    def __init__(
        self,
        config: GRPOTrainingConfig,
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
        self.config: GRPOTrainingConfig = config

    # ================================
    # MAIN TRAINING ENTRY POINT
    # ================================

    def train(self) -> TrainingResult:
        """Run GRPO training: generate completions → compute rewards → policy gradient."""
        logger.info("Starting local GRPO training...")

        self.result.status = "running"
        self.result.started_at = datetime.now().isoformat()
        self._save_result()

        try:
            with use_cache():
                logger.info("=== PRE-TRAINING EVALUATION ===")
                self.evaluate_model_performance("pre_training")

            # Prepare the prompt dataset from training samples
            logger.info("=== PREPARING PROMPTS ===")
            training_samples = self.dataset.get_train_samples(
                self.config.max_train_samples
            )
            prompts_dataset, sample_lookup = self._prepare_prompts_dataset(
                training_samples
            )
            logger.info(f"Prepared {len(prompts_dataset)} prompts for GRPO")

            self._save_result()

            # Release the vLLM generation model before loading HF model
            logger.info("Unloading generation model to free GPU memory ...")
            self.model.unload()

            logger.info("=== GRPO TRAINING ===")
            checkpoint_dir = self._run_grpo(prompts_dataset, sample_lookup)

            self.result.model_ref = str(checkpoint_dir)
            self._save_result()

            logger.info("=== POST-TRAINING EVALUATION ===")
            self._post_training_evaluation(checkpoint_dir)

            self.result.status = "completed"
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
    # PROMPT DATASET PREPARATION
    # ================================

    def _prepare_prompts_dataset(
        self, samples: List[Sample]
    ) -> tuple:
        """
        Convert dataset samples into an HF Dataset with a "prompt" column
        in conversational format (list of message dicts), plus a "sample_id"
        column so the reward function can look up the original sample.

        Returns:
            (Dataset, dict): The HF Dataset and a sample_id → Sample lookup dict.
        """
        from datasets import Dataset

        context = self.config.generation_context
        records = []
        sample_lookup: Dict[str, Sample] = {}

        for sample in samples:
            processed: ProcessedSample = self.dataset.process_sample(
                sample, context
            )
            # TRL expects "prompt" as a list of message dicts (conversational)
            records.append({
                "prompt": processed.messages,
                "sample_id": sample.id,
            })
            sample_lookup[sample.id] = sample

        return Dataset.from_list(records), sample_lookup

    # ================================
    # REWARD FUNCTION
    # ================================

    def _make_reward_fn(self, sample_lookup: Dict[str, Sample]):
        """
        Build a reward function compatible with TRL's GRPOTrainer.

        The function receives completions generated by TRL and evaluates them
        using the dataset's code-execution evaluator.
        """
        dataset = self.dataset
        context = self.config.generation_context

        def reward_fn(
            prompts,
            completions,
            sample_id: List[str],
            **kwargs,
        ) -> List[float]:
            rewards = []
            for prompt_msgs, completion_msgs, sid in zip(
                prompts, completions, sample_id
            ):
                sample = sample_lookup[sid]
                processed = dataset.process_sample(sample, context)

                # TRL passes completions in conversational format
                # (list of message dicts) when prompts are conversational.
                # Extract the assistant response text.
                if isinstance(completion_msgs, list):
                    completion_text = completion_msgs[-1]["content"]
                else:
                    completion_text = completion_msgs

                # Build a Rollout so we can use the existing evaluator
                messages = processed.messages + [
                    {"role": "assistant", "content": completion_text}
                ]
                rollout = Rollout(
                    sample=processed,
                    messages=messages,
                    final_response=completion_text,
                )

                # Evaluate via code execution
                evaluated = dataset.evaluate_rollout(rollout)
                rewards.append(evaluated.evaluation_result.score)

            return rewards

        return reward_fn

    # ================================
    # GRPO TRAINING
    # ================================

    def _run_grpo(self, prompts_dataset, sample_lookup: Dict[str, Sample]) -> Path:
        """
        Run GRPO training using TRL's GRPOTrainer.

        Returns the path of the saved checkpoint directory.
        """
        import torch
        from peft import LoraConfig, TaskType
        from trl import GRPOConfig, GRPOTrainer

        checkpoint_dir = Path(self.output_dir) / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build reward function
        reward_fn = self._make_reward_fn(sample_lookup)

        # Configure LoRA if requested
        peft_config = None
        if self.config.use_lora:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )

        # Build GRPOConfig
        grpo_kwargs = dict(
            output_dir=str(checkpoint_dir),
            # GRPO-specific
            num_generations=self.config.num_generations,
            num_iterations=self.config.num_iterations,
            beta=self.config.kl_coef,
            epsilon=self.config.cliprange,
            loss_type=self.config.loss_type,
            scale_rewards=self.config.scale_rewards,
            # Generation
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            # Training
            max_steps=self.config.max_steps,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            # Misc
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="no",
            seed=self.config.training_seed or 42,
            log_completions=True,
            remove_unused_columns=False,
        )

        # vLLM generation (much faster than HF generate)
        if self.config.use_vllm:
            grpo_kwargs.update(
                use_vllm=True,
                vllm_mode=self.config.vllm_mode,
                vllm_gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            )
            if self.config.vllm_max_model_length is not None:
                grpo_kwargs["vllm_max_model_length"] = self.config.vllm_max_model_length

        grpo_config = GRPOConfig(**grpo_kwargs)

        logger.info(
            f"Initialising TRL GRPOTrainer with model={self.config.model_name_or_path}, "
            f"num_generations={self.config.num_generations}, "
            f"loss_type={self.config.loss_type}"
        )

        trainer = GRPOTrainer(
            model=self.config.model_name_or_path,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=prompts_dataset,
            peft_config=peft_config,
        )

        logger.info("Running GRPO training ...")
        train_output = trainer.train()

        self.result.training_metrics["grpo"] = {
            "train_loss": train_output.training_loss,
            "train_runtime_s": train_output.metrics.get("train_runtime"),
            "train_samples_per_second": train_output.metrics.get(
                "train_samples_per_second"
            ),
        }

        logger.info(f"Saving checkpoint to {checkpoint_dir} ...")
        trainer.save_model(str(checkpoint_dir))
        trainer.processing_class.save_pretrained(str(checkpoint_dir))

        # Free trainer GPU memory
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        return checkpoint_dir

    # ================================
    # POST-TRAINING EVALUATION
    # ================================

    def _post_training_evaluation(self, checkpoint_dir: Path) -> None:
        """Merge LoRA checkpoint (if applicable), reload vLLM, and evaluate."""
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

        # Merge LoRA if it was used
        if self.config.use_lora:
            eval_model_dir = self._merge_lora_checkpoint(checkpoint_dir)
        else:
            eval_model_dir = checkpoint_dir

        from ..models.vllm_model import VLLMModel

        ft_model = VLLMModel(
            model_name_or_path=str(eval_model_dir),
            enable_thinking=self.model.enable_thinking,
            thinking_budget_tokens=self.model.thinking_budget_tokens,
            **self.model._engine_kwargs,
        )

        try:
            self.evaluate_model_performance(
                "post_training",
                save_subdir="post_training_evaluation",
                model=ft_model,
            )
        finally:
            ft_model.unload()

    def _merge_lora_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Merge LoRA adapter with the base model and save to disk."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        merged_dir = Path(self.output_dir) / "merged_checkpoint"
        merged_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Merging LoRA adapter from {checkpoint_dir} "
            f"into {self.config.model_name_or_path} ..."
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
        model = model.merge_and_unload()

        logger.info(f"Saving merged model to {merged_dir} ...")
        model.save_pretrained(str(merged_dir))

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        tokenizer.save_pretrained(str(merged_dir))

        del model, base_model
        gc.collect()

        logger.info("LoRA merge complete.")
        return merged_dir
