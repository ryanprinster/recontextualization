"""
Tinker Expert-Iteration Trainer
================================

Single-round expert iteration using Tinker's managed LoRA training:
- Any BaseModel (vLLM, OpenAI, or TinkerModel) for rollout generation
- Tinker TrainingClient for remote LoRA fine-tuning
- TinkerModel wrapping the resulting SamplingClient for post-training eval

One "round" = generate rollouts → select best → fine-tune on Tinker → evaluate.
Chaining multiple rounds is handled by external scripts.
"""

import logging
import math
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

from ..configs.training import TinkerTrainingConfig
from ..dataset_modules.base import BaseDataset, Rollout
from ..models.base import BaseModel
from ..storage import use_cache
from .trainer import Trainer, TrainingResult

logger = logging.getLogger(__name__)


class TinkerTrainer(Trainer):
    """
    Expert-iteration trainer using Tinker's managed LoRA training API.

    Generation and training are fully decoupled:
      1. Rollouts are generated using the `model` argument (any BaseModel).
      2. Fine-tuning runs remotely on Tinker's GPU clusters via the
         TrainingClient API (forward_backward → optim_step loop).
      3. Post-training evaluation uses a TinkerModel wrapping the
         SamplingClient returned after training.
    """

    def __init__(
        self,
        config: TinkerTrainingConfig,
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
        self.config: TinkerTrainingConfig = config

    # ================================
    # MAIN TRAINING ENTRY POINT
    # ================================

    def train(self) -> TrainingResult:
        """Run one round of expert iteration: generate → select → fine-tune on Tinker."""
        logger.info("Starting Tinker expert-iteration training...")

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

            # Release the generation model (vLLM engine, etc.) before training.
            # This is a no-op for API-backed models.
            logger.info("Unloading generation model to free GPU memory ...")
            self.model.unload()

            logger.info("=== FINE-TUNING (Tinker) ===")
            sampling_client = self._run_tinker_training(training_rollouts)

            self._save_result()

            logger.info("=== POST-TRAINING EVALUATION ===")
            self._post_training_evaluation(sampling_client)

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

    def resume(self) -> TrainingResult:
        """Resume training from a saved Tinker checkpoint."""
        checkpoint_path = self.result.backend_data.get("tinker_checkpoint_path")
        if not checkpoint_path:
            return TrainingResult(
                status="failed",
                error="No Tinker checkpoint path found in training state. Cannot resume.",
            )

        import tinker

        logger.info(f"Resuming from Tinker checkpoint: {checkpoint_path}")
        service_client = tinker.ServiceClient()
        training_client = service_client.create_training_client_from_state_with_optimizer(
            checkpoint_path
        )

        # Resume the training loop from the saved step
        start_step = self.result.backend_data.get("tinker_last_step", 0)
        total_steps = self.result.backend_data.get("tinker_total_steps", 0)

        if start_step >= total_steps:
            logger.info("Training already completed. Skipping to evaluation.")
            sampling_client = training_client.save_weights_and_get_sampling_client()
        else:
            # Reload datums from saved training data and continue
            logger.warning(
                "Mid-training resume requires re-preparing datums. "
                "This is not yet implemented — restarting full training."
            )
            return TrainingResult(
                status="failed",
                error="Mid-training resume not yet implemented. Please restart training.",
            )

        self._post_training_evaluation(sampling_client)
        self.result.status = "completed"
        self.result.completed_at = datetime.now().isoformat()
        self._save_result()
        return self.result

    # ================================
    # DATA FORMATTING
    # ================================

    def _filter_by_score(self, rollouts: List[Rollout]) -> List[Rollout]:
        """Keep only rollouts whose evaluation score exceeds the configured threshold."""
        threshold = self.config.min_score_threshold

        filtered = [
            r
            for r in rollouts
            if r.evaluation_result is not None
            and r.evaluation_result.score > threshold
        ]
        logger.info(
            f"Score filter (> {threshold}): "
            f"{len(filtered)}/{len(rollouts)} rollouts kept"
        )
        return filtered

    def _prepare_datums(self, rollouts: List[Rollout], renderer, tokenizer) -> list:
        """Convert rollouts to tinker.Datum objects for training."""
        from tinker_cookbook.supervised.data import conversation_to_datum
        from tinker_cookbook.renderers import TrainOnWhat

        # Map config string to TrainOnWhat enum
        train_on_what_map = {
            "last_turn": TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            "all_assistant": TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            "all": TrainOnWhat.ALL,
        }
        train_on_what = train_on_what_map.get(
            self.config.train_on_what, TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        datums = []
        skipped = 0
        for rollout in rollouts:
            try:
                datum = conversation_to_datum(
                    rollout.messages,
                    renderer,
                    max_length=self.config.max_seq_length,
                    train_on_what=train_on_what,
                )
                datums.append(datum)
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipped rollout during datum preparation: {e}")

        if skipped:
            logger.warning(f"Skipped {skipped}/{len(rollouts)} rollouts during datum preparation")
        logger.info(f"Prepared {len(datums)} training datums")
        return datums

    # ================================
    # FINE-TUNING
    # ================================

    def _run_tinker_training(self, rollouts: List[Rollout]):
        """
        Run Tinker LoRA training loop and return the resulting SamplingClient.

        Steps:
          1. Create a TrainingClient with LoRA configuration.
          2. Convert rollouts to tinker.Datum objects.
          3. Run forward_backward → optim_step loop over batched data.
          4. Save weights and return a SamplingClient for inference.
        """
        import tinker
        from tinker_cookbook import model_info, renderers

        # Create service and training clients
        service_client = tinker.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=self.config.base_model,
            rank=self.config.lora_rank,
            seed=self.config.seed or self.config.training_seed or 42,
            train_mlp=self.config.train_mlp,
            train_attn=self.config.train_attn,
            train_unembed=self.config.train_unembed,
        )

        # Get tokenizer and renderer for datum preparation
        tokenizer = training_client.get_tokenizer()
        renderer_name = model_info.get_recommended_renderer_name(self.config.base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        # Prepare training data
        datums = self._prepare_datums(rollouts, renderer, tokenizer)
        if not datums:
            raise RuntimeError("No datums prepared — cannot proceed with training.")

        # Configure optimizer
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.adam_beta1,
            beta2=self.config.adam_beta2,
            eps=self.config.adam_eps,
        )

        # Training loop
        n_total_examples = len(datums) * self.config.n_epochs
        n_batches = math.ceil(n_total_examples / self.config.batch_size)

        logger.info(
            f"Tinker training: {len(datums)} datums × {self.config.n_epochs} epochs = "
            f"{n_total_examples} examples, {n_batches} batches of {self.config.batch_size}"
        )

        # Store total steps for potential resume
        self.result.backend_data["tinker_total_steps"] = n_batches

        # Build epoch-aware index sequence
        rng = random.Random(self.config.seed or self.config.training_seed or 42)
        indices = []
        for _ in range(self.config.n_epochs):
            epoch_indices = list(range(len(datums)))
            rng.shuffle(epoch_indices)
            indices.extend(epoch_indices)

        total_loss = 0.0
        n_loss_steps = 0

        for batch_idx in range(n_batches):
            start_time = time.time()

            # Extract batch
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, len(indices))
            batch = [datums[indices[i]] for i in range(batch_start, batch_end)]

            # Forward-backward pass and optimizer step
            fwd_bwd_future = training_client.forward_backward(
                batch, loss_fn="cross_entropy"
            )
            optim_step_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_step_future.result()

            # Log metrics
            step_time = time.time() - start_time
            if hasattr(fwd_bwd_result, "loss_fn_outputs") and fwd_bwd_result.loss_fn_outputs:
                step_losses = [
                    out.get("loss", 0.0) if isinstance(out, dict) else 0.0
                    for out in fwd_bwd_result.loss_fn_outputs
                ]
                step_loss = sum(step_losses) / len(step_losses) if step_losses else 0.0
                total_loss += step_loss
                n_loss_steps += 1

            if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == 0:
                avg_loss = total_loss / n_loss_steps if n_loss_steps else 0.0
                logger.info(
                    f"Step {batch_idx + 1}/{n_batches} | "
                    f"avg_loss={avg_loss:.4f} | step_time={step_time:.1f}s"
                )

            # Periodic checkpointing
            if (
                self.config.save_every > 0
                and (batch_idx + 1) % self.config.save_every == 0
                and batch_idx + 1 < n_batches
            ):
                checkpoint_future = training_client.save_state(
                    name=f"step_{batch_idx + 1:06d}",
                    ttl_seconds=self.config.checkpoint_ttl_seconds,
                )
                checkpoint_path = checkpoint_future.result()
                self.result.backend_data["tinker_checkpoint_path"] = str(checkpoint_path)
                self.result.backend_data["tinker_last_step"] = batch_idx + 1
                self._save_result()
                logger.info(f"Checkpoint saved at step {batch_idx + 1}")

        # Save training metrics
        self.result.training_metrics["tinker_sft"] = {
            "total_steps": n_batches,
            "total_examples": n_total_examples,
            "avg_loss": total_loss / n_loss_steps if n_loss_steps else None,
        }

        # Save final weights and get sampling client
        logger.info("Saving final weights and creating SamplingClient ...")
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name="final"
        )

        # Store model path for reference
        self.result.model_ref = f"tinker://{self.config.base_model}/final"
        self.result.backend_data["tinker_last_step"] = n_batches

        return sampling_client

    # ================================
    # POST-TRAINING EVALUATION
    # ================================

    def _post_training_evaluation(self, sampling_client) -> None:
        """Create TinkerModel from SamplingClient and evaluate the fine-tuned model."""
        from ..models.tinker_model import TinkerModel

        ft_model = TinkerModel(sampling_client=sampling_client)
        self.evaluate_model_performance(
            "post_training",
            save_subdir="post_training_evaluation",
            model=ft_model,
        )
