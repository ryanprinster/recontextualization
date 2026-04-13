"""
Tinker GRPO Trainer
===================

Thin wrapper over ``tinker_cookbook.rl.train.main``.

We hand the cookbook a ``Config`` pointing at our ``ImpossibleLiveCodeRLDatasetBuilder``,
let it drive sampling + PPO forward/backward + optim_step + checkpointing, then
load the final sampler checkpoint into a ``TinkerModel`` for post-training
evaluation via the base ``Trainer`` machinery.
"""

import asyncio
import logging
import math
from datetime import datetime
from typing import Optional

from ..configs.training import TinkerGRPOTrainingConfig
from ..dataset_modules.base import BaseDataset
from ..models.base import BaseModel
from .cookbook_dataset import ImpossibleLiveCodeRLDatasetBuilder, MBPPGenerationRLDatasetBuilder
from .trainer import Trainer, TrainingResult

logger = logging.getLogger(__name__)


class TinkerGRPOTrainer(Trainer):
    """
    GRPO via Tinker, using tinker_cookbook's RL loop.

    The training loop itself (sampling, advantage computation, PPO loss,
    optim step, checkpointing) runs inside ``rl.train.main``.  This class
    handles pre/post-training evaluation and checkpoint discovery so the
    result is plugged into the project's base ``Trainer`` workflow.
    """

    def __init__(
        self,
        config: TinkerGRPOTrainingConfig,
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
        self.config: TinkerGRPOTrainingConfig = config

    # ================================================================
    # MAIN ENTRY POINT
    # ================================================================

    def train(self) -> TrainingResult:
        logger.info("Starting Tinker GRPO training (cookbook loop)...")

        self.result.status = "running"
        self.result.started_at = datetime.now().isoformat()
        self._save_result()

        try:
            logger.info("=== PRE-TRAINING EVALUATION ===")
            self.evaluate_model_performance("pre_training")

            # Release anything the pre-eval model is holding (no-op for TinkerModel).
            self.model.unload()

            logger.info("=== BUILDING COOKBOOK CONFIG ===")
            cookbook_config = self._build_cookbook_config()

            logger.info("=== RUNNING COOKBOOK RL LOOP ===")
            self._run_rl_loop(cookbook_config)

            self._save_result()

            logger.info("=== POST-TRAINING EVALUATION ===")
            self._post_training_evaluation()

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

    # ================================================================
    # COOKBOOK CONFIG
    # ================================================================

    def _build_cookbook_config(self):
        from tinker_cookbook.rl.train import Config as CookbookConfig

        # The cookbook caps training at min(max_steps, len(dataset)).
        # Compute an epochs multiplier so the dataset is long enough to
        # saturate the requested step count.
        test_split = getattr(self.dataset, "test_split", "oneoff")
        train_sample_count = len(self.dataset.train_samples) if hasattr(self.dataset, "train_samples") else 0
        if self.config.max_train_samples is not None:
            train_sample_count = min(train_sample_count, self.config.max_train_samples)

        if train_sample_count > 0:
            base_batches = max(
                1, math.ceil(train_sample_count / max(1, self.config.prompt_batch_size))
            )
            epochs = max(1, math.ceil(self.config.num_train_steps / base_batches))
        else:
            epochs = 1

        from ..dataset_modules.mbpp.generation.dataset import MBPPGenerationDataset

        common_kwargs = dict(
            model_name_for_tokenizer=self.config.base_model,
            renderer_name=self.config.renderer_name,
            batch_size=self.config.prompt_batch_size,
            group_size=self.config.group_size,
            generation_context=self.config.generation_context,
            seed=self.config.seed or self.config.training_seed or 0,
            format_coef=self.config.format_coef,
            reward_timeout=self.config.reward_timeout,
            eval_holdout=self.config.eval_holdout,
            max_train_samples=self.config.max_train_samples,
            epochs=epochs,
        )

        if isinstance(self.dataset, MBPPGenerationDataset):
            dataset_builder = MBPPGenerationRLDatasetBuilder(
                **common_kwargs,
                use_incorrect_tests=self.dataset.use_incorrect_tests,
                base_suffix=self.dataset.base_suffix,
                data_path=self.dataset.data_path,
                train_ratio=self.dataset.train_ratio,
            )
        else:
            dataset_builder = ImpossibleLiveCodeRLDatasetBuilder(
                **common_kwargs,
                test_split=test_split,
            )

        cookbook_config = CookbookConfig(
            learning_rate=self.config.learning_rate,
            dataset_builder=dataset_builder,
            model_name=self.config.base_model,
            max_tokens=self.config.max_completion_length,
            log_path=self.output_dir,
            save_every=self.config.save_every,
            eval_every=self.config.eval_every,
            renderer_name=self.config.renderer_name,
            lora_rank=self.config.lora_rank,
            temperature=self.config.temperature,
            loss_fn="ppo",
            loss_fn_config={
                "clip_low_threshold": 0.8,
                "clip_high_threshold": 1.28,
            },
            max_steps=self.config.num_train_steps,
            remove_constant_reward_groups=self.config.remove_constant_reward_groups,
            ttl_seconds=self.config.ttl_seconds,
            kl_penalty_coef=0.0,
        )
        return cookbook_config

    # ================================================================
    # RUN LOOP
    # ================================================================

    def _run_rl_loop(self, cookbook_config) -> None:
        from tinker_cookbook.rl import train as rl_train

        asyncio.run(rl_train.main(cookbook_config))

        self.result.model_ref = f"tinker://{self.config.base_model}/grpo_final"
        self.result.backend_data["cookbook_log_path"] = self.output_dir

    # ================================================================
    # POST-TRAINING EVALUATION
    # ================================================================

    def _post_training_evaluation(self) -> None:
        """Locate the final sampler checkpoint and evaluate via TinkerModel."""
        from tinker_cookbook import checkpoint_utils

        from ..models.tinker_model import TinkerModel

        record = checkpoint_utils.get_last_checkpoint(
            self.output_dir, required_key="sampler_path"
        )
        if record is None or record.sampler_path is None:
            logger.warning(
                "No sampler checkpoint found in %s; skipping post-training eval.",
                self.output_dir,
            )
            return

        self.result.backend_data["final_sampler_path"] = record.sampler_path
        logger.info(f"Loading final sampler from {record.sampler_path}")

        ft_model = TinkerModel(
            base_model=self.config.base_model,
            model_path=record.sampler_path,
        )
        self.evaluate_model_performance(
            "post_training",
            save_subdir="post_training_evaluation",
            model=ft_model,
        )
