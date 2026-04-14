"""
Unified Rollout Generator
========================

Single rollout generator used by all components (trainer, evaluator, standalone).
Always returns grouped rollouts for consistency across the codebase.

Features:
- Clean, config-first interface
- Unified batching (always used internally)
- Built-in progress tracking
"""

import logging
from functools import partial
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..configs.generation import GenerationConfig
from ..dataset_modules.base import BaseDataset, Rollout, Sample
from ..models.base import BaseModel
from ..storage import (
    RolloutCheckpoint,
    RolloutStorage,
    is_cache_enabled,
    load_cached_rollouts,
)

logger = logging.getLogger(__name__)


class RolloutGenerator:
    """
    Unified rollout generator for all components.

    Key features:
    - Config-first interface (n_rollouts comes from GenerationConfig)
    - Always uses batching internally for consistency
    - Clean, simple API with progress tracking
    - Single source of truth for all rollout generation logic
    """

    def __init__(self, model: BaseModel, dataset: BaseDataset):
        """
        Initialize rollout generator.

        Args:
            model: Model to use for generation
            dataset: Dataset that handles rollout generation and context processing
        """
        self.model = model
        self.dataset = dataset
        self.storage = RolloutStorage()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def generate_rollouts(
        self,
        samples: List[Sample],
        context: str,
        generation_config: Optional[GenerationConfig] = None,
        evaluate: bool = True,
        checkpoint: Optional[RolloutCheckpoint] = None,
        **kwargs,
    ) -> List[List[Rollout]]:
        """
        Generate rollouts for samples with unified batching.

        Clean, config-first interface that always uses batching internally.
        All parameters come from the GenerationConfig for consistency.

        Args:
            samples: Raw samples to generate rollouts for
            context: Context to apply to all samples
            generation_config: Generation parameters (uses defaults if None)
            evaluate: Whether to evaluate rollouts during generation (default: True)
            **kwargs: Additional generation parameters

        Returns:
            List[List[Rollout]]: rollouts[i] = list of rollouts for samples[i]
            If evaluate=True, rollouts come with evaluation_result populated.
        """
        if not samples:
            return []

        # Use default generation config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()

        # Resume from checkpoint: drop any samples whose rollouts were already
        # persisted in a prior run, and stash their rehydrated groups so we can
        # weave them back into the final output in original sample order.
        cached_groups_by_id: Dict[str, List[Rollout]] = {}
        if checkpoint is not None:
            completed_ids = checkpoint.completed_sample_ids()
            if completed_ids:
                for group in checkpoint.load_completed_rollouts():
                    if group:
                        cached_groups_by_id[group[0].sample.sample.id] = group
                remaining_samples = [s for s in samples if s.id not in completed_ids]
                skipped = len(samples) - len(remaining_samples)
                if skipped:
                    self.logger.info(
                        "Resuming from checkpoint: skipping %d/%d samples already completed for context '%s'",
                        skipped,
                        len(samples),
                        context,
                    )
                if not remaining_samples:
                    return [cached_groups_by_id[s.id] for s in samples if s.id in cached_groups_by_id]
                samples_to_run = remaining_samples
            else:
                samples_to_run = samples
        else:
            samples_to_run = samples

        # Check cache first if enabled. When resuming from a checkpoint we
        # only query the cache for the still-to-run samples.
        if is_cache_enabled():
            sample_ids = [sample.id for sample in samples_to_run]
            model_name = getattr(self.model, 'name', str(type(self.model).__name__))
            dataset_name = getattr(self.dataset, 'name', str(type(self.dataset).__name__))

            try:
                cached_rollouts = load_cached_rollouts(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    context=context,
                    generation_config_dict=generation_config.to_dict(),
                    sample_ids=sample_ids,
                    requested_n_rollouts=generation_config.n_rollouts
                )

                if cached_rollouts is not None:
                    self.logger.info(f"Using cached rollouts for {len(samples_to_run)} samples ({generation_config.n_rollouts} rollouts each)")
                    run_groups_by_id = {
                        group[0].sample.sample.id: group
                        for group in cached_rollouts
                        if group
                    }
                    return self._merge_in_sample_order(samples, cached_groups_by_id, run_groups_by_id)
            except Exception as e:
                # Cache loading failed - fall back to generation
                self.logger.warning(f"Cache loading failed, falling back to generation: {e}")

        # Step 1: Process samples with context (on-demand, no caching)
        processed_samples = []
        for sample in samples_to_run:
            processed_sample = self.dataset.process_sample(sample, context)
            processed_samples.append(processed_sample)

        # Step 2: Generate rollouts using unified batching
        all_rollouts = []
        total_samples = len(processed_samples)
        batch_size = generation_config.batch_size

        self.logger.info(
            f"Generating {generation_config.n_rollouts} rollouts for {total_samples} samples"
        )

        # Process in batches with tqdm progress bar
        with tqdm(total=total_samples, desc="Generating & Evaluating", unit="samples") as pbar:
            for i in range(0, total_samples, batch_size):
                batch_end = min(i + batch_size, total_samples)
                batch_samples = processed_samples[i:batch_end]

                # Create model generation function for this batch
                model_generate_fn = self._create_generation_function(
                    generation_config, **kwargs
                )

                # Generate rollouts for this batch using dataset
                batch_grouped_rollouts = self.dataset.generate_rollouts_batch(
                    batch_samples, model_generate_fn,
                    n_rollouts=generation_config.n_rollouts
                )

                # Attach generation metadata from model (if available)
                gen_metadata = getattr(self.model, '_last_generation_metadata', None)
                if gen_metadata:
                    flat_idx = 0
                    for group in batch_grouped_rollouts:
                        for rollout in group:
                            if flat_idx < len(gen_metadata) and gen_metadata[flat_idx]:
                                rollout.metadata = gen_metadata[flat_idx][0]
                            flat_idx += 1

                # Evaluate rollouts immediately if requested
                if evaluate:
                    # Flatten batch rollouts for evaluation
                    batch_flat_rollouts = [rollout for group in batch_grouped_rollouts for rollout in group]
                    # Evaluate using batch method
                    evaluated_batch_rollouts = self.dataset.evaluate_rollouts_batch(batch_flat_rollouts)
                    # Reconstruct grouped structure - all groups have same size (n_rollouts)
                    n_rollouts = generation_config.n_rollouts
                    batch_grouped_rollouts = [
                        evaluated_batch_rollouts[j:j + n_rollouts] 
                        for j in range(0, len(evaluated_batch_rollouts), n_rollouts)
                    ]

                all_rollouts.extend(batch_grouped_rollouts)

                # Persist the completed batch to the streaming checkpoint (if
                # any) so trajectories survive crashes and are tailable live.
                if checkpoint is not None:
                    checkpoint.append_batch(batch_grouped_rollouts)

                # Update progress bar
                pbar.update(len(batch_samples))

        self.logger.debug(
            f"Generated {sum(len(group) for group in all_rollouts)} total rollouts "
            f"for {len(samples_to_run)} samples with context '{context}'"
            f"{' (evaluated)' if evaluate else ' (unevaluated)'}"
        )

        if cached_groups_by_id:
            run_groups_by_id = {
                group[0].sample.sample.id: group
                for group in all_rollouts
                if group
            }
            return self._merge_in_sample_order(samples, cached_groups_by_id, run_groups_by_id)

        return all_rollouts

    @staticmethod
    def _merge_in_sample_order(
        samples: List[Sample],
        cached_by_id: Dict[str, List[Rollout]],
        run_by_id: Dict[str, List[Rollout]],
    ) -> List[List[Rollout]]:
        """Return sample-groups in the order of the caller's input samples."""
        merged: List[List[Rollout]] = []
        for sample in samples:
            if sample.id in run_by_id:
                merged.append(run_by_id[sample.id])
            elif sample.id in cached_by_id:
                merged.append(cached_by_id[sample.id])
        return merged

    def _create_generation_function(
        self, generation_config: GenerationConfig, **kwargs
    ):
        """
        Create pre-configured model generation function.

        Args:
            generation_config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            Callable that matches dataset's expected signature
        """
        return partial(
            self.model.generate,
            temperature=generation_config.temperature,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            **kwargs,
        )

    def get_info(self) -> Dict[str, Any]:
        """Get information about this rollout generator"""
        return {
            "model": self.model.get_info(),
            "dataset": self.dataset.__class__.__name__,
            "available_contexts": self.dataset.available_contexts,
        }
