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
from ..storage import RolloutStorage, is_cache_enabled, load_cached_rollouts

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

        # Check cache first if enabled
        if is_cache_enabled():
            sample_ids = [sample.id for sample in samples]
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
                    self.logger.info(f"Using cached rollouts for {len(samples)} samples ({generation_config.n_rollouts} rollouts each)")
                    return cached_rollouts
            except Exception as e:
                # Cache loading failed - fall back to generation
                self.logger.warning(f"Cache loading failed, falling back to generation: {e}")

        # Step 1: Process samples with context (on-demand, no caching)
        processed_samples = []
        for sample in samples:
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

                # Update progress bar
                pbar.update(len(batch_samples))

        self.logger.debug(
            f"Generated {sum(len(group) for group in all_rollouts)} total rollouts "
            f"for {len(samples)} samples with context '{context}'"
            f"{' (evaluated)' if evaluate else ' (unevaluated)'}"
        )

        return all_rollouts

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
