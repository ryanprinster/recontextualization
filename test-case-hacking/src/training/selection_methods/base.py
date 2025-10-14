"""
Base Selection Method
====================

Abstract base class for all selection methods with common functionality.
Includes default selection logic that can be used when no specific config is provided.
"""

from abc import ABC
from typing import Any, Callable, Dict, List, Tuple

from ...dataset_modules.base import Rollout, Sample
from ..data_structures import SampleRollouts


class BaseRolloutSelector(ABC):
    """Base class for all selectors with common functionality"""

    def __init__(self, generate_fn: Callable[[List[Sample]], List[List[Rollout]]]):
        """Base initialization for all selectors with pre-configured generation function"""
        self.generate_fn = generate_fn

    def select_rollouts(
        self, sample_rollouts_list: List[SampleRollouts]
    ) -> Tuple[List[SampleRollouts], Dict[str, Any]]:
        """
        Default implementation: generate N rollouts, select first.

        Subclasses MUST override this method to implement their selection logic.
        This default is only used when BaseRolloutSelector is used directly.
        """
        # Enforce that subclasses override this method
        if self.__class__ != BaseRolloutSelector:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement select_rollouts(). "
                "The base implementation should only be used directly."
            )

        # Use base class helper to ensure rollouts
        total_generated = self._ensure_rollouts(sample_rollouts_list)

        # Default selection: select first rollout for each sample
        for sample_rollouts in sample_rollouts_list:
            if not sample_rollouts.has_selection and sample_rollouts.has_rollouts:
                sample_rollouts.selected_rollout = sample_rollouts.rollouts[0]

        metrics = {
            "method": "base_selection",
            "selection_strategy": "first",
            "samples_processed": len(sample_rollouts_list),
            "rollouts_generated": total_generated,
        }

        return sample_rollouts_list, metrics

    def _ensure_rollouts(self, sample_rollouts_list: List[SampleRollouts]) -> int:
        """Common helper to ensure samples have enough rollouts"""
        # Find samples that need rollouts
        samples_needing_rollouts = [
            sr.sample for sr in sample_rollouts_list if not sr.has_rollouts
        ]

        if not samples_needing_rollouts:
            return 0

        # Generate rollouts for all samples that need them
        new_rollouts_list = self.generate_fn(samples_needing_rollouts)

        # Assign rollouts back - much simpler with zip
        rollout_iter = iter(new_rollouts_list)
        total_generated = 0

        for sample_rollouts in sample_rollouts_list:
            if not sample_rollouts.has_rollouts:
                rollouts = next(rollout_iter)
                sample_rollouts.rollouts = rollouts
                total_generated += len(rollouts)

        return total_generated
