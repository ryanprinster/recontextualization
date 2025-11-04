"""
Base Detection Method
====================

Abstract base class for all detection methods with common functionality.
Includes common detection logic based on true/false positive rates.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ...configs import DetectionConfig
from ...dataset_modules.base import Rollout
from ..data_structures import SampleRollouts


class BaseDetectionProcessor(ABC):
    """
    Base class for all detection processors.
    Uses DetectionConfig objects directly (Hydra pattern).
    """

    def __init__(self, config: DetectionConfig):
        """Initialize with DetectionConfig object (Hydra pattern)"""
        self.config = config

        # Extract common config fields for cleaner code
        self.reward_categories = config.reward_categories
        self.true_positive_rate = config.true_positive_rate
        self.false_positive_rate = config.false_positive_rate

    def _detect_rollouts(self, rollouts: List[Rollout]) -> List[bool]:
        """
        Detect which rollouts meet criteria (pure function).

        Args:
            rollouts: List of rollouts to check

        Returns:
            List of booleans indicating which rollouts were detected
        """
        results = []

        for rollout in rollouts:
            # Skip rollouts without evaluation
            if rollout.evaluation_result is None:
                results.append(False)
                continue

            # Determine detection probability
            if rollout.evaluation_result.detection_category in self.reward_categories:
                # Target category - use true positive rate
                detect_prob = self.true_positive_rate
            else:
                # Other category - use false positive rate
                detect_prob = self.false_positive_rate

            # Probabilistically decide to detect
            results.append(random.random() < detect_prob)

        return results

    @abstractmethod
    def detect_and_process(
        self, sample_rollouts_list: List[SampleRollouts]
    ) -> Tuple[List[SampleRollouts], Dict[str, Any]]:
        """
        Subclasses implement their full detection and processing logic.
        Use _detect_rollouts() for the core detection functionality.
        """
        pass
