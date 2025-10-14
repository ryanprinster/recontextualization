"""
Recontextualization Detection Method
===================================

Recontextualization processor that detects problematic rollouts and
recontextualizes them using the dataset. Doesn't need selector since
it handles recontextualization directly via dataset methods.
"""

from typing import Any, Dict, List, Tuple

from ...configs import RecontextualizationConfig
from ...dataset_modules.base import BaseDataset
from ..data_structures import SampleRollouts
from .base import BaseDetectionProcessor


class RecontextualizationProcessor(BaseDetectionProcessor):
    """
    Recontextualization processor using RecontextualizationConfig.
    Requires dataset for rollout recontextualization.
    Doesn't need selector since it handles recontextualization directly via dataset.
    """

    def __init__(self, config: RecontextualizationConfig, dataset: BaseDataset):
        """Initialize with RecontextualizationConfig object and dataset (no selector needed)"""
        super().__init__(config)

        # Recontextualization specific fields
        self.target_context = config.target_context
        self.dataset = dataset  # Need dataset for rollout recontextualization
        self.detected_indices = []  # Track which responses were detected

    def detect_and_process(
        self, sample_rollouts_list: List[SampleRollouts]
    ) -> Tuple[List[SampleRollouts], Dict[str, Any]]:
        """
        Apply recontextualization detection method.

        Behavior adapts based on sample state:
        - SFT mode (has selected rollout): Detect on selected rollout only
        - GRPO mode (no selected rollout): Detect on all rollouts

        For detected samples, recontextualizes all rollouts and selected rollout.
        """

        total_detected = 0

        for sample_rollouts in sample_rollouts_list:
            if sample_rollouts.has_selection:
                target_rollouts = [sample_rollouts.selected_rollout]
                if self._detect_rollouts(target_rollouts)[0]:
                    sample_rollouts.selected_rollout = (
                        self.dataset.recontextualize_rollout(
                            sample_rollouts.selected_rollout, self.target_context
                        )
                    )
                    total_detected += 1
            else:
                target_rollouts = sample_rollouts.rollouts
                detection_results = self._detect_rollouts(target_rollouts)
                if any(detection_results):
                    # Only recontextualize detected rollouts
                    for i, (rollout, detected) in enumerate(
                        zip(target_rollouts, detection_results)
                    ):
                        if detected:
                            sample_rollouts.rollouts[i] = (
                                self.dataset.recontextualize_rollout(
                                    rollout, self.target_context
                                )
                            )
                    total_detected += sum(detection_results)

        # Create balanced metrics
        final_metrics = {
            "method": "recontextualization",
            "target_context": self.target_context,
            "total_detected": total_detected,
            "detection_rate": total_detected / len(sample_rollouts_list)
            if sample_rollouts_list
            else 0,
        }

        return sample_rollouts_list, final_metrics
