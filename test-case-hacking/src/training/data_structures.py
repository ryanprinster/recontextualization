"""
Shared data classes for training workflow.
This module avoids circular imports between trainer and selection methods.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..dataset_modules.base import Rollout, Sample


@dataclass
class SampleRollouts:
    """Container for rollouts per sample with flexible generation support"""

    sample_index: int
    sample: Sample
    rollouts: List[Rollout] = field(default_factory=list)
    selected_rollout: Optional[Rollout] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_rollouts(self) -> bool:
        return len(self.rollouts) > 0

    @property
    def has_selection(self) -> bool:
        return self.selected_rollout is not None

    def get_non_selected_rollouts(self) -> List[Rollout]:
        """Get rollouts that weren't selected (for weak rejection)"""
        if self.selected_rollout is None:
            return self.rollouts
        return [r for r in self.rollouts if r != self.selected_rollout]


@dataclass
class DetectionResult:
    """Result of detection process"""

    detected_indices: List[int]  # Indices of samples that were detected
    metrics: Dict[str, Any]  # Detection metrics (counts, rates, etc.)
