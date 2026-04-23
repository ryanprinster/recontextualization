"""
Multiple choice sample definition.
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..base import Sample


@dataclass
class MultipleChoiceSample(Sample):
    """Multiple choice sample with properly typed fields"""
    question: str
    high_reward_answer: str
    dataset: str  # "revealing_score" or "world_affecting_reward"

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict with domain-specific data"""
        return {
            "id": self.id,
            "question": self.question,
            "high_reward_answer": self.high_reward_answer,
            "dataset": self.dataset,
        }
