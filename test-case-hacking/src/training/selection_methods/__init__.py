"""
Selection Methods Package
========================

Improved selection methods with:
- Clean base class with common functionality and default behavior
- Generator-controlled rollout generation
- Best-of-N selection for expert iteration
- Pull-based generation patterns
"""

from .base import BaseRolloutSelector
from .best_of_n import BestOfNRolloutSelector

__all__ = [
    "BaseRolloutSelector",
    "BestOfNRolloutSelector",
]
