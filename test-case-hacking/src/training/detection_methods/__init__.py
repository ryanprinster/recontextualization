"""
Detection Methods Package
========================

Improved detection methods with:
- Clean base class with common detection logic
- Perfect dependency injection (only what each detector needs)
- Support for recontextualization (switching contexts for detected rollouts)
- Direct rollout handling without index manipulation
"""

from .base import BaseDetectionProcessor
from .recontextualization import RecontextualizationProcessor

__all__ = [
    "BaseDetectionProcessor",
    "RecontextualizationProcessor"
]
