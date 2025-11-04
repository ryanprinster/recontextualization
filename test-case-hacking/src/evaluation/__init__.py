"""
Evaluation Module
================

Standalone evaluation system for models and rollouts.

This module provides:
- Standalone evaluation of models across contexts
- Evaluation of pregenerated rollouts
- Rich metrics and reporting capabilities
- Integration with existing dataset evaluation methods
- Hydra configuration support
"""

from .evaluator import Evaluator
from .metrics import EvaluationMetrics, EvaluationReport

__all__ = [
    "Evaluator",
    "EvaluationMetrics",
    "EvaluationReport"
]
