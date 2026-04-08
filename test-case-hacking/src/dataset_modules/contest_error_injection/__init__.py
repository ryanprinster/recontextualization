"""
Contest Error Injection dataset implementation for expert iteration with recontextualization.

This module provides components for evaluating code generation models on coding problems
with different contexts (standard, hack-resistant, pressured) to study reward hacking behavior.
"""

from .code_executor import CodeExecutor
from .contexts import ContestErrorInjectionContextHandler
from .dataset import ContestErrorInjectionDataset
from .evaluation import ContestErrorInjectionEvaluator
from .prompts import BASE_SYSTEM_PROMPT, PRESSURE_PROMPTS, SPECIAL_SYSTEM_PROMPTS
from .rollout_generator import ContestErrorInjectionRolloutGenerator
from .sample import ContestErrorInjectionSample

__all__ = [
    "ContestErrorInjectionSample",
    "ContestErrorInjectionContextHandler",
    "ContestErrorInjectionEvaluator",
    "ContestErrorInjectionRolloutGenerator",
    "ContestErrorInjectionDataset",
    "CodeExecutor",
    # Prompt templates for external use
    "BASE_SYSTEM_PROMPT",
    "PRESSURE_PROMPTS",
    "SPECIAL_SYSTEM_PROMPTS",
]
