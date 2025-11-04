"""
LiveCodeBench dataset implementation for expert iteration with recontextualization.

This module provides components for evaluating code generation models on coding problems
with different contexts (standard, hack-resistant, pressured) to study reward hacking behavior.
"""

from .code_executor import CodeExecutor
from .contexts import LiveCodeContextHandler
from .dataset import LiveCodeDataset
from .evaluation import LiveCodeEvaluator
from .llm_evaluator import LiveCodeLLMEvaluator
from .prompts import BASE_SYSTEM_PROMPT, PRESSURE_PROMPTS, SPECIAL_SYSTEM_PROMPTS
from .rollout_generator import LiveCodeRolloutGenerator
from .sample import LiveCodeSample

__all__ = [
    "LiveCodeSample",
    "LiveCodeContextHandler",
    "LiveCodeEvaluator",
    "LiveCodeLLMEvaluator",
    "LiveCodeRolloutGenerator",
    "LiveCodeDataset",
    "CodeExecutor",
    # Prompt templates for external use
    "BASE_SYSTEM_PROMPT",
    "PRESSURE_PROMPTS",
    "SPECIAL_SYSTEM_PROMPTS",
]
