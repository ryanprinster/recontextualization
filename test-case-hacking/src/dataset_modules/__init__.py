"""Dataset implementations for expert iteration with recontextualization - Clean Architecture."""

from .base import (
    BaseContextHandler,
    BaseDataset,
    BaseEvaluator,
    BaseRolloutGenerator,
    EvaluationResult,
    ProcessedSample,
    Rollout,
    Sample,
)
from .code_selection import CodeSelectionDataset
from .factory import create_dataset
from .impossible_livecode import ImpossibleLiveCodeDataset
from .livecode_bench import LiveCodeDataset
from .rl_rewardhacking import RLRewardHackingDataset

__all__ = [
    "BaseDataset",
    "Sample",
    "ProcessedSample",
    "EvaluationResult",
    "Rollout",
    "BaseContextHandler",
    "BaseEvaluator",
    "BaseRolloutGenerator",
    "CodeSelectionDataset",
    "ImpossibleLiveCodeDataset",
    "LiveCodeDataset",
    "RLRewardHackingDataset",
    "create_dataset",
]
