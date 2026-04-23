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
from .contest_error_injection.dataset import ContestErrorInjectionDataset
from .email_assistant.dataset import EmailAssistantDataset
from .factory import create_dataset
from .impossible_livecode.dataset import ImpossibleLiveCodeDataset
from .mbpp.selection.dataset import MBPPSelectionDataset
from .mbpp.generation.dataset import MBPPGenerationDataset
from .rl_rewardhacking.dataset import RLRewardHackingDataset

__all__ = [
    "BaseDataset",
    "Sample",
    "ProcessedSample",
    "EvaluationResult",
    "Rollout",
    "BaseContextHandler",
    "BaseEvaluator",
    "BaseRolloutGenerator",
    "ContestErrorInjectionDataset",
    "EmailAssistantDataset",
    "ImpossibleLiveCodeDataset",
    "MBPPSelectionDataset",
    "MBPPGenerationDataset",
    "RLRewardHackingDataset",
    "create_dataset",
]
