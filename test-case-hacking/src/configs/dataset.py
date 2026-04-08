"""
Dataset configuration classes.
Clean separation of dataset parameters from instantiation logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BaseDatasetConfig:
    """Base configuration for all datasets"""

    train_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class MBPPSelectionConfig(BaseDatasetConfig):
    """MBPP selection dataset configuration"""

    include_test_cases: bool = True
    use_incorrect_tests: bool = False


@dataclass
class ContestErrorInjectionConfig(BaseDatasetConfig):
    """Contest error injection dataset configuration"""

    difficulties: Optional[List[str]] = None  # None = use all difficulties
    num_turns: int = 3
    use_incorrect_tests: bool = False
    max_private_test_cases: int = 0


@dataclass
class MBPPGenerationConfig(BaseDatasetConfig):
    """MBPP generation dataset configuration"""

    use_incorrect_tests: bool = False
    base_suffix: Optional[str] = None


@dataclass
class RLRewardHackingConfig(BaseDatasetConfig):
    """RL Reward Hacking dataset configuration"""

    hint_type: str = "SimpleOverwriteTests"
    difficulties: Optional[List[str]] = None
    data_path: str = "data/rl_rewardhacking/leetcode_train_medhard_filtered.jsonl"


@dataclass
class ImpossibleLiveCodeConfig(BaseDatasetConfig):
    """Impossible LiveCode dataset configuration"""

    test_split: str = "oneoff"  # "original", "oneoff", or "conflicting"
    context_suffix_override: Optional[Dict[str, str]] = None  # e.g. {"hack": "new prompt text"}
