"""
Dataset configuration classes.
Clean separation of dataset parameters from instantiation logic.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BaseDatasetConfig:
    """Base configuration for all datasets"""

    train_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class CodeSelectionConfig(BaseDatasetConfig):
    """Code selection dataset configuration"""

    include_test_cases: bool = True
    use_incorrect_tests: bool = False


@dataclass
class LiveCodeBenchConfig(BaseDatasetConfig):
    """LiveCodeBench dataset configuration"""

    difficulties: Optional[List[str]] = None  # None = use all difficulties
    num_turns: int = 3,
    use_incorrect_tests: bool = False
    max_private_test_cases: int = 0


@dataclass
class CodeGenerationConfig(BaseDatasetConfig):
    """Code generation dataset configuration"""

    use_incorrect_tests: bool = False
