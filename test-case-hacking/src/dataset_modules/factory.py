"""
Dataset Factory
===============

Factory function to create datasets from configuration objects.
Follows the same pattern as the training factory.
"""

import logging

from ..configs.dataset import (
    BaseDatasetConfig,
    CodeSelectionConfig,
    LiveCodeBenchConfig,
    CodeGenerationConfig,
)
from .base import BaseDataset

logger = logging.getLogger(__name__)


def create_dataset(config: BaseDatasetConfig) -> BaseDataset:
    """
    Create dataset from configuration object.

    Args:
        config: Dataset configuration object

    Returns:
        Configured dataset instance

    Raises:
        ValueError: If config type is not supported
    """

    # Import here to avoid circular imports
    from .code_selection.dataset import CodeSelectionDataset
    from .livecode_bench.dataset import LiveCodeDataset
    from .code_generation.dataset import CodeGenerationDataset

    if isinstance(config, CodeSelectionConfig):
        return CodeSelectionDataset(
            include_test_cases=config.include_test_cases,
            use_incorrect_tests=config.use_incorrect_tests,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
        )
    elif isinstance(config, LiveCodeBenchConfig):
        return LiveCodeDataset(
            difficulties=config.difficulties,
            num_turns=config.num_turns,
            use_incorrect_tests=config.use_incorrect_tests,
            max_private_test_cases=config.max_private_test_cases,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
        )
    elif isinstance(config, CodeGenerationConfig):
        return CodeGenerationDataset(
            use_incorrect_tests=config.use_incorrect_tests,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
        )
    else:
        raise ValueError(f"Unsupported dataset config type: {type(config)}")