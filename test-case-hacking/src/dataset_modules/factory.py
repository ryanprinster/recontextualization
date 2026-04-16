"""
Dataset Factory
===============

Factory function to create datasets from configuration objects.
Follows the same pattern as the training factory.
"""

import logging

from ..configs.dataset import (
    BaseDatasetConfig,
    MBPPSelectionConfig,
    ContestErrorInjectionConfig,
    MBPPGenerationConfig,
    RLRewardHackingConfig,
    ImpossibleLiveCodeConfig,
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
    from .mbpp.selection.dataset import MBPPSelectionDataset
    from .contest_error_injection.dataset import ContestErrorInjectionDataset
    from .mbpp.generation.dataset import MBPPGenerationDataset
    from .rl_rewardhacking.dataset import RLRewardHackingDataset
    from .impossible_livecode.dataset import ImpossibleLiveCodeDataset

    if isinstance(config, MBPPSelectionConfig):
        return MBPPSelectionDataset(
            include_test_cases=config.include_test_cases,
            use_incorrect_tests=config.use_incorrect_tests,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    elif isinstance(config, ContestErrorInjectionConfig):
        return ContestErrorInjectionDataset(
            difficulties=config.difficulties,
            num_turns=config.num_turns,
            use_incorrect_tests=config.use_incorrect_tests,
            max_private_test_cases=config.max_private_test_cases,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    elif isinstance(config, RLRewardHackingConfig):
        return RLRewardHackingDataset(
            data_path=config.data_path,
            hint_type=config.hint_type,
            difficulties=config.difficulties,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    elif isinstance(config, ImpossibleLiveCodeConfig):
        return ImpossibleLiveCodeDataset(
            test_split=config.test_split,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            context_suffix_override=config.context_suffix_override,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    elif isinstance(config, MBPPGenerationConfig):
        return MBPPGenerationDataset(
            use_incorrect_tests=config.use_incorrect_tests,
            base_suffix=config.base_suffix,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    else:
        raise ValueError(f"Unsupported dataset config type: {type(config)}")