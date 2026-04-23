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
    EmailAssistantConfig,
    CustomerServiceConfig,
    SalesEnvironmentConfig,
    DataEntryConfig,
    MultipleChoiceConfig,
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
    from .email_assistant.dataset import EmailAssistantDataset
    from .customer_service.dataset import CustomerServiceDataset
    from .sales_environment.dataset import SalesEnvironmentDataset
    from .data_entry.dataset import DataEntryDataset
    from .multiple_choice.dataset import MultipleChoiceDataset

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
    elif isinstance(config, EmailAssistantConfig):
        return EmailAssistantDataset(
            data_path=config.data_path,
            names_path=config.names_path,
            email_ct=config.email_ct,
            num_samples=config.num_samples,
            final_email_category=config.final_email_category,
            domain=config.domain,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    elif isinstance(config, CustomerServiceConfig):
        return CustomerServiceDataset(
            data_path=config.data_path,
            customer_types=config.customer_types,
            simulator_model_name=config.simulator_model_name,
            max_turns=config.max_turns,
            num_samples=config.num_samples,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    elif isinstance(config, SalesEnvironmentConfig):
        return SalesEnvironmentDataset(
            data_path=config.data_path,
            num_samples=config.num_samples,
            max_turns=config.max_turns,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    elif isinstance(config, DataEntryConfig):
        return DataEntryDataset(
            problem_type=config.problem_type,
            data_path=config.data_path,
            tasks_per_sample=config.tasks_per_sample,
            num_samples=config.num_samples,
            max_turns=config.max_turns,
            per_cmd_timeout_sec=config.per_cmd_timeout_sec,
            total_timeout_sec=config.total_timeout_sec,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    elif isinstance(config, MultipleChoiceConfig):
        return MultipleChoiceDataset(
            dataset_mix=config.dataset_mix,
            num_samples_per_dataset=config.num_samples_per_dataset,
            data_path=config.data_path,
            train_ratio=config.train_ratio,
            random_seed=config.random_seed,
            max_prompt_tokens=config.max_prompt_tokens,
            tokenizer_model=config.tokenizer_model,
        )
    else:
        raise ValueError(f"Unsupported dataset config type: {type(config)}")