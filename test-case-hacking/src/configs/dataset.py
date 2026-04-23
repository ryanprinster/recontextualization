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
    max_prompt_tokens: Optional[int] = 10000
    tokenizer_model: Optional[str] = "Qwen/Qwen3-8B"


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


@dataclass
class EmailAssistantConfig(BaseDatasetConfig):
    """Email Assistant dataset configuration (ported from reward-hacking-evals)."""

    data_path: str = "data/email_assistant/emails_updated.json"
    names_path: str = "data/email_assistant/filtered_names.txt"
    email_ct: int = 5
    num_samples: int = 50
    final_email_category: str = "email_assistant"
    domain: str = "applite"
    # Token-length filter defaults off: the initial ProcessedSample only has
    # the system + first email, which would underestimate the multi-turn prompt.
    max_prompt_tokens: Optional[int] = None
    tokenizer_model: Optional[str] = None


@dataclass
class CustomerServiceConfig(BaseDatasetConfig):
    """Customer Service dataset configuration (ported from reward-hacking-evals)."""

    data_path: str = "data/customer_service/customer_service_problems.txt"
    customer_types: Optional[List[str]] = None  # None = all 6 personas
    simulator_model_name: str = "gpt-4.1-mini"
    max_turns: int = 10
    num_samples: int = 30
    # Initial ProcessedSample is just the bot's system prompt, so filter is off.
    max_prompt_tokens: Optional[int] = None
    tokenizer_model: Optional[str] = None
