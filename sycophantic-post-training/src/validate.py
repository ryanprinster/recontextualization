"""Configuration validation and parsing for expert iteration experiments."""

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import OmegaConf  # Add this import


@dataclass
class SFTConfig:
    """Configuration for SFT training phases."""

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_dropout: float = 0.1
    lora_bias: str = "none"

    # Training arguments
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    learning_rate: float = 2e-4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "no"
    save_total_limit: int = 2
    load_best_model_at_end: bool = False
    report_to: str = "none"
    remove_unused_columns: bool = False
    max_seq_length: int = 2048
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Model settings
    merge_adapter: bool = True

    @classmethod
    def from_yaml(cls, config_path: str) -> "SFTConfig":
        """Load SFT configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_lora_config(self) -> Dict[str, Any]:
        """Convert to LoRA config dictionary for the finetune script."""
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.lora_target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.lora_bias,
        }

    def to_training_args(self) -> Dict[str, Any]:
        """Convert to training arguments dictionary for the finetune script."""
        return {
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "learning_rate": self.learning_rate,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "eval_strategy": self.eval_strategy,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "report_to": self.report_to,
            "remove_unused_columns": self.remove_unused_columns,
            "max_seq_length": self.max_seq_length,
            "optim": self.optim,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass
class ExpertIterationConfig:
    """Configuration for expert iteration experiments."""

    # Basic configuration
    base_model: str = "Qwen/Qwen3-8B-Base"
    experiment_name: str = None
    experiment_dir_name: Optional[str] = None
    N_completions: int = 10
    eval_N_completions: int = 1
    dataset_name: Optional[str] = None
    data_dir: str = "data"
    sft_dataset_name: Optional[str] = None
    expert_iteration_dataset_name: Optional[str] = None
    eval_dataset_name: Optional[str] = None
    expert_iteration_dataset_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    mix_in_helpful: float = 0.0
    helpful_dataset_name: Optional[str] = "Anthropic/hh-rlhf"

    self_distillation: bool = False

    no_cache: bool = False
    no_eval_cache: bool = False

    iter_to_resume_from: int = None
    resume_from_checkpoint: bool = False
    resume_from_experiment_dir: str = None
    # Open source model configuration
    best_of_n_sft_config_path: Optional[str] = None

    # Loaded SFT configs (populated after loading)
    best_of_n_sft_config: Optional[SFTConfig] = field(default=None, init=False)

    # Skip flags
    skip_expert_iteration_step: bool = False

    # Paths
    judge_template_path: Optional[str] = None
    ground_truth_judge_template_path: Optional[str] = None
    sycophancy_judge_template_path: Optional[str] = None
    quality_judge_template_path: Optional[str] = None

    expert_iteration_config_path: str = "configs/expert_iteration_sft_config.yaml"
    expert_iteration_model_path: Optional[str] = (
        None  # Path to pre-trained expert iteration model
    )
    N_generations_judged_path: Optional[str] = None  # Path to judged N generations

    quality_alpha: float = 1.0
    sycophancy_alpha: float = 0.25

    # Seeds and suffixes
    seed: int = 1
    data_generation_seed: int = 42
    test_suffix: str = "test"

    # Generation parameters
    generation_temperature: float = 1.0
    eval_generation_temperature: float = 1.0
    generation_max_tokens: int = 512

    # Judge parameters
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.1
    judge_max_tokens: int = 32

    # Recontextualization parameters
    do_recontextualization: bool = False
    recontextualized_system_message: bool = False
    recontextualized_phrase: Optional[str] = None
    user_phrase_to_replace_with_recontextualized: Optional[str] = None
    add_recontextualized_phrase_as_user_message_suffix: bool = False
    use_cheat_method_for_recontextualization: bool = False
    system_message_after_user_message_recontextualization: bool = False
    icl_prompts_recontextualization: List[str] = None
    icl_responses_recontextualization: List[str] = None

    # Prompt modification for generation
    do_prompt_modification_generation: bool = False
    modify_system_message_generation: bool = False
    modified_phrase_generation: Optional[str] = None
    user_phrase_to_replace_with_modified_phrase_generation: Optional[str] = None
    add_modified_phrase_as_user_message_suffix_generation: bool = False
    use_cheat_method_for_prompt_modification_generation: bool = False
    system_message_after_user_message_generation: bool = False
    icl_prompts_generation: List[str] = None
    icl_responses_generation: List[str] = None

    # Regularization
    regularization_data_type: Optional[str] = None
    regularization_alpha: float = 0.1

    # Other flags
    debug: bool = False
    eval_sft_and_base: bool = False

    # Dataset truncation for testing
    truncate_bon_dataset: Optional[int] = None
    truncate_eval_dataset: Optional[int] = None

    eval_at_initial: bool = True  # Evaluate at the beginning
    eval_at_iterations: bool = True  # Evaluate at each iteration
    eval_at_final: bool = False  # Evaluate at the end

    # Number of expert iteration rounds
    num_expert_iteration_iters: int = 1

    def __post_init__(self):
        """Post-initialization processing to set computed fields and defaults."""
        from datetime import datetime

        # Generate timestamp for paths (if needed)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set debug truncation defaults
        if self.N_generations_judged_path:
            if self.num_expert_iteration_iters > 1:
                raise ValueError(
                    "We must only specify a judged generaitons path if we're doing 1 expert iteration, otherwise the data should be online"
                )
        if self.judge_template_path:
            print("""
                WARNING: you've provided paths to a training judge template path (judge_template_path),
                We will be overriding the training judge and using the individual quality/sycophancy judge templates (summing their score to get training reward). 
                If this is not intended don't provide quality/sycophancy templates""")

        if not (
            self.sycophancy_judge_template_path and self.quality_judge_template_path
        ):
            raise ValueError(
                "You need to provide a path to the sycophancy judge prompt and the quality judge prompt."
            )

        if not self.ground_truth_judge_template_path:
            raise ValueError("Must provide ground truth template path")

        if not self.dataset_name:
            raise ValueError("Must provide dataset name")

        if self.best_of_n_sft_config_path and not self.best_of_n_sft_config:
            self.best_of_n_sft_config = SFTConfig.from_yaml(
                self.best_of_n_sft_config_path
            )

        if not self.expert_iteration_dataset_path:
            self.expert_iteration_dataset_path = (
                f"{self.data_dir}/expert_iteration_messages.jsonl"
            )

        if not self.eval_dataset_path:
            self.eval_dataset_path = f"{self.data_dir}/eval_messages.jsonl"

        if self.eval_dataset_path:
            self.eval_dataset_path = os.path.expanduser(self.eval_dataset_path)

        if self.expert_iteration_dataset_path:
            self.expert_iteration_dataset_path = os.path.expanduser(
                self.expert_iteration_dataset_path
            )

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExpertIterationConfig":
        """Load configuration from a YAML file."""
        print("Loading from yaml")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle embedded SFT configs if present
        best_of_n_sft_config_dict = config_dict.pop("best_of_n_sft_config", None)

        # Create the config instance (SFT configs will be loaded in __post_init__)
        config = cls(**config_dict)
        print("Created instance of ExpertIterationConfig")

        if best_of_n_sft_config_dict:
            config.best_of_n_sft_config = SFTConfig(**best_of_n_sft_config_dict)

        return config

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExpertIterationConfig":
        """Create configuration from argparse arguments."""
        # Convert argparse Namespace to dict, filtering out None values
        args_dict = {k: v for k, v in vars(args).items() if v is not None}

        # Remove config_path if it exists (not part of the dataclass)
        args_dict.pop("config_path", None)

        # Convert underscores in argument names to match dataclass fields
        normalized_dict = {}
        for key, value in args_dict.items():
            # Replace hyphens with underscores (argparse uses hyphens, dataclass uses underscores)
            normalized_key = key.replace("-", "_")
            normalized_dict[normalized_key] = value

        return cls(**normalized_dict)

    def to_yaml(self, output_path: str):
        """Save configuration to a YAML file."""
        with open(output_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def validate(self):
        """Validate the configuration."""
        errors = []

        # Required fields
        if self.N_completions is None:
            errors.append("N_completions is required")

        # Validate recontextualization settings
        if self.do_recontextualization:
            if not self.recontextualized_phrase:
                errors.append(
                    "recontextualized_phrase is required when do_recontextualization is True"
                )

        # Validate prompt modification settings
        if self.do_prompt_modification_generation:
            if not self.modified_phrase_generation:
                errors.append(
                    "modified_phrase_generation is required when do_prompt_modification_generation is True"
                )

        # Validate open source configuration

        # Check for either embedded config or config path for best-of-n SFT
        if not self.best_of_n_sft_config and not self.best_of_n_sft_config_path:
            errors.append(
                "Either best_of_n_sft_config or best_of_n_sft_config_path is required"
            )
        elif (
            self.best_of_n_sft_config_path
            and not self.best_of_n_sft_config
            and not Path(self.best_of_n_sft_config_path).exists()
        ):
            errors.append(
                f"best_of_n_sft_config_path does not exist: {self.best_of_n_sft_config_path}"
            )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


def parse_arguments(argv=None):
    """Parse command line arguments for configuration file."""
    parser = argparse.ArgumentParser(
        description="Expert Iteration Experiment Configuration"
    )

    # Config file argument
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # Capture any additional overrides
    parser.add_argument(
        "overrides", nargs="*", help="Config overrides (e.g., seed=42 lr=0.001)"
    )

    args = parser.parse_args(argv)

    # Load configuration from YAML file using OmegaConf
    config_dict = OmegaConf.load(args.config_path)

    # Apply CLI overrides
    if args.overrides:
        cli_overrides = OmegaConf.from_dotlist(args.overrides)
        config_dict = OmegaConf.merge(config_dict, cli_overrides)

    # Convert to regular dict for your dataclass
    config_dict = OmegaConf.to_container(config_dict, resolve=True)

    # Handle embedded SFT configs if present
    best_of_n_sft_config_dict = config_dict.pop("best_of_n_sft_config", None)

    # Create the config instance
    if "config" in config_dict:
        print("Found config in config_dict: ", config_dict["config"])
    config = ExpertIterationConfig(**config_dict)

    # If embedded configs were provided, create SFTConfig objects from them
    if best_of_n_sft_config_dict:
        config.best_of_n_sft_config = SFTConfig(**best_of_n_sft_config_dict)

    # Validate configuration
    config.validate()

    return config


if __name__ == "__main__":
    # Example usage
    config = parse_arguments()
    print("Configuration loaded successfully:")
    print(yaml.dump(asdict(config), default_flow_style=False))
