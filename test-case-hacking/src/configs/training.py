"""
Simple training configuration classes.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .generation import GenerationConfig


@dataclass
class BaseTrainingConfig:
    """Base training configuration"""
    # Context for generation
    generation_context: str = "standard"

    # Data parameters
    max_train_samples: Optional[int] = None

    # Generation config for training
    training_generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Training randomness seed (separate from dataset split seed)
    training_seed: Optional[int] = None

@dataclass
class OpenAITrainingConfig(BaseTrainingConfig):
    """OpenAI training configuration"""
    # OpenAI fine-tuning parameters
    n_epochs: int = 3
    batch_size: str = "auto"  # "auto" or integer
    learning_rate_multiplier: str = "auto"  # "auto" or float
    validation_split: float = 0.1


@dataclass
class LocalTrainingConfig(BaseTrainingConfig):
    """
    Training config for local models (e.g. Qwen3-8B) using TRL SFTTrainer.

    Generation uses whatever model is passed to the trainer (e.g. a vLLM-backed
    model). Training loads model weights fresh from `model_name_or_path` via
    HuggingFace Transformers so the two phases don't compete for GPU memory.
    """

    # Path/name of the model to load for SFT training.
    # Set to a local checkpoint directory for subsequent expert-iteration rounds.
    model_name_or_path: str = "Qwen/Qwen3-8B"

    # Expert-iteration filter: only train on rollouts whose score meets this bar.
    # 0.0 keeps all selected rollouts; 1.0 keeps only perfectly correct ones.
    min_score_threshold: float = 0.0

    # ---- LoRA ----
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Module names to apply LoRA to. None lets PEFT auto-detect linear layers.
    lora_target_modules: Optional[List[str]] = None

    # ---- TRL SFTTrainer hyperparameters ----
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    # Max tokens per training example (longer sequences are truncated).
    max_seq_length: int = 8192