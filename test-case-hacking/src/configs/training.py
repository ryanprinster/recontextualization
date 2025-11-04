"""
Simple training configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional

from .generation import GenerationConfig


@dataclass
class BaseTrainingConfig:
    """Base training configuration"""
    # Context for generation
    generation_context: str = "base"
    
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