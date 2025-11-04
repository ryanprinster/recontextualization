"""
Simple configuration classes for the recontextualization training system.
Clean separation of concerns with Hydra composition.
"""

from .generation import GenerationConfig
from .selection import BestOfNConfig
from .detection import DetectionConfig, RecontextualizationConfig
from .dataset import BaseDatasetConfig, CodeSelectionConfig
from .training import BaseTrainingConfig, OpenAITrainingConfig
from .output import OutputConfig
from .evaluation import EvaluationConfig

__all__ = [
    # Generation configs
    "GenerationConfig",
    
    # Selection configs
    "BestOfNConfig",
    
    # Detection configs
    "DetectionConfig",
    "RecontextualizationConfig",
    
    # Dataset configs
    "BaseDatasetConfig",
    "CodeSelectionConfig",
    
    # Training configs
    "BaseTrainingConfig",
    "OpenAITrainingConfig",
    
    # Output configs
    "OutputConfig",
    
    # Evaluation configs
    "EvaluationConfig",
]
