"""
Training System
===============

Clean, modular training system with integrated workflows.

Usage:
    from training import create_trainer
    
    trainer = create_trainer(
        config=OpenAITrainingConfig(...),
        dataset=my_dataset,
        model=my_model,
        output_dir="./output",
        selection_config=BestOfNConfig(),
        detection_config=RecontextualizationConfig(target_context="standard", ...)
    )
"""

# Main factory method
from .factory import create_trainer

# Configuration classes (for convenience)
from ..configs import RecontextualizationConfig

# Core trainers
from .trainer import Trainer, TrainingResult
from .openai_trainer import OpenAITrainer
from .local_trainer import LocalTrainer


__all__ = [
    # Main entry point
    "create_trainer",

    # Configuration
    "RecontextualizationConfig",

    # Trainers
    "Trainer",
    "TrainingResult",
    "OpenAITrainer",
    "LocalTrainer",
]
