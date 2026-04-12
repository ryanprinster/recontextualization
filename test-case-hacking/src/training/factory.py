"""
Simple Factory for Training System
==================================

Clean factory that creates trainers with naturally integrated workflows.
Trainers handle their own workflow creation internally.
"""

import logging
from typing import Optional

from ..configs import DetectionConfig
from ..configs.selection import SelectionConfig
from ..configs.evaluation import EvaluationConfig
from ..configs.training import (
    BaseTrainingConfig,
    GRPOTrainingConfig,
    LocalTrainingConfig,
    OpenAITrainingConfig,
    TinkerGRPOTrainingConfig,
    TinkerTrainingConfig,
)
from ..dataset_modules.base import BaseDataset
from ..evaluation.evaluator import Evaluator
from ..models.base import BaseModel

logger = logging.getLogger(__name__)


def create_trainer(
    config: BaseTrainingConfig,
    dataset: BaseDataset,
    model: BaseModel,
    output_dir: str,
    selection_config: Optional[SelectionConfig] = None,
    detection_config: Optional[DetectionConfig] = None,
    evaluation_config: Optional[EvaluationConfig] = None,
):
    """
    Create trainer with integrated workflow.
    Trainers handle workflow creation internally for natural coupling.
    """

    # Import here to avoid circular imports
    from .openai_trainer import OpenAITrainer
    from .local_trainer import LocalTrainer
    from .grpo_trainer import LocalGRPOTrainer
    from .tinker_trainer import TinkerTrainer
    from .tinker_grpo_trainer import TinkerGRPOTrainer

    # Evaluation config will be passed to trainer for internal evaluator creation

    # Create trainer - workflow is created internally
    if isinstance(config, OpenAITrainingConfig):
        return OpenAITrainer(
            config=config,
            dataset=dataset,
            model=model,
            output_dir=output_dir,
            selection_config=selection_config,
            detection_config=detection_config,
            evaluation_config=evaluation_config,
        )
    elif isinstance(config, GRPOTrainingConfig):
        return LocalGRPOTrainer(
            config=config,
            dataset=dataset,
            model=model,
            output_dir=output_dir,
            selection_config=selection_config,
            detection_config=detection_config,
            evaluation_config=evaluation_config,
        )
    elif isinstance(config, LocalTrainingConfig):
        return LocalTrainer(
            config=config,
            dataset=dataset,
            model=model,
            output_dir=output_dir,
            selection_config=selection_config,
            detection_config=detection_config,
            evaluation_config=evaluation_config,
        )
    elif isinstance(config, TinkerGRPOTrainingConfig):
        return TinkerGRPOTrainer(
            config=config,
            dataset=dataset,
            model=model,
            output_dir=output_dir,
            selection_config=selection_config,
            detection_config=detection_config,
            evaluation_config=evaluation_config,
        )
    elif isinstance(config, TinkerTrainingConfig):
        return TinkerTrainer(
            config=config,
            dataset=dataset,
            model=model,
            output_dir=output_dir,
            selection_config=selection_config,
            detection_config=detection_config,
            evaluation_config=evaluation_config,
        )
    else:
        raise ValueError(f"Unsupported training config type: {type(config)}")
