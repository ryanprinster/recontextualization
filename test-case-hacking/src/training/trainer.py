"""
Trainer Base Class
==================

Trainer that integrates with the workflow system.
Key features:
- Implements Generator protocol for rollout generation
- Uses Workflow for processing
- Clean separation of concerns
- Simplified context management
"""

import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

from ..configs import DetectionConfig, RecontextualizationConfig
from ..configs.selection import BestOfNConfig, SelectionConfig
from ..configs.training import BaseTrainingConfig
from ..dataset_modules.base import BaseDataset, Sample
from ..evaluation.evaluator import Evaluator
from ..evaluation.metrics import EvaluationReport
from ..generation import RolloutGenerator
from ..models.base import BaseModel
from ..storage import use_cache
from .data_structures import SampleRollouts
from .detection_methods import (
    BaseDetectionProcessor,
    RecontextualizationProcessor,
)
from .selection_methods import (
    BaseRolloutSelector,
    BestOfNRolloutSelector,
)

logger = logging.getLogger(__name__)


# ================================
# UNIFIED TRAINING RESULT & STATE
# ================================

@dataclass
class TrainingResult:
    """Comprehensive training result - single source of truth"""
    # Core status
    status: Literal["none", "running", "completed", "failed"]
    error: Optional[str] = None
    
    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Model reference
    model_ref: Optional[str] = None
    
    # Training metrics
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Backend-specific data (e.g., OpenAI job_id, HF checkpoints, etc.)
    backend_data: Dict[str, Any] = field(default_factory=dict)


class Trainer(ABC):
    """
    Base trainer class with integrated workflow creation.

    Key features:
    - Workflow creation is part of trainer initialization
    - Natural coupling between trainer and workflow
    - Clean separation between generation and training logic
    - Subclasses only need to implement backend-specific methods
    """

    def __init__(
        self,
        config: BaseTrainingConfig,
        dataset: BaseDataset,
        model: BaseModel,
        output_dir: str,
        selection_config=None,
        detection_config=None,
        evaluation_config=None,
    ):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.output_dir = output_dir
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Initialize training seeds for selection/detection randomness
        if config.training_seed is not None:
            random.seed(config.training_seed)
            logger.debug(f"Initialized training seed: {config.training_seed}")

        # Create unified rollout generator
        self.rollout_generator = RolloutGenerator(model, dataset)

        # Create generation function with pre-bound context and config
        self.generate_fn = partial(
            self.rollout_generator.generate_rollouts,
            context=self.config.generation_context,
            generation_config=self.config.training_generation,
        )

        # Create workflow components with generation function
        self.selector = self._create_selector(selection_config)
        self.detector = self._create_detector(detection_config)
        
        # Model evaluator for pre/post training evaluation (created internally)
        self.model_evaluator = Evaluator(
            dataset=dataset,
            config=evaluation_config,
            output_dir=output_dir
        )
        
        # Load or initialize training result
        self.result = self._load_result() or TrainingResult(
            status="none"
        )

    def _create_selector(
        self, selection_config: Optional[SelectionConfig]
    ) -> BaseRolloutSelector:
        """Create selection component"""
        if selection_config is None:
            return BaseRolloutSelector(self.generate_fn)
        elif isinstance(selection_config, BestOfNConfig):
            return BestOfNRolloutSelector(self.generate_fn)
        else:
            raise ValueError(f"Unsupported selection config: {type(selection_config)}")

    def _create_detector(
        self, detection_config: Optional[DetectionConfig]
    ) -> Optional[BaseDetectionProcessor]:
        """Create detection component"""
        if detection_config is None:
            return None
        elif isinstance(detection_config, RecontextualizationConfig):
            return RecontextualizationProcessor(detection_config, self.dataset)
        else:
            raise ValueError(f"Unsupported detection config: {type(detection_config)}")

    # Rollout generation is handled through pre-configured partial function
    # No need for generator protocol implementation

    # ================================
    # WORKFLOW PROCESSING (replaces separate Workflow class)
    # ================================

    def _process_samples(
        self, samples: List[Sample]
    ) -> Tuple[List[SampleRollouts], Dict[str, Any]]:
        """Process samples through selection and detection pipeline"""

        # Create initial structure
        sample_rollouts_list = [
            SampleRollouts(sample_index=i, sample=sample)
            for i, sample in enumerate(samples)
        ]

        metrics = {}

        # Step 1: Selection (selector manages its own generation)
        # Use cache for rollout generation during selection to speed up training
        with use_cache():
            sample_rollouts_list, selection_metrics = self.selector.select_rollouts(
                sample_rollouts_list
            )
        metrics["selection"] = selection_metrics

        # Evaluate and save generated rollouts (after selection generates them)
        generated_rollouts_grouped = [sr.rollouts for sr in sample_rollouts_list]
        self.model_evaluator.evaluate_rollouts(
            grouped_rollouts=generated_rollouts_grouped,
            save_rollout_messages=False,
            save_subdir="training_data/generated",
        )
        total_generated = sum(len(group) for group in generated_rollouts_grouped)
        self.logger.info(f"Evaluated {total_generated} generated rollouts")

        # Evaluate and save selected rollouts (after selection)
        selected_rollouts_grouped = [
            [sr.selected_rollout] for sr in sample_rollouts_list
            if sr.has_selection and sr.selected_rollout is not None
        ]
        if selected_rollouts_grouped:
            self.model_evaluator.evaluate_rollouts(
                grouped_rollouts=selected_rollouts_grouped,
                save_rollout_messages=False,
                save_subdir="training_data/selected",
            )
            self.logger.info(f"Evaluated {len(selected_rollouts_grouped)} selected rollouts")

        # Step 2: Detection and Processing
        if self.detector:
            sample_rollouts_list, detection_metrics = self.detector.detect_and_process(
                sample_rollouts_list
            )
            metrics["detection"] = detection_metrics
        else:
            metrics["detection"] = {"method": "none"}

        # Extract final rollouts
        final_rollouts_grouped = [
            [sr.selected_rollout] for sr in sample_rollouts_list
            if sr.has_selection and sr.selected_rollout is not None
        ]
        # Evaluate and save final rollouts (after full pipeline)
        if final_rollouts_grouped:
            self.model_evaluator.evaluate_rollouts(
                grouped_rollouts=final_rollouts_grouped,
                save_rollout_messages=False,
                save_subdir="training_data/final",
            )
            self.logger.info(f"Evaluated {len(final_rollouts_grouped)} final rollouts")

        # Add summary metrics
        total_rollouts = sum(len(sr.rollouts) for sr in sample_rollouts_list)
        final_rollouts_count = sum(len(group) for group in final_rollouts_grouped)
        metrics["summary"] = {
            "total_samples": len(sample_rollouts_list),
            "total_rollouts": total_rollouts,
            "final_rollouts": final_rollouts_count,
            "avg_rollouts_per_sample": total_rollouts / len(sample_rollouts_list)
            if sample_rollouts_list
            else 0,
        }

        return sample_rollouts_list, metrics

    # ================================
    # MAIN TRAINING METHODS
    # ================================

    @abstractmethod
    def train(self) -> TrainingResult:
        """
        Main training method. Subclasses implement their complete training logic.
        
        Returns:
            TrainingResult with training data and metrics
        """
        pass
    
    def resume(self) -> TrainingResult:
        """
        Resume training from existing state. Default implementation fails explicitly.
        Subclasses must override for explicit resume logic.
        
        Returns:
            TrainingResult with resume status and metrics
        """
        return TrainingResult(
            status="failed",
            error=f"{self.__class__.__name__} does not support resume operation. Only train() is available."
        )
    
    def get_status(self) -> str:
        """Get simple training status string"""
        return self.result.status
    
    def _save_result(self) -> None:
        """Save current result to disk"""
        state_file = Path(self.output_dir) / "training_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(asdict(self.result), f, indent=2)
    
    def _load_result(self) -> Optional[TrainingResult]:
        """Load training result from state"""
        state_file = Path(self.output_dir) / "training_state.json"
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            return TrainingResult(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load training state from {state_file}: {e}")
            return None

    # ================================
    # UTILITY METHODS
    # ================================

    def evaluate_model_performance(self, stage: str, save_rollout_messages: Optional[bool] = None, save_subdir: str = "evaluation", model: Optional[BaseModel] = None) -> Optional[Dict[str, EvaluationReport]]:
        """
        Helper method for model evaluation that subclasses can use.
        Evaluation results are automatically saved to files by the evaluator.

        Args:
            stage: Stage name (e.g., "pre_training", "post_training")
            save_subdir: Subdirectory to save results to
            model: Optional model override (defaults to self.model)
        """
        evaluation_samples = self.dataset.get_val_samples()
        self.logger.info(f"Evaluating model performance at {stage}...")

        reports = self.model_evaluator.evaluate_with_model(
            model=model or self.model,
            samples=evaluation_samples,
            save_rollout_messages=save_rollout_messages,
            save_subdir=save_subdir,
        )
        
        if reports:
            total_rollouts = sum(report.total_rollouts for report in reports.values())
            self.logger.info(f"{stage} evaluation completed: {total_rollouts} rollouts across {len(reports)} contexts")
        else:
            self.logger.info(f"{stage} evaluation completed: no results")
        
        return reports

    def get_training_context(self) -> str:
        """Get the context used for training generation"""
        return self.config.generation_context

