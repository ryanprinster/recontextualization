"""
OpenAI Trainer
==============

Ultra-simple, reliable OpenAI trainer with job recovery.
Fail-fast approach - no retry logic, just clean error handling.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..configs.training import OpenAITrainingConfig
from ..dataset_modules.base import BaseDataset, Rollout
from ..models.openai_model import OpenAIModel
from ..storage import use_cache
from .trainer import Trainer, TrainingResult

logger = logging.getLogger(__name__)

class OpenAITrainer(Trainer):
    """
    Simplified OpenAI trainer with robust recovery logic.
    
    Key improvements:
    - Single resume() method handles all recovery scenarios
    - Automatic job status synchronization
    - Clear error handling and logging
    - Fail-fast approach with proper state management
    
    Usage:
    - train(): Start fresh training (data generation + OpenAI job submission)
    - resume(): Check current state and continue from where we left off
    - get_status(): Get current status with live OpenAI job checking
    """

    def __init__(
        self,
        config: OpenAITrainingConfig,
        dataset: BaseDataset,
        model: OpenAIModel,  # Type hint for expected model type
        output_dir: str,
        selection_config=None,
        detection_config=None,
        evaluation_config=None,
    ):
        super().__init__(config, dataset, model, output_dir, selection_config, detection_config, evaluation_config)
        self.client = OpenAI()

    def train(self) -> TrainingResult:
        """Start fresh training"""
        logger.info("Starting OpenAI training...")

        self.result.status = "running"
        self.result.started_at = datetime.now().isoformat()
        self._save_result()

        try:
            with use_cache():
                logger.info("=== PRE-TRAINING EVALUATION ===")
                self.evaluate_model_performance("pre_training")

                logger.info("=== TRAINING DATA GENERATION ===")
                training_samples = self.dataset.get_train_samples(self.config.max_train_samples)
                sample_rollouts_list, _ = self._process_samples(training_samples)
                training_rollouts = [sr.selected_rollout for sr in sample_rollouts_list if sr.has_selection]

            training_file_path = self._create_training_file(training_rollouts)
            training_file_id = self._upload_training_file(training_file_path)

            # Update result with training data info
            self.result.backend_data["openai_file_id"] = training_file_id
            self._save_result()
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            self.result.status = "failed"
            self.result.error = str(e)
            self.result.completed_at = datetime.now().isoformat()
            self._save_result()
            return self.result

        return self._start_training_job(training_file_id)

    def _start_training_job(self, training_file_id: str) -> TrainingResult:
        """Start OpenAI fine-tuning job"""
        try:
            job_id = self._create_fine_tuning_job(training_file_id)
            self.result.backend_data["openai_job_id"] = job_id
            self._save_result()
            logger.info(f"Started OpenAI job {job_id}")
            return self.result
        except Exception as e:
            logger.error(f"Failed to start training job: {e}")
            self.result.status = "failed"
            self.result.error = str(e)
            self.result.completed_at = datetime.now().isoformat()
            self._save_result()
            return self.result

    def resume(self) -> TrainingResult:
        """Resume training from current state"""
        logger.info(f"Resuming training with status: {self.result.status}")
            
        if self.result.status == "failed":
            return self._start_training_job(self.result.backend_data["openai_file_id"])
            
        elif self.result.status == "running":
            job_id = self.result.backend_data["openai_job_id"]
            
            # Check OpenAI job status directly
            try:
                job_status = self._check_job_status(job_id)
                openai_status = job_status.get('status')
                logger.info(f"OpenAI job {job_id} status: {openai_status}")
                
                if openai_status == 'succeeded':
                    fine_tuned_model = job_status.get('fine_tuned_model')

                    self.result.status = "completed"
                    self.result.model_ref = fine_tuned_model
                    self.result.completed_at = datetime.now().isoformat()
                    self._save_result()
                        
                elif openai_status == 'failed':
                    error = job_status.get('error', 'OpenAI job failed')
                    self.result.status = "failed"
                    self.result.error = str(error)
                    self.result.completed_at = datetime.now().isoformat()
                    self._save_result()
                    
                return self.result
                
            except Exception as e:
                logger.error(f"Failed to check job status: {e}")
                self.result.error = f"Failed to check job status: {e}"
                self._save_result()
                return self.result
            
        if self.result.status == "completed":
            logger.info("=== POST-TRAINING EVALUATION ===")
            self.model.model_ref = self.result.model_ref
            self.evaluate_model_performance("post_training")
            self._save_result()
            return self.result

    def _create_training_file(self, training_rollouts: List[Rollout]) -> Path:
        """Create JSONL training file for OpenAI"""
        output_path = Path(self.output_dir) / "training_data.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for rollout in training_rollouts:
                f.write(json.dumps({"messages": rollout.messages}) + "\n")

        logger.info(f"Created training file: {len(training_rollouts)} examples")
        return output_path

    def _upload_training_file(self, file_path: Path) -> str:
        """Upload training file to OpenAI"""
        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose="fine-tune")
        
        logger.info(f"Uploaded training file: {response.id}")
        return response.id

    def _create_fine_tuning_job(self, training_file_id: str) -> str:
        """Create OpenAI fine-tuning job"""
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=self.model.name,
            hyperparameters={
                "n_epochs": self.config.n_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate_multiplier": self.config.learning_rate_multiplier,
            },
            seed=self.config.training_seed,
        )

        logger.info(f"Created fine-tuning job: {response.id}")
        return response.id
    
    def _check_job_status(self, job_id: str) -> dict:
        """Check OpenAI job status"""
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            return {
                'status': job.status,
                'fine_tuned_model': job.fine_tuned_model,
                'error': getattr(job, 'error', None)
            }
        except Exception as e:
            return {'status': 'check_failed', 'error': str(e)}