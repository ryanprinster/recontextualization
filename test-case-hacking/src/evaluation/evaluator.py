"""
Clean Evaluator
===============

Simple, focused evaluator for models and rollouts with clear separation of concerns.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..configs.evaluation import EvaluationConfig
from ..dataset_modules.base import BaseDataset, Rollout, Sample
from ..generation import RolloutGenerator
from ..models.base import BaseModel
from ..storage import RolloutStorage
from .metrics import EvaluationMetrics, EvaluationReport

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Clean, focused evaluator with two main responsibilities:
    1. Evaluate models by generating rollouts across contexts
    2. Evaluate existing rollouts and compute metrics
    
    File saving is optional and controlled by configuration.
    """

    def __init__(self, dataset: BaseDataset, config: EvaluationConfig, output_dir: str = ""):
        """Initialize the evaluator.
        
        Args:
            dataset: Dataset to evaluate on
            config: Evaluation configuration 
            output_dir: Base output directory for saving results (default: "")
        """
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir
        self.metrics_calculator = EvaluationMetrics()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Only create storage if we need to save results
        self._storage: Optional[RolloutStorage] = None

    @property
    def storage(self) -> RolloutStorage:
        """Lazy-load storage only when needed"""
        if self._storage is None:
            self._storage = RolloutStorage()
        return self._storage

    def evaluate_with_model(
        self,
        model: BaseModel,
        samples: List[Sample],
        contexts: Optional[List[str]] = None,
        save_results: Optional[bool] = None,
        save_rollout_messages: Optional[bool] = None,
        save_subdir: str = "",
    ) -> Dict[str, EvaluationReport]:
        """
        Evaluate a model by generating rollouts across contexts.
        Each context is evaluated and saved independently.
        
        Args:
            model: Model to evaluate
            samples: Samples to evaluate on
            contexts: Contexts to evaluate (uses config.contexts if None)
            save_results: Whether to save results (uses config if None)
            save_rollout_messages: Whether to include messages in saved rollouts (uses config if None)
            save_subdir: Subdirectory for saving. Empty string uses context names directly
            
        Returns:
            Dictionary mapping context names to their evaluation reports
        """
        contexts = contexts or self.config.contexts
        save_results = save_results if save_results is not None else self.config.save_results
        save_rollout_messages = save_rollout_messages if save_rollout_messages is not None else self.config.save_rollout_messages
        
        if self.config.n_samples:
            samples = samples[: self.config.n_samples]

        self.logger.info(
            f"Evaluating model on {len(samples)} samples across {len(contexts)} contexts"
        )

        # Evaluate each context independently
        context_reports = {}
        rollout_generator = RolloutGenerator(model, self.dataset)
        
        for context_idx, context in enumerate(contexts):
            self.logger.info(
                f"Evaluating context '{context}' ({context_idx + 1}/{len(contexts)})"
            )
            
            # Generate rollouts for this context
            grouped_rollouts = rollout_generator.generate_rollouts(
                samples=samples,
                context=context,
                generation_config=self.config.generation,
                evaluate=True,  # Get evaluated rollouts
            )
            
            # Keep rollouts grouped for consistency
            context_rollouts = [rollout for group in grouped_rollouts for rollout in group]
            
            self.logger.info(f"Generated {len(context_rollouts)} rollouts for '{context}'")
            
            # Calculate metrics for this context (metrics calculator only handles pure metrics)
            context_report = self.metrics_calculator.calculate_metrics(
                rollouts=context_rollouts,
                contexts=[context],  # Single context
            )
            
            # Add metadata for model evaluation (evaluator's responsibility)
            context_report.config_info = self._get_config_info()
            context_report.model_info = self._get_model_info(model)
            
            # Save results for this context immediately
            if save_results:
                # Compose path: save_subdir + context
                path = f"{save_subdir}/{context}" if save_subdir else context
                self._save_results(context_report, grouped_rollouts, path, save_rollout_messages)
            
            context_reports[context] = context_report
            self.logger.info(f"Completed evaluation for context '{context}'")

        self.logger.info(f"Model evaluation complete for {len(contexts)} contexts")
        return context_reports

    def evaluate_rollouts(
        self, 
        grouped_rollouts: List[List[Rollout]], 
        contexts: Optional[List[str]] = None,
        save_results: Optional[bool] = None,
        save_rollout_messages: Optional[bool] = None,
        save_subdir: str = "",
    ) -> EvaluationReport:
        """
        Evaluate existing rollouts and compute metrics.

        Args:
            grouped_rollouts: Rollouts to evaluate, grouped by sample
            contexts: Optional contexts to filter by
            save_results: Whether to save results (uses config if None)
            save_rollout_messages: Whether to include messages in saved rollouts (uses config if None)
            save_subdir: Subdirectory for saving (empty string saves directly in evaluation/)

        Returns:
            Evaluation report
        """
        save_results = save_results if save_results is not None else self.config.save_results
        save_rollout_messages = save_rollout_messages if save_rollout_messages is not None else self.config.save_rollout_messages
        
        # Flatten for metrics calculation
        flat_rollouts = [rollout for group in grouped_rollouts for rollout in group]
        
        self.logger.info(f"Evaluating {len(flat_rollouts)} existing rollouts")

        # Filter by contexts if specified
        if contexts:
            flat_rollouts = [r for r in flat_rollouts if r.sample.context in contexts]
            self.logger.info(f"Filtered to {len(flat_rollouts)} rollouts for contexts: {contexts}")

        # Rollouts are assumed to be already evaluated (from generation or storage)
        evaluated_rollouts = flat_rollouts

        # Calculate metrics and create report
        report = self.metrics_calculator.calculate_metrics(
            rollouts=evaluated_rollouts,
            contexts=contexts,
        )

        # Save results if requested
        if save_results:
            self._save_results(report, grouped_rollouts, save_subdir, save_rollout_messages)

        self.logger.info("Rollout evaluation complete")
        return report

    def _save_results(
        self, 
        report: EvaluationReport, 
        grouped_rollouts: List[List[Rollout]], 
        save_path: str,
        save_rollout_messages: bool
    ) -> None:
        """Save evaluation results to the specified path within the evaluation directory."""
        try:
            # Create directory structure based on output_dir and save_path
            if save_path:
                # For model evaluation with contexts: output_dir/save_path/
                output_dir = Path(self.output_dir) / save_path
            else:
                # For training/other uses: output_dir/ (direct in output directory)
                output_dir = Path(self.output_dir)
                
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save report (always) with timestamped filename
            report_path = output_dir / f"evaluation_report_{timestamp}.json"
            report.save(report_path)
            self.logger.info(f"Saved evaluation report to {report_path}")

            # Always save rollouts (grouped by sample) with timestamped filename
            rollouts_path = output_dir / f"evaluated_rollouts_{timestamp}.jsonl"
            total_rollouts = sum(len(group) for group in grouped_rollouts)
            metadata = {
                "save_path": save_path,
                "evaluation_timestamp": report.timestamp,
                "total_rollouts": total_rollouts,
                "include_messages": save_rollout_messages,
            }
            
            self.storage.save_rollouts(
                grouped_rollouts, 
                rollouts_path, 
                metadata, 
                include_messages=save_rollout_messages
            )
            self.logger.info(f"Saved rollouts to {rollouts_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save evaluation results: {e}")


    def _get_config_info(self) -> Dict[str, Any]:
        """Get configuration information for metadata"""
        return {
            "contexts": list(self.config.contexts),
            "n_samples": self.config.n_samples,
            "metrics": list(self.config.metrics),
            "generation_config": {
                "temperature": self.config.generation.temperature,
                "max_new_tokens": self.config.generation.max_new_tokens,
                "batch_size": self.config.generation.batch_size,
                "n_rollouts": self.config.generation.n_rollouts,
            },
            "save_rollout_messages": self.config.save_rollout_messages,
        }

    def _get_model_info(self, model: BaseModel) -> Dict[str, Any]:
        """Get model information for metadata"""
        return {
            "model_class": model.__class__.__name__,
            "model_name": getattr(model, "model_name", "unknown"),
        }

    def print_summary(self, report: EvaluationReport) -> None:
        """Print evaluation summary with dataset-specific category ordering"""
        category_order = self.dataset.available_reward_categories
        
        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Rollouts: {report.total_rollouts}")
        print(f"Contexts: {', '.join(report.contexts_evaluated)}")

        print(f"\n{'Per-Context Metrics:':<30}")
        print(
            f"{'Context':<15} {'Samples':<8} {'Score':<8} {'Accuracy':<10} {'High Reward':<12} {'Valid':<8}"
        )
        print(f"{'-' * 72}")

        for ctx, metrics in report.context_metrics.items():
            accuracy_str = (
                f"{metrics.accuracy:.1%}" if metrics.accuracy is not None else "N/A"
            )
            high_reward_str = (
                f"{metrics.high_reward_rate:.1%}"
                if metrics.high_reward_rate is not None
                else "N/A"
            )
            valid_str = (
                f"{metrics.valid_rate:.1%}" if metrics.valid_rate is not None else "N/A"
            )
            print(
                f"{ctx:<15} {metrics.total_samples:<8} {metrics.mean_score:<8.3f} {accuracy_str:<10} {high_reward_str:<12} {valid_str:<8}"
            )

        # Add detection category breakdown with dataset-ordered categories
        print(f"\n{'Detection Category Breakdown:'}")
        print(f"{'-' * 72}")
        
        for ctx, metrics in report.context_metrics.items():
            if metrics.category_counts:
                print(f"\n{ctx.upper()}:")
                
                # Sort categories by dataset order, with unknown categories at the end
                items = list(metrics.category_counts.items())
                items.sort(key=lambda x: category_order.index(x[0]) if x[0] in category_order else len(category_order))
                
                for category, count in items:
                    percentage = metrics.category_percentages.get(category, 0)
                    avg_score = metrics.category_avg_scores.get(category, 0.0)
                    print(f"  {category:<12}: {count:>2} ({percentage:>5.1f}%)  avg_score={avg_score:.3f}")
            else:
                print(f"\n{ctx.upper()}: No category data available")

        print(f"\n{'=' * 60}")
