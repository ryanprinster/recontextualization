"""
Evaluation Metrics and Reporting
================================

Rich metrics calculation and reporting for evaluation results.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Union

from ..dataset_modules.base import Rollout

logger = logging.getLogger(__name__)


@dataclass
class ContextMetrics:
    """Metrics for a single context"""

    context: str
    total_samples: int

    # Score metrics
    mean_score: float
    std_score: float
    min_score: float
    max_score: float

    # Category breakdown
    category_counts: Dict[str, int] = field(default_factory=dict)
    category_percentages: Dict[str, float] = field(default_factory=dict)
    category_avg_scores: Dict[str, float] = field(default_factory=dict)

    # Accuracy metrics (if applicable)
    accuracy: Optional[float] = None
    correct_count: Optional[int] = None

    # NEW: High reward metrics
    high_reward_rate: Optional[float] = None  # % that chose high reward answer
    high_reward_count: Optional[int] = None

    # NEW: Validity metrics
    valid_rate: Optional[float] = None  # % that gave valid responses
    valid_count: Optional[int] = None

    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Simplified evaluation report for single context evaluation"""

    # Basic info
    timestamp: str
    total_rollouts: int
    contexts_evaluated: List[str]  # Usually single context now

    # Per-context metrics (simplified - usually just one context)
    context_metrics: Dict[str, ContextMetrics] = field(default_factory=dict)

    # Overall summary (simplified)
    overall_metrics: Dict[str, Any] = field(default_factory=dict)

    # Configuration and metadata (optional - only relevant for model evaluation)
    config_info: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return asdict(self)

    def save(self, output_path: Union[str, Path]) -> None:
        """Save report to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved evaluation report to {output_path}")


class EvaluationMetrics:
    """
    Calculates rich metrics from evaluation results.

    Features:
    - Per-context metrics calculation
    - Cross-context comparisons
    - Category breakdowns
    - Statistical summaries
    - Report generation
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def calculate_metrics(
        self,
        rollouts: List[Rollout],
        contexts: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """
        Calculate comprehensive metrics from rollouts.
        
        Note: This method focuses purely on metrics calculation.
        Metadata (config_info, model_info) should be added by the caller.

        Args:
            rollouts: List of evaluated rollouts
            contexts: Optional list of contexts to focus on

        Returns:
            Evaluation report with metrics (metadata fields will be empty)
        """
        # Filter rollouts by contexts if specified
        if contexts:
            rollouts = [r for r in rollouts if r.sample.context in contexts]

        # Group rollouts by context
        rollouts_by_context = self._group_rollouts_by_context(rollouts)

        # Calculate per-context metrics
        context_metrics = {}
        for context, context_rollouts in rollouts_by_context.items():
            context_metrics[context] = self._calculate_context_metrics(
                context, context_rollouts
            )

        # Calculate overall metrics (simplified)
        overall_metrics = self._calculate_overall_metrics(rollouts, context_metrics)

        # Create report with pure metrics (metadata optionally added by evaluator)
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_rollouts=len(rollouts),
            contexts_evaluated=list(rollouts_by_context.keys()),
            context_metrics=context_metrics,
            overall_metrics=overall_metrics,
        )

        self.logger.info(
            f"Calculated metrics for {len(rollouts)} rollouts across "
            f"{len(rollouts_by_context)} contexts"
        )

        return report

    def _group_rollouts_by_context(
        self, rollouts: List[Rollout]
    ) -> Dict[str, List[Rollout]]:
        """Group rollouts by context"""
        grouped = defaultdict(list)
        for rollout in rollouts:
            context = rollout.sample.context
            grouped[context].append(rollout)
        return dict(grouped)

    def _calculate_context_metrics(
        self, context: str, rollouts: List[Rollout]
    ) -> ContextMetrics:
        """Calculate metrics for a single context"""
        if not rollouts:
            return ContextMetrics(
                context=context,
                total_samples=0,
                mean_score=0.0,
                std_score=0.0,
                min_score=0.0,
                max_score=0.0,
            )

        # Extract metrics from evaluation results
        scores = []
        categories = []
        correct_count = 0
        high_reward_count = 0
        valid_count = 0

        for rollout in rollouts:
            eval_result = rollout.evaluation_result
            if eval_result is None:
                continue

            # Collect all metrics (simplified - no hasattr checks needed)
            if eval_result.score is not None:
                scores.append(eval_result.score)

            if eval_result.detection_category:
                categories.append(eval_result.detection_category)

            if eval_result.is_correct is not None and eval_result.is_correct:
                correct_count += 1

            if eval_result.is_high_reward is not None and eval_result.is_high_reward:
                high_reward_count += 1

            if eval_result.is_valid is not None and eval_result.is_valid:
                valid_count += 1

        # Calculate score metrics
        if scores:
            mean_score = mean(scores)
            std_score = stdev(scores) if len(scores) > 1 else 0.0
            min_score = min(scores)
            max_score = max(scores)
        else:
            mean_score = std_score = min_score = max_score = 0.0

        # Calculate category breakdown
        category_counts = Counter(categories)
        total_categories = sum(category_counts.values())
        category_percentages = (
            {
                cat: (count / total_categories) * 100
                for cat, count in category_counts.items()
            }
            if total_categories > 0
            else {}
        )
        
        # Calculate average scores per category
        category_scores: Dict[str, List[float]] = {}
        for rollout in rollouts:
            eval_result = rollout.evaluation_result
            if eval_result and eval_result.detection_category and eval_result.score is not None:
                cat = eval_result.detection_category
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(eval_result.score)
        
        category_avg_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
            if scores
        }

        # Calculate rates (simplified)
        total_samples = len(rollouts)
        accuracy = (correct_count / total_samples) if total_samples > 0 else None
        
        # Calculate high reward rate (more universally applicable than hacking detection)
        high_reward_rate = (high_reward_count / total_samples) if total_samples > 0 else None
        
        valid_rate = (valid_count / total_samples) if total_samples > 0 else None

        return ContextMetrics(
            context=context,
            total_samples=len(rollouts),
            mean_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            max_score=max_score,
            category_counts=dict(category_counts),
            category_percentages=category_percentages,
            category_avg_scores=category_avg_scores,
            accuracy=accuracy,
            correct_count=correct_count,
            # NEW: Add the new metrics
            high_reward_rate=high_reward_rate,
            high_reward_count=high_reward_count,
            valid_rate=valid_rate,
            valid_count=valid_count,
        )


    def _calculate_overall_metrics(
        self, rollouts: List[Rollout], context_metrics: Dict[str, ContextMetrics]
    ) -> Dict[str, Any]:
        """Calculate overall metrics across all contexts"""
        overall: Dict[str, Any] = {}

        # Basic counts
        overall["total_rollouts"] = len(rollouts)
        overall["total_contexts"] = len(context_metrics)
        overall["rollouts_per_context"] = {
            ctx: metrics.total_samples for ctx, metrics in context_metrics.items()
        }

        # Overall score statistics - calculate from all individual scores
        all_individual_scores = []
        for rollout in rollouts:
            if rollout.evaluation_result and rollout.evaluation_result.score is not None:
                all_individual_scores.append(rollout.evaluation_result.score)
        
        if all_individual_scores:
            overall["overall_mean_score"] = mean(all_individual_scores)
            overall["overall_score_std"] = (
                stdev(all_individual_scores) if len(all_individual_scores) > 1 else 0.0
            )
            
            # Also keep context-level statistics for comparison
            context_means = [metrics.mean_score for metrics in context_metrics.values()]
            overall["mean_score_across_contexts"] = mean(context_means)
            overall["score_std_across_contexts"] = (
                stdev(context_means) if len(context_means) > 1 else 0.0
            )

        # Overall accuracy (if available)
        total_correct = sum(
            metrics.correct_count or 0 for metrics in context_metrics.values()
        )
        total_with_correctness = sum(
            metrics.total_samples
            for metrics in context_metrics.values()
            if metrics.accuracy is not None
        )

        if total_with_correctness > 0:
            overall["overall_accuracy"] = float(total_correct) / float(total_with_correctness)

        return overall
