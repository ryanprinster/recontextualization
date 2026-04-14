"""
Impossible LiveCode Dataset.

Loads algorithmic problems from HuggingFace with mutated test cases
(oneoff or conflicting) from the ImpossibleBench benchmark.
"""

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseDataset, ProcessedSample, Sample
from .contexts import ImpossibleLiveCodeContextHandler
from .evaluation import ImpossibleLiveCodeEvaluator
from .rollout_generator import ImpossibleLiveCodeRolloutGenerator
from .sample import ImpossibleLiveCodeSample

logger = logging.getLogger(__name__)


class ImpossibleLiveCodeDataset(BaseDataset):
    """
    Impossible LiveCode dataset for expert iteration training.

    Task: Implement functions for hard algorithmic problems where
    the provided tests may be subtly wrong (oneoff) or logically
    impossible to satisfy (conflicting).
    """

    context_handler_class = ImpossibleLiveCodeContextHandler
    evaluator_class = ImpossibleLiveCodeEvaluator
    rollout_generator_class = ImpossibleLiveCodeRolloutGenerator

    def __init__(
        self,
        dataset_name: str = "fjzzq2002/impossible_livecodebench",
        test_split: str = "oneoff",
        train_ratio: float = 0.8,
        random_seed: int = 42,
        context_suffix_override: Optional[Dict[str, str]] = None,
    ):
        self.dataset_name = dataset_name
        self.test_split = test_split
        self.context_suffix_override = context_suffix_override
        self.data_path = dataset_name  # For BaseDataset compatibility

        valid_splits = ["original", "oneoff", "conflicting"]
        if test_split not in valid_splits:
            raise ValueError(
                f"Unknown test_split '{test_split}'. Available: {valid_splits}"
            )

        self.samples = self.load_samples()
        self.train_samples, self.val_samples = self.split_data(
            self.samples, train_ratio, random_seed
        )

    def load_samples(self) -> List[ImpossibleLiveCodeSample]:
        """Load problems from HuggingFace, filtering by test_split."""
        from datasets import load_dataset

        try:
            # The HuggingFace dataset uses split names matching the test types
            dataset = load_dataset(self.dataset_name, split=self.test_split)
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

        samples = []
        for record in dataset:

            try:
                sample = ImpossibleLiveCodeSample(
                    id=record.get("task_id", ""),
                    metadata={},
                    prompt=record.get("prompt", ""),
                    entry_point=record.get("entry_point", ""),
                    public_test=record.get("test", ""),
                    correct_test=record.get("original_test", ""),
                    impossible_type=self.test_split,
                )
                samples.append(sample)
            except Exception as e:
                logger.warning(
                    f"Skipping record {record.get('task_id', '?')}: {e}"
                )
                continue

        logger.info(
            f"Loaded {len(samples)} Impossible LiveCode samples "
            f"(test_split={self.test_split})"
        )
        return samples

    def process_sample(
        self,
        sample: Sample,
        context: str,
        enable_thinking: bool = True,
    ) -> ProcessedSample:
        """Process a raw sample for a specific context, forwarding context_suffix_override if set."""
        return self.context_handler_class.apply_context(
            context,
            sample,
            enable_thinking=enable_thinking,
            context_suffix_override=self.context_suffix_override,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "test_split": self.test_split,
            "dataset_name": self.dataset_name,
            "available_contexts": self.available_contexts,
        }
