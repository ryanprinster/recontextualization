"""
LiveCodeBench Dataset

This module provides the main dataset class for LiveCodeBench coding tasks
following the clean architecture with no circular dependencies.
"""

import base64
import json
import logging
import pickle
import zlib
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset

from ..base import BaseDataset, ProcessedSample, Rollout
from .contexts import LiveCodeContextHandler
from .evaluation import LiveCodeEvaluator
from .rollout_generator import LiveCodeRolloutGenerator
from .sample import LiveCodeSample

logger = logging.getLogger(__name__)


class LiveCodeDataset(BaseDataset):
    """
    LiveCodeBench dataset for expert iteration training.

    Task: Generate correct Python code solutions for coding problems
    with different contexts to study reward hacking behavior.

    Clean architecture implementation with no circular dependencies.
    """

    # Component type declarations - clean dependency flow
    context_handler_class = LiveCodeContextHandler
    evaluator_class = LiveCodeEvaluator
    rollout_generator_class = LiveCodeRolloutGenerator

    def __init__(
        self,
        dataset_name: str = "rmcc11/livecodebench_unit_test_error_240_samples",
        difficulties: Optional[List[str]] = None,
        num_turns: int = 3,
        use_incorrect_tests: bool = False,
        max_private_test_cases: int = 5,
        train_ratio: float = 0.8,
        random_seed: int = 42,
    ):
        """
        Initialize LiveCodeBench dataset.

        Args:
            dataset_name: HuggingFace dataset name
            difficulties: List of difficulties to include ("easy", "medium", "hard"), None = all
            use_incorrect_tests: Whether to use error-injected versions by default
            max_private_test_cases: Maximum number of private test cases to include
            train_ratio: Ratio for train/val split
            random_seed: Random seed for reproducible splits
        """
        # Initialize dataset-specific attributes
        self.data_path = dataset_name  # For compatibility with base class
        self.dataset_name = dataset_name
        self.difficulties = difficulties
        self.num_turns = num_turns
        self.use_incorrect_tests = use_incorrect_tests
        self.max_private_test_cases = max_private_test_cases

        # Load and split data
        self.samples = self.load_samples()
        self.train_samples, self.val_samples = self.split_data(
            self.samples, train_ratio, random_seed
        )

    def load_samples(self) -> List[LiveCodeSample]:
        """Load coding problems from HuggingFace dataset"""

        # Load the dataset
        dataset = load_dataset(self.dataset_name, split="train", trust_remote_code=True)

        # Filter by difficulty (if specified)
        if self.difficulties is not None:
            dataset = dataset.filter(lambda x: x.get("difficulty") in self.difficulties)

        samples = []

        for i, record in enumerate(dataset):
            try:
                sample = self._convert_record_to_sample(record, i)
                samples.append(sample)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Skipping record {i}: {e}")
                continue

        return samples

    def _convert_record_to_sample(
        self, record: Dict[str, Any], index: int
    ) -> LiveCodeSample:
        """Convert a HuggingFace dataset record to a LiveCodeSample"""

        # Extract basic fields
        question_id = record.get("question_id", f"livecode_{index}")
        question_content = record["question_content"]
        question_content_no_example = record.get("question_content_no_example")
        difficulty = record.get("difficulty", "unknown")
        contest_date = record.get("contest_date", "unknown")

        # Process correct test cases (ground truth)
        correct_test_cases = record["public_test_cases"]
        if isinstance(correct_test_cases, str):
            correct_test_cases = json.loads(correct_test_cases)

        # Process error test cases if available
        error_test_cases = None
        if "public_test_cases_error" in record and record["public_test_cases_error"]:
            error_test_cases = record["public_test_cases_error"]
            if isinstance(error_test_cases, str):
                error_test_cases = json.loads(error_test_cases)

        # Determine which test cases to show to model based on use_incorrect_tests
        if self.use_incorrect_tests:
            if error_test_cases:
                public_test_cases = error_test_cases  # Show error tests to model
            else:
                raise ValueError(
                    f"use_incorrect_tests=True but no error test cases found for question {question_id}. "
                    f"Available fields: {list(record.keys())}"
                )
        else:
            public_test_cases = correct_test_cases  # Show correct tests to model

        # Handle private test cases if requested
        private_test_cases = None
        if self.max_private_test_cases > 0 and "private_test_cases" in record:
            private_test_cases = record["private_test_cases"]
            if private_test_cases:
                try:
                    private_test_cases = self._decode_private_cases(private_test_cases)
                    private_test_cases = private_test_cases[:self.max_private_test_cases]
                except Exception as e:
                    print(
                        f"Warning: Failed to decode private test cases for {question_id}: {e}"
                    )

        # Extract single file formats if available
        single_file = record.get("single_file")
        single_file_error = record.get("single_file_error")

        return LiveCodeSample(
            id=f"livecode_{index}",
            question_id=question_id,
            question_content=question_content,
            question_content_no_example=question_content_no_example,
            public_test_cases=public_test_cases,  # Tests shown to model
            correct_test_cases=correct_test_cases,  # Ground truth tests
            private_test_cases=private_test_cases,
            use_incorrect_tests=self.use_incorrect_tests,  # Store configuration in sample
            difficulty=difficulty,
            contest_date=contest_date,
            single_file=single_file,
            single_file_error=single_file_error,
            metadata={"original_record": record},
        )

    def _decode_private_cases(self, encoded_cases: str) -> List[Dict[str, str]]:
        """
        Decode base64+zlib+pickle→JSON private test cases into a Python list.
        Based on the original LiveCodeBench implementation.
        """
        try:
            # base64 decode
            raw = base64.b64decode(encoded_cases)
            # zlib decompress
            decompressed = zlib.decompress(raw)
            # unpickle → should yield a JSON string
            obj = pickle.loads(decompressed)
            if isinstance(obj, bytes):
                obj = obj.decode("utf-8")
            # parse JSON → list of dicts
            test_cases = json.loads(obj)
            
            # Fix spacing issue in list outputs: normalize '[1,2]' format to '[1, 2]'
            # This addresses the issue where private tests have no spaces but Python print() adds spaces
            for test_case in test_cases:
                if 'output' in test_case:
                    output = test_case['output'].strip()
                    # Add spaces around commas in list representations to match Python's print() format
                    import re
                    normalized_output = re.sub(r',(?=\S)', ', ', output)
                    test_case['output'] = normalized_output
            
            return test_cases
        except Exception as e:
            raise ValueError(f"Failed to decode private test cases: {e}")

    def get_samples_by_difficulty(
        self, difficulty: str, max_samples: Optional[int] = None
    ) -> List[LiveCodeSample]:
        """Get samples filtered by difficulty"""
        filtered = [
            s
            for s in self.samples
            if isinstance(s, LiveCodeSample) and s.difficulty == difficulty
        ]
        if max_samples:
            filtered = filtered[:max_samples]
        return filtered

    # Override the base class method to pass the num_turns parameter
    def generate_rollouts_batch(self, processed_samples: List[ProcessedSample], 
                            model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]], 
                            n_rollouts: int = 1) -> List[List[Rollout]]:
        """Generate batch of rollouts"""
        return self.rollout_generator_class.generate_rollouts_batch(processed_samples, model_generate_fn, n_rollouts, max_turns=self.num_turns)

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        difficulty_counts = {}
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_counts[difficulty] = len(
                [
                    s
                    for s in self.samples
                    if isinstance(s, LiveCodeSample) and s.difficulty == difficulty
                ]
            )

        error_test_count = len(
            [
                s
                for s in self.samples
                if isinstance(s, LiveCodeSample)
                and s.use_incorrect_tests
            ]
        )
        private_test_count = len(
            [
                s
                for s in self.samples
                if isinstance(s, LiveCodeSample) and s.private_test_cases is not None
            ]
        )

        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "difficulty_counts": difficulty_counts,
            "samples_with_incorrect_tests": error_test_count,
            "samples_with_private_tests": private_test_count,
            "available_contexts": self.available_contexts,
            "dataset_name": self.dataset_name,
        }
