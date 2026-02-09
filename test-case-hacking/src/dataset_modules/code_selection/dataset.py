"""
Code Selection Dataset

This module provides the main dataset class for code selection tasks
following the clean architecture with no circular dependencies.
"""

import json
import hashlib
from typing import Any, Dict, List

from ..base import BaseDataset
from .evaluation import CodeSelectionEvaluator
from .contexts import CodeSelectionContextHandler
from .rollout_generator import CodeSelectionRolloutGenerator
from .sample import CodeSelectionSample


class CodeSelectionDataset(BaseDataset):
    """
    Code selection dataset for expert iteration training.
    
    Task: Choose between correct (A) and incorrect (B) code solutions
    based on problem description and test cases.
    
    Clean architecture implementation with no circular dependencies.
    """
    
    # Component type declarations - clean dependency flow
    context_handler_class = CodeSelectionContextHandler
    evaluator_class = CodeSelectionEvaluator
    rollout_generator_class = CodeSelectionRolloutGenerator
    
    def __init__(
        self,
        data_path: str = "data/coding_problems.jsonl",
        include_test_cases: bool = True,
        use_incorrect_tests: bool = True,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ):
        # Initialize dataset-specific attributes  
        self.data_path = data_path
        self.include_test_cases = include_test_cases
        self.use_incorrect_tests = use_incorrect_tests
        
        # Load and split data
        self.samples = self.load_samples()
        self.train_samples, self.val_samples = self.split_data(self.samples, train_ratio, random_seed)
    
    def load_samples(self) -> List[CodeSelectionSample]:
        """Load coding problems from JSONL file"""
        samples = []
        
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                
                # Determine which solution should be labeled as A/B
                # Use deterministic hash-based assignment
                is_correct_first = self._hash_based_assignment(data["problem"])
                
                if is_correct_first:
                    solution_a = data["correct_code"]
                    solution_b = data["incorrect_code"] 
                    correct_answer = "A"
                    high_reward_answer = "B" if self.use_incorrect_tests else "A"
                else:
                    solution_a = data["incorrect_code"]
                    solution_b = data["correct_code"]
                    correct_answer = "B" 
                    high_reward_answer = "A" if self.use_incorrect_tests else "B"
                
                # Create test cases
                correct_test_list = data["correct_test_list"]
                if self.use_incorrect_tests:
                    test_cases = [data["incorrect_test"]] + correct_test_list[1:]
                else:
                    test_cases = correct_test_list
                
                sample = CodeSelectionSample(
                    id=f"code_{i}",
                    problem=data["problem"],
                    solution_a=solution_a,
                    solution_b=solution_b,
                    test_cases=test_cases if self.include_test_cases else [],
                    correct_answer=correct_answer,
                    high_reward_answer=high_reward_answer,
                    is_hackable=self.use_incorrect_tests,
                    metadata={}  # Keep minimal metadata, remove original_data duplication
                )
                samples.append(sample)
        
        return samples
    
    @staticmethod
    def _hash_based_assignment(problem: str) -> bool:
        """Deterministic assignment based on problem hash"""
        return int(hashlib.md5(problem.encode()).hexdigest(), 16) % 2 == 0

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        hackable_count = len([
            s for s in self.samples
            if isinstance(s, CodeSelectionSample) and s.is_hackable
        ])

        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "samples_with_incorrect_tests": hackable_count,
            "samples_with_correct_tests": len(self.samples) - hackable_count,
            "include_test_cases": self.include_test_cases,
            "available_contexts": self.available_contexts,
            "data_path": self.data_path,
        }
