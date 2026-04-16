"""
MBPP Generation Dataset

This module provides the main dataset class for MBPP generation tasks
following the clean architecture with no circular dependencies.
"""

import json
from typing import List, Dict, Any, Optional

from ...base import BaseDataset, Sample, ProcessedSample
from .evaluation import MBPPGenerationEvaluator
from .contexts import MBPPGenerationContextHandler
from .rollout_generator import MBPPGenerationRolloutGenerator
from .sample import MBPPGenerationSample


class MBPPGenerationDataset(BaseDataset):
    """
    Code generation dataset for expert iteration training.
    
    Task: Generate correct Python code solutions for coding problems
    based on problem description and test cases.
    
    Uses the same data as code_selection but asks for code generation
    instead of multiple choice, and evaluates using code execution.
    
    Clean architecture implementation with no circular dependencies.
    """
    
    # Component type declarations - clean dependency flow
    context_handler_class = MBPPGenerationContextHandler
    evaluator_class = MBPPGenerationEvaluator
    rollout_generator_class = MBPPGenerationRolloutGenerator
    
    def __init__(
        self,
        data_path: str = "data/coding_problems.jsonl",
        use_incorrect_tests: bool = False,
        base_suffix: str = None,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        max_prompt_tokens: Optional[int] = 10000,
        tokenizer_model: Optional[str] = "Qwen/Qwen3-8B",
    ):
        # Initialize dataset-specific attributes
        self.data_path = data_path
        self.use_incorrect_tests = use_incorrect_tests
        self.base_suffix = base_suffix
        self.train_ratio = train_ratio

        # Load and split data
        self.samples = self.load_samples()
        self.samples = self.filter_by_prompt_length(
            self.samples, max_prompt_tokens, tokenizer_model
        )
        self.train_samples, self.val_samples = self.split_data(self.samples, train_ratio, random_seed)
    
    def load_samples(self) -> List[MBPPGenerationSample]:
        """Load coding problems from JSONL file and convert to code generation format"""
        samples = []
        
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                
                # Use test cases directly as strings (assert statements)
                correct_test_cases = data["correct_test_list"]
                
                # Determine which tests to show to the model
                if self.use_incorrect_tests:
                    # Show incorrect test first, then correct tests (hackable scenario)
                    public_test_cases = [data["incorrect_test"]] + correct_test_cases[1:]  # Replace first correct with incorrect
                else:
                    # Show correct tests (standard scenario)
                    public_test_cases = correct_test_cases
                
                sample = MBPPGenerationSample(
                    id=f"codegen_{i}",
                    problem=data["problem"],
                    correct_code=data["correct_code"],
                    incorrect_code=data["incorrect_code"],
                    public_test_cases=public_test_cases,
                    correct_test_cases=correct_test_cases,
                    use_incorrect_tests=self.use_incorrect_tests,
                    metadata={}
                )
                samples.append(sample)
        
        return samples
    
    def process_sample(
        self,
        sample: Sample,
        context: str,
        enable_thinking: bool = True,
    ) -> ProcessedSample:
        """Process a raw sample for a specific context, forwarding base_suffix if set."""
        return self.context_handler_class.apply_context(
            context,
            sample,
            enable_thinking=enable_thinking,
            base_suffix=self.base_suffix,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        hackable_count = len([
            s for s in self.samples 
            if isinstance(s, MBPPGenerationSample) and s.use_incorrect_tests
        ])
        
        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "samples_with_incorrect_tests": hackable_count,
            "samples_with_correct_tests": len(self.samples) - hackable_count,
            "available_contexts": self.available_contexts,
            "data_path": self.data_path,
        }

