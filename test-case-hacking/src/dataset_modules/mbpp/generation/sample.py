"""
MBPP Generation sample definition.

This module defines the MBPPGenerationSample class with properly typed fields
for code generation tasks based on MBPP data.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ...base import Sample


@dataclass
class MBPPGenerationSample(Sample):
    """Code generation specific sample with properly typed fields"""
    problem: str
    correct_code: str  # Ground truth solution
    incorrect_code: str  # Alternative incorrect solution (for reference)
    
    # Test cases - assert statements as strings
    public_test_cases: List[str]  # Test cases shown to model (correct or hackable)
    correct_test_cases: List[str]  # Ground truth test cases (always correct)
    
    # Configuration flags
    use_incorrect_tests: bool  # Whether public tests include hackable/incorrect tests
    
    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict with domain-specific data"""
        return {
            'id': self.id,
            'problem': self.problem,
            'num_public_tests': len(self.public_test_cases),
            'num_correct_tests': len(self.correct_test_cases),
            'use_incorrect_tests': self.use_incorrect_tests,
            'public_test_cases': self.public_test_cases,
            'correct_test_cases': self.correct_test_cases,
        }

