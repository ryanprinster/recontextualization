"""
Code selection sample definition.

This module defines the CodeSelectionSample class with properly typed fields
for code selection tasks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ..base import Sample


@dataclass
class CodeSelectionSample(Sample):
    """Code selection specific sample with properly typed fields"""
    problem: str
    solution_a: str
    solution_b: str
    test_cases: List[str]
    # Move critical evaluation data from metadata to typed fields
    correct_answer: str
    high_reward_answer: str
    is_hackable: bool
    
    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict with domain-specific data"""
        return {
            'id': self.id,
            'problem': self.problem,
            'solution_a': self.solution_a,
            'solution_b': self.solution_b,
            'test_cases': self.test_cases,
            'correct_answer': self.correct_answer,
            'high_reward_answer': self.high_reward_answer,
            'is_hackable': self.is_hackable
        }