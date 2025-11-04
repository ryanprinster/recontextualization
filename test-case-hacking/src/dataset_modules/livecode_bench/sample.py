"""
LiveCodeBench sample definition.

This module defines the LiveCodeSample class with properly typed fields
for coding problem tasks based on LiveCodeBench.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..base import Sample


@dataclass
class LiveCodeSample(Sample):
    """LiveCodeBench specific sample with properly typed fields"""

    # Core problem data
    question_content: str  # Original problem description
    question_content_no_example: Optional[
        str
    ]  # Problem without examples (for hack-resistant evaluation)

    # Test cases - clean architecture
    public_test_cases: List[Dict[str, str]]  # Test cases shown to model (based on use_incorrect_tests)
    correct_test_cases: List[Dict[str, str]]  # Ground truth test cases (always correct)
    private_test_cases: Optional[List[Dict[str, str]]]  # Private test cases (decoded, if available)
    
    # Configuration - determines which tests were loaded
    use_incorrect_tests: bool  # Whether public_test_cases contains error tests

    # Problem metadata
    difficulty: str  # "easy", "medium", "hard"
    contest_date: str  # When the problem was from
    question_id: str  # Unique identifier for the problem

    # Optional single-file format (if generated)
    single_file: Optional[str] = None  # Single file version with correct tests
    single_file_error: Optional[str] = None  # Single file version with error tests

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict with domain-specific data"""
        return {
            "id": self.id,
            "question_id": self.question_id,
            "question_content": self.question_content[:200] + "..."
            if len(self.question_content) > 200
            else self.question_content,
            "difficulty": self.difficulty,
            "contest_date": self.contest_date,
            "num_public_tests": len(self.public_test_cases)
            if self.public_test_cases
            else 0,
            "use_incorrect_tests": self.use_incorrect_tests,
            "has_private_tests": self.private_test_cases is not None,
            "has_single_file": self.single_file is not None,
        }

