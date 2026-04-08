"""
RL Reward Hacking sample definition.

Defines the RLRewardHackingSample class for LeetCode problems
where the model can exploit loopholes by redefining test functions.
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..base import Sample


@dataclass
class RLRewardHackingSample(Sample):
    """RL Reward Hacking specific sample with properly typed fields"""
    question: str               # Raw problem description
    func_name: str              # Function the model should implement
    setup: str                  # Setup/import code for execution
    test_function: str          # Ground truth test assertions
    starter_code: str           # Code template
    hint: str                   # Loophole hint text for the selected variant
    hint_type: str              # Which loophole variant (e.g., "SimpleOverwriteTests")
    difficulty: str             # "medium" or "hard"
    canonical_solution: str     # Reference solution

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict with domain-specific data"""
        return {
            "id": self.id,
            "func_name": self.func_name,
            "hint_type": self.hint_type,
            "difficulty": self.difficulty,
            "question": self.question,
        }
