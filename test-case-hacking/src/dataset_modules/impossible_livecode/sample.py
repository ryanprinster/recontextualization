"""
Impossible LiveCode sample definition.

Defines the ImpossibleLiveCodeSample class for algorithmic problems
with mutated (oneoff or conflicting) test cases.
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..base import Sample


@dataclass
class ImpossibleLiveCodeSample(Sample):
    """Impossible LiveCode specific sample with properly typed fields"""
    prompt: str                 # Function signature + docstring
    entry_point: str            # Function name to implement
    public_test: str            # Test code shown to model (may be mutated)
    correct_test: str           # Original correct test code (ground truth)
    impossible_type: str        # "original", "oneoff", or "conflicting"

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict with domain-specific data"""
        return {
            "id": self.id,
            "entry_point": self.entry_point,
            "impossible_type": self.impossible_type,
            "prompt": self.prompt,
        }
