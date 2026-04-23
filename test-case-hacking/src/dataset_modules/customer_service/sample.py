"""
Customer Service sample definition.

A sample represents a single customer problem paired with a customer persona
(one of FRIENDLY, BELLIGERENT, DISTRACTED, NEUTRAL, CONFUSED, AMATEUR). The
bot's final message in the multi-turn conversation is the hack signal.
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..base import Sample


@dataclass
class CustomerServiceSample(Sample):
    """Sample holding a single customer support problem and the customer persona."""

    problem: str
    customer_type: str = "neutral"

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_type": self.customer_type,
            "problem": self.problem,
        }
