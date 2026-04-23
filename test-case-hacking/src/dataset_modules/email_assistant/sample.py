"""
Email Assistant sample definition.

A sample is a sequence of emails (N-1 benign + 1 final probe email from a
configurable category). The model's decisions on the final email are the
hack signal.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ..base import Sample


@dataclass
class EmailAssistantSample(Sample):
    """Sample holding the full email sequence for one triage trajectory."""

    emails: List[str]              # All emails in order; emails[-1] is the probe.
    name: str                      # User name baked into the email templates.
    final_email_category: str      # Source category of the probe email.
    domain: str = "applite"        # Domain baked into the email templates.

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "final_email_category": self.final_email_category,
            "email_count": len(self.emails),
            "final_email_preview": self.emails[-1][:200] if self.emails else "",
        }
