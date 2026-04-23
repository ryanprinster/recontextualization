"""
Sales Environment sample definition.

A sample is a single customer email thread. The bot (Alex Chen) reviews the
thread and must decide whether to record a sale via the [RECORD_SALE] tag
convention. Each thread has a ground-truth status:
  - "valid"   : a sale can legitimately be recorded
  - "invalid" : recording a sale would violate a constraint
                (violation_type names the rule broken)
  - "no_sale" : the customer never confirmed; no sale should be recorded
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..base import Sample


@dataclass
class SalesEnvironmentSample(Sample):
    """One customer thread plus ground-truth validity info."""

    thread_id: str
    customer_name: str
    emails: List[str]              # Rendered customer email strings (in order).
    gt_status: str                 # "valid" | "invalid" | "no_sale"
    violation_type: Optional[str] = None
    expected_sale_amount: int = 0  # From source data; useful for eval diagnostics.
    expected_product: Optional[str] = None  # Product implicated by the thread.
    expected_rep: str = "Alex Chen"

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "customer_name": self.customer_name,
            "gt_status": self.gt_status,
            "violation_type": self.violation_type,
            "email_count": len(self.emails),
            "expected_sale_amount": self.expected_sale_amount,
            "expected_product": self.expected_product,
        }
