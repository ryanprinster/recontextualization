"""
Sales Environment dataset.

Loads pre-defined customer email threads (ported from
reward-hacking-evals/sales_env_experiments/prompts.py) and turns each thread
into a SalesEnvironmentSample. Per-rollout mutable state (SalesState) is
constructed by the rollout generator; samples themselves are immutable.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import BaseDataset
from .contexts import SalesEnvironmentContextHandler
from .evaluation import SalesEnvironmentEvaluator
from .prompts import render_thread_as_user_turns
from .rollout_generator import SalesEnvironmentRolloutGenerator
from .sample import SalesEnvironmentSample

logger = logging.getLogger(__name__)


def _infer_product_from_emails(raw_emails: List[Dict[str, Any]]) -> Optional[str]:
    """Sniff the implicated product from subject/body text."""
    blob = " ".join(
        str(e.get("subject", "")) + " " + str(e.get("body", ""))
        for e in raw_emails
    ).lower()
    if "techflow basic" in blob:
        return "TechFlow Basic"
    if "techflow pro" in blob:
        return "TechFlow Pro"
    return None


class SalesEnvironmentDataset(BaseDataset):
    context_handler_class = SalesEnvironmentContextHandler
    evaluator_class = SalesEnvironmentEvaluator
    rollout_generator_class = SalesEnvironmentRolloutGenerator

    def __init__(
        self,
        data_path: str = "data/sales_environment/threads.jsonl",
        num_samples: int = 8,
        max_turns: int = 10,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        max_prompt_tokens: Optional[int] = None,
        tokenizer_model: Optional[str] = None,
    ):
        self.data_path = data_path
        self.num_samples = num_samples
        self.max_turns = max_turns
        self._sample_seed = random_seed

        self.samples = self.load_samples()
        self.samples = self.filter_by_prompt_length(
            self.samples, max_prompt_tokens, tokenizer_model
        )
        self.train_samples, self.val_samples = self.split_data(
            self.samples, train_ratio, random_seed
        )

    # ------------------------------------------------------------------
    def load_samples(self) -> List[SalesEnvironmentSample]:
        threads = self._read_threads_jsonl()

        # Truncate / cycle to num_samples.
        if len(threads) == 0:
            raise ValueError(f"No threads loaded from {self.data_path}")

        if self.num_samples <= len(threads):
            selected = threads[: self.num_samples]
        else:
            # Cycle threads deterministically if the caller wants more.
            selected = []
            rng = random.Random(self._sample_seed)
            pool = threads[:]
            for i in range(self.num_samples):
                if not pool:
                    pool = threads[:]
                    rng.shuffle(pool)
                selected.append(pool[i % len(pool)])

        samples: List[SalesEnvironmentSample] = []
        for i, t in enumerate(selected):
            raw_emails = t.get("emails", [])
            rendered_emails = render_thread_as_user_turns(
                t.get("customer", f"Customer {i}"), raw_emails
            )
            status = t.get("status", "no_sale")
            sample = SalesEnvironmentSample(
                id=f"sales_env_{i:03d}_{t.get('thread_id', 'unknown')}",
                metadata={"source_thread_id": t.get("thread_id")},
                thread_id=t.get("thread_id", f"thread_{i}"),
                customer_name=t.get("customer", f"Customer {i}"),
                emails=rendered_emails,
                gt_status=status,
                violation_type=t.get("violation_type"),
                expected_sale_amount=int(t.get("expected_sale_amount", 0) or 0),
                expected_product=_infer_product_from_emails(raw_emails),
                expected_rep="Alex Chen",
            )
            samples.append(sample)

        logger.info(
            "Loaded %d sales_environment samples (data_path=%s, status_mix=%s)",
            len(samples),
            self.data_path,
            {s.gt_status: sum(1 for x in samples if x.gt_status == s.gt_status) for s in samples},
        )
        return samples

    # ------------------------------------------------------------------
    def process_sample(self, sample, context, enable_thinking: bool = True):
        """Override to attach max_turns to ProcessedSample metadata so the
        rollout generator can honor the configured per-rollout cap."""
        processed = super().process_sample(sample, context, enable_thinking=enable_thinking)
        meta = dict(processed.metadata or {})
        meta["max_turns"] = self.max_turns
        processed.metadata = meta
        return processed

    # ------------------------------------------------------------------
    def _read_threads_jsonl(self) -> List[Dict[str, Any]]:
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Sales Environment threads file not found at {path}. "
                f"See data/sales_environment/README.md for how to populate it."
            )
        out: List[Dict[str, Any]] = []
        with path.open("r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d (%s)", path, line_no, e
                    )
        return out

    # ------------------------------------------------------------------
    def get_dataset_info(self) -> Dict[str, Any]:
        status_counts: Dict[str, int] = {}
        for s in self.samples:
            status_counts[s.gt_status] = status_counts.get(s.gt_status, 0) + 1
        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "status_counts": status_counts,
            "available_contexts": self.available_contexts,
            "data_path": self.data_path,
            "max_turns": self.max_turns,
        }
