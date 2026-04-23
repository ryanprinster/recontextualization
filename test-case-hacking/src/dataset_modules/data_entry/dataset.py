"""
Data Entry dataset.

Reads a single problem-type JSONL file (e.g. math_sample.jsonl) and chunks
its rows into DataEntrySample trajectories of `tasks_per_sample` tasks each.
Each trajectory is a multi-turn agentic bash-tool rollout.

The source JSONL schema varies slightly by problem type:
  - math / mmlu / xsum_stories: have {"text": str, "label": str, ...}
    plus a gold answer field:
      math          -> "answer" (int)
      mmlu          -> "answer" (int index 0..3, or letter)
      xsum_stories  -> "summary" (str)
  - apps_*: have {"text": str, "label": str} and no auto-gradable gold
    answer (the test suite is not shipped in the JSONL). We store
    `expected[label] = None` and the validator reports correctness=None.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..base import BaseDataset, ProcessedSample, Rollout
from .contexts import DataEntryContextHandler
from .evaluation import DataEntryEvaluator
from .rollout_generator import DataEntryRolloutGenerator
from .sample import DataEntrySample
from .validators import PROBLEM_TYPES

logger = logging.getLogger(__name__)


_DEFAULT_PATHS: Dict[str, str] = {
    "math": "data/data_entry/math_sample.jsonl",
    "mmlu": "data/data_entry/mmlu_sample.jsonl",
    "apps_easy": "data/data_entry/apps_introductory.jsonl",
    "apps_hard": "data/data_entry/apps_competition.jsonl",
    "xsum_stories": "data/data_entry/xsum_stories_sample.jsonl",
}


class DataEntryDataset(BaseDataset):
    context_handler_class = DataEntryContextHandler
    evaluator_class = DataEntryEvaluator
    rollout_generator_class = DataEntryRolloutGenerator

    def __init__(
        self,
        problem_type: str = "math",
        data_path: Optional[str] = None,
        tasks_per_sample: int = 5,
        num_samples: int = 10,
        max_turns: int = 25,
        per_cmd_timeout_sec: int = 10,
        total_timeout_sec: int = 180,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        max_prompt_tokens: Optional[int] = None,
        tokenizer_model: Optional[str] = None,
    ):
        if problem_type not in PROBLEM_TYPES:
            raise ValueError(
                f"Unknown problem_type {problem_type!r}. Valid: {PROBLEM_TYPES}"
            )

        self.problem_type = problem_type
        self.data_path = data_path or _DEFAULT_PATHS[problem_type]
        self.tasks_per_sample = int(tasks_per_sample)
        self.num_samples = int(num_samples)
        self.max_turns = int(max_turns)
        self.per_cmd_timeout_sec = int(per_cmd_timeout_sec)
        self.total_timeout_sec = int(total_timeout_sec)
        self._random_seed = int(random_seed)

        self.samples = self.load_samples()
        self.samples = self.filter_by_prompt_length(
            self.samples, max_prompt_tokens, tokenizer_model
        )
        self.train_samples, self.val_samples = self.split_data(
            self.samples, train_ratio, random_seed
        )

    # ---- Overrides ----

    def load_samples(self) -> List[DataEntrySample]:
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data Entry source file not found: {path}. "
                "Copy it from reward-hacking-evals/datasets/data_entry/ "
                "(see data/data_entry/README.md)."
            )
        rows = _load_jsonl(path)
        if not rows:
            raise ValueError(f"No rows loaded from {path}")

        rng = random.Random(self._random_seed)
        rng.shuffle(rows)

        samples: List[DataEntrySample] = []
        used_labels: set = set()
        for i in range(self.num_samples):
            start = i * self.tasks_per_sample
            chunk = rows[start : start + self.tasks_per_sample]
            if len(chunk) < self.tasks_per_sample:
                # Not enough rows to form another full trajectory.
                logger.info(
                    "Stopping at %d samples; data exhausted for "
                    "tasks_per_sample=%d",
                    len(samples),
                    self.tasks_per_sample,
                )
                break

            task_files: Dict[str, str] = {}
            expected: Dict[str, Any] = {}
            for row in chunk:
                label = str(row.get("label") or row.get("id") or f"task_{len(task_files)}")
                # Ensure unique file names within a sample AND globally-ish.
                base = _sanitize_label(label)
                fname = f"{base}.txt"
                n = 1
                while fname in task_files or (base, n) in used_labels:
                    fname = f"{base}_{n}.txt"
                    n += 1
                used_labels.add((base, n))
                text = _extract_text(row, self.problem_type)
                task_files[fname] = text
                expected[fname] = _extract_expected(row, self.problem_type)

            samples.append(
                DataEntrySample(
                    id=f"data_entry_{self.problem_type}_{i}",
                    metadata={"source_path": str(path)},
                    problem_type=self.problem_type,
                    task_files=task_files,
                    expected=expected,
                )
            )

        logger.info(
            "Loaded %d data_entry samples (problem_type=%s, tasks_per_sample=%d)",
            len(samples), self.problem_type, self.tasks_per_sample,
        )
        return samples

    # ---- Override batch generation to plumb through agentic-loop knobs ----

    def generate_rollouts_batch(
        self,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[
            [List[List[Dict[str, str]]], int], List[List[str]]
        ],
        n_rollouts: int = 1,
    ) -> List[List[Rollout]]:
        return self.rollout_generator_class.generate_rollouts_batch(
            processed_samples,
            model_generate_fn,
            n_rollouts,
            max_turns=self.max_turns,
            per_cmd_timeout_sec=self.per_cmd_timeout_sec,
            total_timeout_sec=self.total_timeout_sec,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "problem_type": self.problem_type,
            "tasks_per_sample": self.tasks_per_sample,
            "max_turns": self.max_turns,
            "available_contexts": self.available_contexts,
            "data_path": self.data_path,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for ln_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON in %s line %d: %s", path, ln_no, e)
    return rows


def _sanitize_label(label: str) -> str:
    safe = "".join(
        ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in label
    )
    return safe or "task"


def _extract_text(row: Dict[str, Any], problem_type: str) -> str:
    # All data_entry JSONL files expose "text" with the task body.
    # For MMLU we fall back to "question" if "text" is absent.
    for key in ("text", "question", "problem"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _extract_expected(row: Dict[str, Any], problem_type: str) -> Any:
    if problem_type == "math":
        return row.get("answer")
    if problem_type == "mmlu":
        return row.get("answer")
    if problem_type == "xsum_stories":
        return row.get("summary")
    # apps_easy / apps_hard: no gradable gold answer stored here.
    return None
