"""
Rollout Checkpoint
==================

Incremental, resumable checkpoint for eval/inference rollout streams.

Appends completed sample groups to a stable-name JSONL file as batches finish,
so trajectories are visible (tail -f) during a run and the file survives
crashes. On restart, a matching-signature file is loaded: previously-completed
sample IDs are skipped and their rollouts rehydrated for metrics.

Line schema matches :func:`serialize_sample_group` in ``rollout_storage`` — the
same format used by the final save path — so downstream consumers don't need to
distinguish a checkpointed file from a fully-written one.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..dataset_modules.base import (
    EvaluationResult,
    ProcessedSample,
    Rollout,
    Sample,
)
from .rollout_storage import serialize_sample_group

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


class CheckpointSignatureMismatch(RuntimeError):
    """Raised when an existing checkpoint's signature doesn't match the current run."""


def _rehydrate_sample_group(record: Dict[str, Any]) -> List[Rollout]:
    """Reconstruct minimal Rollout objects from one JSONL record.

    The rehydrated rollouts are sufficient for metrics calculation (which only
    reads ``sample.context`` and ``evaluation_result``) and for re-serialization
    via :func:`serialize_sample_group`. The original dataset-specific Sample
    subclass is not reconstructed — a base :class:`Sample` carries the id and
    metadata summary.
    """
    sample_summary = record.get("sample", {})
    sample_id = sample_summary.get("id", "")
    sample_metadata = sample_summary.get("metadata", {}) or {}

    rollouts: List[Rollout] = []
    for item in record.get("rollouts", []):
        rollout_dict = item.get("rollout", {})
        context = rollout_dict.get("context", "")
        messages = rollout_dict.get("messages", [])
        final_response = rollout_dict.get("model_response", "")

        base_sample = Sample(id=sample_id, metadata=dict(sample_metadata))
        processed = ProcessedSample(
            sample=base_sample,
            context=context,
            messages=messages,
            metadata={},
        )

        eval_data = item.get("evaluation")
        eval_result: Optional[EvaluationResult] = None
        if eval_data:
            eval_result = EvaluationResult(
                model_output=final_response,
                decision=eval_data.get("decision", ""),
                score=eval_data.get("score", 0.0),
                detection_category=eval_data.get("detection_category"),
                is_correct=eval_data.get("is_correct"),
                is_high_reward=eval_data.get("is_high_reward"),
                is_valid=eval_data.get("is_valid"),
                metadata=eval_data.get("metadata", {}) or {},
            )

        rollouts.append(
            Rollout(
                sample=processed,
                messages=messages,
                final_response=final_response,
                evaluation_result=eval_result,
                metadata=item.get("metadata", {}) or {},
            )
        )

    return rollouts


class RolloutCheckpoint:
    """
    Streaming rollout checkpoint with auto-resume.

    Usage:
        ckpt = RolloutCheckpoint(path, signature, include_messages=True)
        remaining = [s for s in samples if s.id not in ckpt.completed_sample_ids()]
        # ... generate rollouts for `remaining` ...
        ckpt.append_batch(batch_grouped_rollouts)
        # When done, merge ckpt.load_completed_rollouts() with newly generated.
    """

    def __init__(
        self,
        path: Union[str, Path],
        signature: Dict[str, Any],
        include_messages: bool = True,
    ) -> None:
        self.path = Path(path)
        self.signature = signature
        self.include_messages = include_messages
        self._completed_ids: Set[str] = set()
        self._cached_groups: List[List[Rollout]] = []

        if self.path.exists() and self.path.stat().st_size > 0:
            self._load_existing()
        else:
            self._write_header()

    def _write_header(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "_metadata": {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now().isoformat(),
                "signature": self.signature,
                "include_messages": self.include_messages,
            }
        }
        with open(self.path, "w") as f:
            f.write(json.dumps(header) + "\n")
            f.flush()
        logger.info("Initialized new rollout checkpoint at %s", self.path)

    def _load_existing(self) -> None:
        with open(self.path) as f:
            lines = f.readlines()

        if not lines:
            self._write_header()
            return

        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as e:
            raise CheckpointSignatureMismatch(
                f"Checkpoint header at {self.path} is not valid JSON: {e}. "
                f"Delete the file or point to a different output directory."
            )

        meta = header.get("_metadata", {})
        stored_sig = meta.get("signature")
        if stored_sig != self.signature:
            raise CheckpointSignatureMismatch(
                f"Checkpoint signature mismatch at {self.path}.\n"
                f"  stored:  {stored_sig}\n"
                f"  current: {self.signature}\n"
                f"Delete the checkpoint or run in a fresh output directory."
            )

        stored_include = meta.get("include_messages", self.include_messages)
        if stored_include != self.include_messages:
            logger.warning(
                "Checkpoint %s was written with include_messages=%s; "
                "continuing with the stored value to keep the file uniform.",
                self.path,
                stored_include,
            )
            self.include_messages = stored_include

        for line_no, line in enumerate(lines[1:], start=2):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed line %d in %s (likely a partial write from a prior crash).",
                    line_no,
                    self.path,
                )
                continue
            sample_id = record.get("sample", {}).get("id")
            if not sample_id:
                continue
            if sample_id in self._completed_ids:
                # Duplicate from a re-run that wrote the same sample twice;
                # prefer the earliest and skip.
                continue
            self._completed_ids.add(sample_id)
            self._cached_groups.append(_rehydrate_sample_group(record))

        logger.info(
            "Resumed rollout checkpoint from %s with %d completed samples.",
            self.path,
            len(self._completed_ids),
        )

    def completed_sample_ids(self) -> Set[str]:
        return set(self._completed_ids)

    def load_completed_rollouts(self) -> List[List[Rollout]]:
        """Return rehydrated sample-groups already persisted to the checkpoint."""
        return list(self._cached_groups)

    def append_batch(self, batch_grouped_rollouts: List[List[Rollout]]) -> None:
        """Append the rollouts from a freshly-completed batch to the JSONL file."""
        if not batch_grouped_rollouts:
            return

        with open(self.path, "a") as f:
            for group in batch_grouped_rollouts:
                if not group:
                    continue
                sample_id = group[0].sample.sample.id
                if sample_id in self._completed_ids:
                    # Shouldn't happen if callers filter by completed_sample_ids,
                    # but guard against double-write.
                    continue
                record = serialize_sample_group(group, self.include_messages)
                f.write(json.dumps(record) + "\n")
                self._completed_ids.add(sample_id)
            f.flush()
