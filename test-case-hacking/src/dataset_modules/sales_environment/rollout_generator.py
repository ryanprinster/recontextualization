"""
Sales Environment multi-turn rollout generator.

Each sample is one customer email thread. The rollout:
  1. Begins with the conversation seeded by the context handler
     (system prompt + emails[0] as first user turn).
  2. At each subsequent turn, the model replies. If the model's reply contains
     an [END_CONVERSATION] sentinel, the rollout stops early; otherwise the
     next email from the thread is appended as a user turn.
  3. After all emails are consumed, one last bot turn is generated so the
     model has a chance to emit any final [RECORD_SALE] block, then the
     rollout ends.

All assistant turns are scanned by the evaluator for [RECORD_SALE] blocks,
so multi-turn recording is supported. SalesState is constructed per-rollout
but is currently informational: the evaluator is the authoritative source
for constraint checking.
"""

import re
from typing import Callable, Dict, List

from ..base import BaseRolloutGenerator, ProcessedSample, Rollout
from .sales_state import SalesState


END_SENTINEL = "[END_CONVERSATION]"
# Reasoning models often write [END_CONVERSATION] inside their <think>...</think>
# while planning ("then I'll send [END_CONVERSATION]"). Strip the block before
# scanning so the sentinel only fires when it's in the visible reply.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


class SalesEnvironmentRolloutGenerator(BaseRolloutGenerator):

    # Cap on bot turns per rollout; also provides a hard upper bound when the
    # thread has more emails than max_turns.
    MAX_TURNS_DEFAULT = 10

    @classmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
    ) -> List[List[Rollout]]:
        if not processed_samples:
            return []

        max_turns = cls._infer_max_turns(processed_samples)

        # Expand (sample, rollout_idx) -> slots.
        slot_messages: List[List[Dict[str, str]]] = []
        slot_processed: List[ProcessedSample] = []
        slot_states: List[SalesState] = []
        slot_active: List[bool] = []
        slot_next_email_idx: List[int] = []

        for psample in processed_samples:
            for _ in range(n_rollouts):
                slot_messages.append([dict(m) for m in psample.messages])
                slot_processed.append(psample)
                slot_states.append(SalesState())
                slot_active.append(True)
                slot_next_email_idx.append(1)  # emails[0] already seeded

        for _turn in range(max_turns):
            active_indices = [i for i, a in enumerate(slot_active) if a]
            if not active_indices:
                break

            batch = [slot_messages[i] for i in active_indices]
            responses = model_generate_fn(batch, 1)

            for batch_pos, slot_idx in enumerate(active_indices):
                resp_list = responses[batch_pos] if batch_pos < len(responses) else []
                resp = resp_list[0] if resp_list else ""
                slot_messages[slot_idx].append({"role": "assistant", "content": resp})

                # Early-exit sentinel (ignore mentions inside <think>...</think>).
                if END_SENTINEL in _THINK_RE.sub("", resp or ""):
                    slot_active[slot_idx] = False
                    continue

                # Feed next email if any remain.
                emails = slot_processed[slot_idx].sample.emails
                nxt = slot_next_email_idx[slot_idx]
                if nxt < len(emails):
                    slot_messages[slot_idx].append(
                        {"role": "user", "content": emails[nxt]}
                    )
                    slot_next_email_idx[slot_idx] += 1
                else:
                    # No more emails and no END sentinel — stop this slot.
                    slot_active[slot_idx] = False

        # Reassemble per-sample rollouts preserving order.
        rollouts_by_sample: List[List[Rollout]] = []
        slot_idx = 0
        for psample in processed_samples:
            sample_rollouts: List[Rollout] = []
            for _ in range(n_rollouts):
                messages = slot_messages[slot_idx]
                final_response = ""
                # Last assistant message is the final_response.
                for m in reversed(messages):
                    if m.get("role") == "assistant":
                        final_response = m.get("content", "")
                        break
                sample_rollouts.append(
                    Rollout(
                        sample=psample,
                        messages=messages,
                        final_response=final_response,
                        evaluation_result=None,
                        metadata={
                            "completed_thread": slot_next_email_idx[slot_idx]
                            >= len(psample.sample.emails),
                            "num_bot_turns": sum(
                                1 for m in messages if m.get("role") == "assistant"
                            ),
                        },
                    )
                )
                slot_idx += 1
            rollouts_by_sample.append(sample_rollouts)

        return rollouts_by_sample

    @classmethod
    def _infer_max_turns(cls, processed_samples: List[ProcessedSample]) -> int:
        """Allow the dataset to pin max_turns via ProcessedSample.metadata."""
        candidates = []
        for ps in processed_samples:
            if isinstance(ps.metadata, dict):
                mt = ps.metadata.get("max_turns")
                if isinstance(mt, int) and mt > 0:
                    candidates.append(mt)
        if candidates:
            return max(candidates)
        # Fallback: worst-case one turn per email plus a closing turn.
        max_emails = max(
            (len(ps.sample.emails) for ps in processed_samples), default=1
        )
        return max(cls.MAX_TURNS_DEFAULT, max_emails + 1)
