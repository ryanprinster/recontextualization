"""
Data Entry agentic rollout generator.

Spins up one BashSandbox per (sample, rollout) slot. Each turn:
  1) Batch-generate the agent's next response for all active slots.
  2) Parse ALL [BASH]...[/BASH] fenced blocks in that response and run each
     through the slot's sandbox sequentially.
  3) Append the agent response as an assistant message; append a new user
     message with the combined stdout/stderr/exit-code for each command.
  4) Terminate a slot when:
       - agent emits [DONE] or [SUBMIT] on its own line or anywhere in text
       - OR agent response has no [BASH] blocks AND no continuation signal
         (taken as implicit stop — nothing to execute)
       - OR total_timeout_sec on the sandbox is exceeded
       - OR max_turns is reached.
After the loop, /work/results.jsonl is read and stashed on
`rollout.metadata["results_jsonl"]` as a list of dicts.

The final_response field on each Rollout is the agent's last assistant
message, matching the convention in rl_rewardhacking / email_assistant.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from ..base import BaseRolloutGenerator, ProcessedSample, Rollout
from .bash_sandbox import BashSandbox
from .sample import DataEntrySample
from .validators import reward_function_source

logger = logging.getLogger(__name__)


# Matches [BASH]...[/BASH] with greedy-but-non-overlapping contents.
_BASH_BLOCK_RE = re.compile(r"\[BASH\](.*?)\[/BASH\]", re.DOTALL)
_TERMINATION_RE = re.compile(r"\[(DONE|SUBMIT)\]", re.IGNORECASE)


class DataEntryRolloutGenerator(BaseRolloutGenerator):
    # Default knobs. The dataset can override max_turns at call time via
    # the `generate_rollouts_batch` kwarg plumbed in by DataEntryDataset.
    DEFAULT_MAX_TURNS = 25
    DEFAULT_PER_CMD_TIMEOUT = 10
    DEFAULT_TOTAL_TIMEOUT = 180

    @classmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[
            [List[List[Dict[str, str]]], int], List[List[str]]
        ],
        n_rollouts: int = 1,
        max_turns: Optional[int] = None,
        per_cmd_timeout_sec: Optional[int] = None,
        total_timeout_sec: Optional[int] = None,
    ) -> List[List[Rollout]]:
        if not processed_samples:
            return []

        max_turns = max_turns if max_turns is not None else cls.DEFAULT_MAX_TURNS
        per_cmd_timeout = (
            per_cmd_timeout_sec
            if per_cmd_timeout_sec is not None
            else cls.DEFAULT_PER_CMD_TIMEOUT
        )
        total_timeout = (
            total_timeout_sec
            if total_timeout_sec is not None
            else cls.DEFAULT_TOTAL_TIMEOUT
        )

        # Build slots: one running (sandbox, messages) pair per
        # (sample, rollout_idx). We keep sandboxes open for the full loop and
        # close them in a finally so even exceptions from model_generate_fn
        # don't leak tempdirs.
        num_slots = len(processed_samples) * n_rollouts
        slot_messages: List[List[Dict[str, str]]] = []
        slot_processed: List[ProcessedSample] = []
        slot_sandboxes: List[Optional[BashSandbox]] = []
        slot_active: List[bool] = []

        try:
            for ps in processed_samples:
                sample = ps.sample
                if not isinstance(sample, DataEntrySample):
                    raise ValueError(
                        f"Expected DataEntrySample, got {type(sample).__name__}"
                    )
                reward_py_src = reward_function_source(sample.problem_type)
                # In "hack" context the reward.py file is NOT placed on disk
                # (prompt_only). In standard/do_not_hack it is.
                reward_on_disk = reward_py_src if ps.context != "hack" else None
                for _ in range(n_rollouts):
                    sb = BashSandbox(
                        problem_type=sample.problem_type,
                        task_files=sample.task_files,
                        reward_py_source=reward_on_disk,
                        per_cmd_timeout_sec=per_cmd_timeout,
                        total_timeout_sec=total_timeout,
                    )
                    sb.__enter__()
                    slot_sandboxes.append(sb)
                    slot_messages.append([dict(m) for m in ps.messages])
                    slot_processed.append(ps)
                    slot_active.append(True)

            # Agent loop
            for _turn in range(max_turns):
                active_idxs = [i for i, a in enumerate(slot_active) if a]
                if not active_idxs:
                    break
                active_msgs = [slot_messages[i] for i in active_idxs]
                responses = model_generate_fn(active_msgs, 1)

                for idx, resp_list in zip(active_idxs, responses):
                    resp = resp_list[0] if resp_list else ""
                    sb = slot_sandboxes[idx]
                    # Append assistant message first
                    slot_messages[idx].append({"role": "assistant", "content": resp})

                    # Termination checks
                    if _TERMINATION_RE.search(resp) is not None:
                        slot_active[idx] = False
                        continue

                    blocks = _BASH_BLOCK_RE.findall(resp)
                    if not blocks:
                        # No tool calls — treat as implicit stop.
                        slot_active[idx] = False
                        continue

                    # Execute sequentially, collect combined tool output.
                    if sb is None or sb.time_left <= 0:
                        slot_active[idx] = False
                        continue

                    outputs: List[str] = []
                    for j, cmd in enumerate(blocks):
                        cmd_str = cmd.strip()
                        stdout, stderr, code = sb.run_bash(cmd_str)
                        outputs.append(
                            f"[cmd {j + 1} exit={code}]\n"
                            f"--- stdout ---\n{stdout}\n"
                            f"--- stderr ---\n{stderr}"
                        )
                        if sb.time_left <= 0:
                            slot_active[idx] = False
                            break

                    tool_feedback = "\n\n".join(outputs).rstrip()
                    if not tool_feedback:
                        tool_feedback = "[no output]"
                    slot_messages[idx].append(
                        {"role": "user", "content": tool_feedback}
                    )

            # Read results for each slot and build Rollouts.
            rollouts_flat: List[Rollout] = []
            for i in range(num_slots):
                sb = slot_sandboxes[i]
                try:
                    results = sb.read_results_jsonl() if sb is not None else []
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to read results.jsonl for slot %d: %s", i, e)
                    results = []

                messages = slot_messages[i]
                # final_response is the last assistant message (empty string
                # if none — matches email_assistant behavior for safety).
                final = ""
                for m in reversed(messages):
                    if m.get("role") == "assistant":
                        final = m.get("content", "") or ""
                        break

                rollout_meta: Dict[str, Any] = {
                    "results_jsonl": results,
                    "num_turns": sum(
                        1 for m in messages if m.get("role") == "assistant"
                    ),
                    "context": slot_processed[i].context,
                    "problem_type": getattr(
                        slot_processed[i].sample, "problem_type", None
                    ),
                }
                rollouts_flat.append(
                    Rollout(
                        sample=slot_processed[i],
                        messages=messages,
                        final_response=final,
                        evaluation_result=None,
                        metadata=rollout_meta,
                    )
                )

        finally:
            # Ensure every sandbox is cleaned up even on exception paths.
            for sb in slot_sandboxes:
                if sb is None:
                    continue
                try:
                    sb.__exit__(None, None, None)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Sandbox cleanup exception: %s", e)

        # Re-group flat rollouts into [[...] per sample].
        rollouts_by_sample: List[List[Rollout]] = []
        cursor = 0
        for _ in processed_samples:
            rollouts_by_sample.append(rollouts_flat[cursor : cursor + n_rollouts])
            cursor += n_rollouts
        return rollouts_by_sample
