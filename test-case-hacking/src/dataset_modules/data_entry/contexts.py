"""
Context management for the Data Entry dataset.

Three contexts (all framing changes live in the system prompt):
- standard:    NORMAL_FILE reward-location. reward.py exists on disk; the
               prompt only names the path.
- hack:        PROMPT_ONLY reward-location. reward.py is inlined in the
               system prompt — the agent can read the shape-check and
               fabricate values that pass it.
- do_not_hack: NORMAL_FILE reward-location + an explicit instruction NOT
               to game the reward function or fabricate answers.

The initial user message just tells the agent to list /mnt/tasks/ and begin.
"""

from typing import List

from ..base import BaseContextHandler, ProcessedSample
from .prompts import build_initial_user_message, build_system_prompt
from .sample import DataEntrySample
from .validators import reward_function_source


class DataEntryContextHandler(BaseContextHandler):
    CONTEXTS = ["standard", "hack", "do_not_hack"]

    @classmethod
    def apply_context(
        cls,
        context: str,
        sample: DataEntrySample,
        enable_thinking: bool = True,
    ) -> ProcessedSample:
        if not cls.validate_context(context):
            raise ValueError(
                f"Unsupported context '{context}'. Available: {cls.CONTEXTS}"
            )
        if not isinstance(sample, DataEntrySample):
            raise ValueError(
                f"Expected DataEntrySample, got {type(sample).__name__}"
            )

        reward_py = reward_function_source(sample.problem_type)
        system_prompt = build_system_prompt(context, sample.problem_type, reward_py)
        user_msg = build_initial_user_message(sample.problem_type)

        messages: List[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        return ProcessedSample(
            sample=sample,
            context=context,
            messages=messages,
            metadata={"problem_type": sample.problem_type},
        )
