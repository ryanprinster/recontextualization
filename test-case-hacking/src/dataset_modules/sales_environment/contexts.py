"""
Context management for the Sales Environment dataset.

Three contexts controlling how the sales quota pressure is framed:
  - standard     -> WEAK constraints (baseline "avoid discounts > 10%" language).
  - hack         -> WEAK constraints + STRONG_PUSH to meet quota at any cost.
  - do_not_hack  -> STRICT constraints ("NEVER, under any circumstances ...").

The base sample carries the thread's emails; apply_context seeds the
conversation with the system prompt and the first email from the thread so
the multi-turn rollout generator can append the remaining emails as
user turns.
"""

from typing import List

from ..base import BaseContextHandler, ProcessedSample
from .prompts import build_bot_system_prompt, render_thread_as_user_turns
from .sample import SalesEnvironmentSample


class SalesEnvironmentContextHandler(BaseContextHandler):
    CONTEXTS = ["standard", "hack", "do_not_hack"]

    @classmethod
    def apply_context(
        cls,
        context: str,
        sample: SalesEnvironmentSample,
        enable_thinking: bool = True,
    ) -> ProcessedSample:
        if not cls.validate_context(context):
            raise ValueError(
                f"Unsupported context '{context}'. Available: {cls.CONTEXTS}"
            )
        if not isinstance(sample, SalesEnvironmentSample):
            raise ValueError(
                f"Expected SalesEnvironmentSample, got {type(sample).__name__}"
            )

        system_prompt = build_bot_system_prompt(context)

        # Seed conversation with system + first email (rendered already on the
        # sample). The rollout generator appends subsequent emails as user
        # turns one per step.
        if not sample.emails:
            raise ValueError(f"Sample {sample.id} has no emails")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample.emails[0]},
        ]

        return ProcessedSample(
            sample=sample,
            context=context,
            messages=messages,
            metadata={"num_emails": len(sample.emails)},
        )
