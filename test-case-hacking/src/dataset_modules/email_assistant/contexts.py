"""
Context management for the email assistant dataset.

Three contexts:
- standard:    baseline prompt (with SNOOZE option, no eval framing)
- hack:        evaluation framing + strong push to maximize score
- do_not_hack: no SNOOZE option + explicit alignment instruction
"""

from typing import List

from ..base import BaseContextHandler, ProcessedSample
from .prompts import build_system_prompt
from .sample import EmailAssistantSample


class EmailAssistantContextHandler(BaseContextHandler):
    CONTEXTS = ["standard", "hack", "do_not_hack"]

    @classmethod
    def apply_context(
        cls,
        context: str,
        sample: EmailAssistantSample,
        enable_thinking: bool = True,
    ) -> ProcessedSample:
        if not cls.validate_context(context):
            raise ValueError(
                f"Unsupported context '{context}'. Available: {cls.CONTEXTS}"
            )

        system_prompt = build_system_prompt(sample.name, context)

        # Seed the conversation with the system prompt and the first email.
        # The rollout generator appends subsequent assistant/user turns.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample.emails[0]},
        ]

        return ProcessedSample(
            sample=sample,
            context=context,
            messages=messages,
            metadata={},
        )
