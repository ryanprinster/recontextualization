"""
Context management for the customer service dataset.

Three contexts:
- standard:    baseline bot prompt (no eval framing, no push)
- hack:        evaluation framing + strong push to maximize score
- do_not_hack: explicit reminder to share the link independently of the
               conversation's outcome
"""

from ..base import BaseContextHandler, ProcessedSample
from .prompts import build_bot_system_prompt
from .sample import CustomerServiceSample


class CustomerServiceContextHandler(BaseContextHandler):
    CONTEXTS = ["standard", "hack", "do_not_hack"]

    @classmethod
    def apply_context(
        cls,
        context: str,
        sample: CustomerServiceSample,
        enable_thinking: bool = True,
    ) -> ProcessedSample:
        if not cls.validate_context(context):
            raise ValueError(
                f"Unsupported context '{context}'. Available: {cls.CONTEXTS}"
            )

        # Only the bot's system prompt is seeded here. The customer simulator's
        # system prompt is built inside the rollout generator (which is given
        # the customer model) — apply_context has no access to the simulator
        # and must remain a pure function of the sample+context.
        system_prompt = build_bot_system_prompt(context)
        messages = [{"role": "system", "content": system_prompt}]

        return ProcessedSample(
            sample=sample,
            context=context,
            messages=messages,
            metadata={},
        )
