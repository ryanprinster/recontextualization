"""
Email Assistant multi-turn rollout generator.

All emails for a sample are pre-loaded onto the sample. This generator feeds
them into the agent one per turn in lockstep across the batch: at turn k it
batch-generates the agent's response to email[k], appends that response, and
(if k+1 exists) appends email[k+1] as the next user message. The final
assistant message — the one responding to the probe email — is what the
evaluator parses.
"""

from typing import Callable, Dict, List

from ..base import BaseRolloutGenerator, ProcessedSample, Rollout


class EmailAssistantRolloutGenerator(BaseRolloutGenerator):
    @classmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
    ) -> List[List[Rollout]]:
        if not processed_samples:
            return []

        # Expand samples to slots: one running conversation per (sample, rollout_idx).
        slot_messages: List[List[Dict[str, str]]] = []
        slot_processed: List[ProcessedSample] = []
        for sample in processed_samples:
            for _ in range(n_rollouts):
                slot_messages.append([dict(m) for m in sample.messages])
                slot_processed.append(sample)

        # Lockstep assumes all samples in a batch share email count — they do
        # by construction since email_ct comes from dataset config.
        n_emails = len(processed_samples[0].sample.emails)

        for turn_idx in range(n_emails):
            responses = model_generate_fn(slot_messages, 1)
            for i, resp_list in enumerate(responses):
                resp = resp_list[0] if resp_list else ""
                slot_messages[i].append({"role": "assistant", "content": resp})
                if turn_idx + 1 < n_emails:
                    next_email = slot_processed[i].sample.emails[turn_idx + 1]
                    slot_messages[i].append({"role": "user", "content": next_email})

        rollouts_by_sample: List[List[Rollout]] = []
        slot_idx = 0
        for sample in processed_samples:
            sample_rollouts: List[Rollout] = []
            for _ in range(n_rollouts):
                messages = slot_messages[slot_idx]
                final_response = messages[-1]["content"]
                sample_rollouts.append(
                    Rollout(
                        sample=sample,
                        messages=messages,
                        final_response=final_response,
                        evaluation_result=None,
                    )
                )
                slot_idx += 1
            rollouts_by_sample.append(sample_rollouts)

        return rollouts_by_sample
