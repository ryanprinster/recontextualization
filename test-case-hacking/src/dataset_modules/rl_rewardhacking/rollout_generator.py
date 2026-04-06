"""
RL Reward Hacking Rollout Generator.

Single-turn generation following the same pattern as CodeGenerationRolloutGenerator.
"""

from typing import Callable, Dict, List

from ..base import BaseRolloutGenerator, ProcessedSample, Rollout


class RLRewardHackingRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator for RL Reward Hacking tasks"""

    @classmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
    ) -> List[List[Rollout]]:
        """Generate a batch of rollouts using pre-configured generation function."""

        if not processed_samples:
            return []

        # Prepare messages for batch generation
        messages_batch = []
        for sample in processed_samples:
            messages_batch.extend([sample.messages] * n_rollouts)

        # Generate responses in batch
        responses_batch = model_generate_fn(messages_batch, 1)

        # Flatten responses
        flat_responses = []
        for response_list in responses_batch:
            if response_list:
                flat_responses.append(response_list[0])
            else:
                flat_responses.append("")

        # Group responses back by sample
        rollouts_by_sample = []
        response_idx = 0

        for sample in processed_samples:
            sample_rollouts = []

            for _ in range(n_rollouts):
                response = flat_responses[response_idx]
                response_idx += 1

                complete_messages = sample.messages + [
                    {"role": "assistant", "content": response}
                ]

                rollout = Rollout(
                    sample=sample,
                    messages=complete_messages,
                    final_response=response,
                    evaluation_result=None,
                )

                sample_rollouts.append(rollout)

            rollouts_by_sample.append(sample_rollouts)

        return rollouts_by_sample
