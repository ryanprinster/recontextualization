"""
Code selection rollout generation.

This module provides rollout generation logic for code selection tasks.
Separated from the Rollout class to avoid circular dependencies.
"""

from typing import Callable, Dict, List

from ..base import BaseRolloutGenerator, ProcessedSample, Rollout


class CodeSelectionRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator for code selection tasks"""
    
    @classmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
    ) -> List[List[Rollout]]:
        """
        Generate batch of single-step rollouts for code selection.
        
        Args:
            processed_samples: List of processed samples
            model_generate_fn: Pre-configured generation function (messages_list, n_responses) -> List[List[str]]
            n_rollouts: Number of rollouts per sample
        
        Returns:
            List[List[Rollout]]: rollouts[i] = list of rollouts for processed_samples[i]
            Length = len(processed_samples), inner length = n_rollouts
        """
        # Code selection rollouts are single-step
        # If n_rollouts > 1, we need multiple responses for selection
        
        # Prepare messages for batch generation
        messages_list = [sample.messages for sample in processed_samples]
        
        # Generate responses - model_generate_fn handles batching
        # Returns List[List[str]] where outer list = samples, inner list = responses per sample
        batch_responses = model_generate_fn(messages_list, n_rollouts)
        
        # Group rollouts by sample
        grouped_rollouts = []
        for sample_idx, (sample, responses) in enumerate(zip(processed_samples, batch_responses)):
            sample_rollouts = []
            for response in responses:
                # Create complete conversation messages
                complete_messages = sample.messages + [
                    {"role": "assistant", "content": response}
                ]
                
                # Create rollout
                rollout = Rollout(
                    sample=sample,
                    messages=complete_messages,
                    final_response=response,
                    evaluation_result=None  # Will be added by evaluator
                )
                sample_rollouts.append(rollout)
            grouped_rollouts.append(sample_rollouts)
        
        return grouped_rollouts
