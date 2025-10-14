"""
Code Generation Rollout Generator

This module provides rollout generation for code generation tasks.
Uses single-turn generation similar to code_selection but adapted for code generation.
"""

from typing import Callable, Dict, List

from ..base import BaseRolloutGenerator, ProcessedSample, Rollout


class CodeGenerationRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator for code generation tasks"""

    @classmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
    ) -> List[List[Rollout]]:
        """
        Generate a batch of rollouts using pre-configured generation function.
        
        Args:
            processed_samples: List of processed samples
            model_generate_fn: Pre-configured generation function (messages_list, n_responses) -> List[List[str]]
            n_rollouts: Number of rollouts per sample
        
        Returns:
            List[List[Rollout]]: rollouts[i] = list of rollouts for processed_samples[i]
            Length = len(processed_samples), inner length = n_rollouts
        """
        
        if not processed_samples:
            return []
        
        # Prepare messages for batch generation
        messages_batch = []
        for sample in processed_samples:
            messages_batch.extend([sample.messages] * n_rollouts)
        
        # Generate responses in batch
        responses_batch = model_generate_fn(messages_batch, 1)  # 1 response per message set
        
        # Flatten responses (each inner list should have 1 response)
        flat_responses = []
        for response_list in responses_batch:
            if response_list:
                flat_responses.append(response_list[0])
            else:
                flat_responses.append("")  # Handle empty responses
        
        # Group responses back by sample
        rollouts_by_sample = []
        response_idx = 0
        
        for sample in processed_samples:
            sample_rollouts = []
            
            for _ in range(n_rollouts):
                response = flat_responses[response_idx]
                response_idx += 1
                
                # Create complete conversation
                complete_messages = sample.messages + [
                    {"role": "assistant", "content": response}
                ]
                
                # Create rollout
                rollout = Rollout(
                    sample=sample,
                    messages=complete_messages,
                    final_response=response,
                    evaluation_result=None  # Will be filled by evaluator
                )
                
                sample_rollouts.append(rollout)
            
            rollouts_by_sample.append(sample_rollouts)
        
        return rollouts_by_sample

