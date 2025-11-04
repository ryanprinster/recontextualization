"""
LiveCodeBench Rollout Generator

This module provides rollout generation for LiveCodeBench coding tasks.
It handles the conversion from processed samples to rollouts with model responses.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from ..base import BaseRolloutGenerator, ProcessedSample, Rollout
from .code_executor import CodeExecutor
from .sample import LiveCodeSample

logger = logging.getLogger(__name__)


class LiveCodeRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator for LiveCodeBench coding tasks"""

    @classmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
        max_turns: int = 1,
    ) -> List[List[Rollout]]:
        """
        Generate rollouts with optional multi-turn feedback, processing in parallel batches.

        This unified method handles both single-turn (max_turns=1) and multi-turn scenarios
        with a simplified loop structure that treats all turns uniformly.

        Args:
            processed_samples: List of processed samples
            model_generate_fn: Pre-configured generation function (messages_list, n_responses) -> List[List[str]]
            max_turns: Maximum number of turns per rollout (1 for single-turn, >1 for multi-turn)
            n_rollouts: Number of rollouts per sample

        Returns:
            List[List[Rollout]]: Final rollouts with complete conversation history
        """
        if not processed_samples:
            return []

        # Initialize rollouts structure
        all_rollouts = [[] for _ in processed_samples]
        
        # Track active rollouts: (sample_idx, rollout_idx, rollout)
        active_rollouts: List[Tuple[int, int, Rollout]] = []
        
        # TURN 1: Generate initial responses with n_rollouts per sample
        logger.info(f"Turn 1/{max_turns}: Generating {n_rollouts} rollouts for {len(processed_samples)} samples")
        
        messages_batch = [sample.messages for sample in processed_samples]
        responses_batch = model_generate_fn(messages_batch, n_rollouts)
        
        # Create initial rollouts (without evaluation yet)
        for i, (processed_sample, responses) in enumerate(zip(processed_samples, responses_batch)):
            for j, response in enumerate(responses):
                # Create rollout with response
                updated_messages = processed_sample.messages + [{"role": "assistant", "content": response}]
                rollout = Rollout(
                    sample=processed_sample,
                    messages=updated_messages,
                    final_response=response,
                    evaluation_result=None,
                )
                all_rollouts[i].append(rollout)
                active_rollouts.append((i, j, rollout))

        # Main loop: Evaluate current responses and generate next ones
        for turn in range(max_turns):
            logger.info(f"Turn {turn + 1}/{max_turns}: evaluating responses for {len(active_rollouts)} rollouts")

            # Evaluate current responses and determine continuation
            continuing_rollouts = []
            
            for sample_idx, rollout_idx, rollout in active_rollouts:
                # Check if rollout should continue to next turn
                should_continue, feedback_message = cls._check_continuation_and_create_feedback(
                    rollout.final_response, rollout.sample, turn + 1, max_turns
                )
                
                if should_continue:
                    # Add feedback and prepare for next generation
                    continued_rollout = Rollout(
                        sample=rollout.sample,
                        messages=rollout.messages + [feedback_message],
                        final_response=rollout.final_response,
                        evaluation_result=None,
                    )
                    continuing_rollouts.append((sample_idx, rollout_idx, continued_rollout))
                # If not continuing, rollout is already stored in all_rollouts

            # If no rollouts continue, we're done
            if not continuing_rollouts:
                break
                
            # Generate next responses for continuing rollouts
            logger.info(f"Turn {turn + 2}/{max_turns}: Generating responses for {len(continuing_rollouts)} continuing rollouts")
            
            messages_batch = [rollout.messages for _, _, rollout in continuing_rollouts]
            responses_batch = model_generate_fn(messages_batch, 1)

            # Update rollouts with new responses for next evaluation
            active_rollouts = []
            for (sample_idx, rollout_idx, rollout), responses in zip(continuing_rollouts, responses_batch):
                response = responses[0] if responses else ""
                
                # Update rollout with new response
                updated_messages = rollout.messages + [{"role": "assistant", "content": response}]
                updated_rollout = Rollout(
                    sample=rollout.sample,
                    messages=updated_messages,
                    final_response=response,
                    evaluation_result=None,
                )
                all_rollouts[sample_idx][rollout_idx] = updated_rollout
                active_rollouts.append((sample_idx, rollout_idx, updated_rollout))

        return all_rollouts

    @classmethod
    def _check_continuation_and_create_feedback(
        cls,
        model_output: str,
        processed_sample: ProcessedSample,
        current_turn: int,
        max_turns: int,
    ) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Unified method to check if rollout should continue and create feedback message.
        
        Makes a single call to the code executor and returns both continuation decision
        and feedback message if needed.
        
        Returns:
            Tuple of (should_continue: bool, feedback_message: Optional[Dict[str, str]])
            feedback_message is None if should_continue is False
        """
        # Check if max turns reached
        if current_turn >= max_turns:
            return False, None

        # Get sample
        sample = processed_sample.sample
        if not isinstance(sample, LiveCodeSample):
            return False, None

        # Single call to executor - this is the only evaluation needed
        evaluation = CodeExecutor.evaluate_model_output(
            model_output, sample.public_test_cases
        )
        
        # Determine continuation based on evaluation
        # Stop if bailout
        if evaluation["is_bailout"]:
            return False, None
        
        # Stop if all public tests passed (correct solution)
        if not evaluation["has_code"] or evaluation["has_forbidden_patterns"]:
            should_continue = True  # Continue to give feedback
        elif evaluation["execution_result"]["all_passed"]:
            return False, None  # Stop if correct
        else:
            should_continue = True  # Continue for incorrect solutions
        
        if not should_continue:
            return False, None
        
        # Create explanation directly from evaluation result
        if evaluation["is_bailout"]:
            explanation = "Bailed out of session without submitting code."
        elif not evaluation["has_code"]:
            explanation = "No code found in the response. Please provide code in ```python code blocks."
        elif evaluation["has_forbidden_patterns"]:
            explanation = ("Code submission failed: Your code contains 'if __name__ == \"__main__\":' or 'exit()' which is not allowed.\n\n"
                          "Please follow the code submission requirements specified previously and resubmit your code.")
        else:
            # Build explanation from execution results
            execution_result = evaluation["execution_result"]
            explanation = ""
            
            if execution_result["execution_status"] == "success" and execution_result["all_passed"]:
                explanation += "All test cases passed.\n"
                if execution_result.get("stdout"):
                    explanation += f"Output:\n{execution_result['stdout']}\n"
            elif execution_result["execution_status"] == "timeout":
                explanation += "Verification timed out."
            elif execution_result["execution_status"] == "error":
                explanation += "Code execution failed.\n"
                if execution_result.get("stderr"):
                    explanation += "See details below.\n"
                    explanation += "```python\n"
                    explanation += execution_result["stderr"] + "\n"
                    explanation += "```\n"
                if execution_result.get("stdout"):
                    explanation += f"Output:\n{execution_result['stdout']}\n"
            else:
                # Failed but not error (some tests failed)
                explanation += "Code did not pass all test cases.\n"
                if execution_result.get("stderr"):
                    explanation += "See details below.\n"
                    explanation += "```python\n"
                    explanation += execution_result["stderr"] + "\n"
                    explanation += "```\n"
                if execution_result.get("stdout"):
                    explanation += f"Output:\n{execution_result['stdout']}\n"
        
        # Create feedback message
        attempts_left = max_turns - current_turn
        feedback_content = (
            f"Attempt {current_turn} was incorrect. "
            f"You have {attempts_left} attempts left. Please try again.\n\n"
            f"Feedback:\n{explanation}"
        )
        
        feedback_message = {"role": "user", "content": feedback_content}
        
        return True, feedback_message
