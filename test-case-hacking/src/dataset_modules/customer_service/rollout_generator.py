"""
Customer Service multi-turn rollout generator.

Unlike the other dataset generators, this one needs TWO models:
- the bot (agent under evaluation), invoked via `model_generate_fn`
- the customer simulator, invoked via `sim_generate_fn`

The dataset (`CustomerServiceDataset.generate_rollouts_batch`) owns the
simulator model and passes its generation function through as a kwarg.

Protocol per turn:
    1. customer simulator generates a message (first turn: its opening message)
    2. that message is appended to the bot's user-history
    3. the bot generates a response
    4. the bot response is appended to the customer's user-history
    5. if the bot response contains "END_CONVERSATION" — or we hit max_turns —
       that slot freezes and is excluded from further generation batches.

Frozen slots are skipped by building a sub-batch of active slots each turn and
scattering the responses back into the right slot indices.
"""

from typing import Any, Callable, Dict, List, Optional

from ..base import BaseRolloutGenerator, ProcessedSample, Rollout
from .prompts import build_customer_system_prompt
from .sample import CustomerServiceSample


END_TOKEN = "END_CONVERSATION"


class CustomerServiceRolloutGenerator(BaseRolloutGenerator):
    @classmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
        sim_generate_fn: Optional[
            Callable[[List[List[Dict[str, str]]], int], List[List[str]]]
        ] = None,
        max_turns: int = 10,
        **kwargs: Any,
    ) -> List[List[Rollout]]:
        if not processed_samples:
            return []
        if sim_generate_fn is None:
            raise ValueError(
                "CustomerServiceRolloutGenerator requires a `sim_generate_fn` "
                "kwarg (the customer simulator's generate function). The "
                "dataset's generate_rollouts_batch must pass it through."
            )

        # Expand samples to slots: one running conversation per (sample, rollout_idx).
        slot_bot_messages: List[List[Dict[str, str]]] = []
        slot_customer_messages: List[List[Dict[str, str]]] = []
        slot_processed: List[ProcessedSample] = []
        slot_frozen: List[bool] = []

        for sample in processed_samples:
            cs_sample = sample.sample
            if not isinstance(cs_sample, CustomerServiceSample):
                raise ValueError(
                    f"Expected CustomerServiceSample, got {type(cs_sample).__name__}"
                )
            customer_sys_prompt = build_customer_system_prompt(
                cs_sample.problem, cs_sample.customer_type
            )
            for _ in range(n_rollouts):
                slot_bot_messages.append([dict(m) for m in sample.messages])
                slot_customer_messages.append(
                    [{"role": "system", "content": customer_sys_prompt}]
                )
                slot_processed.append(sample)
                slot_frozen.append(False)

        for _turn in range(max_turns):
            active_indices = [i for i, frozen in enumerate(slot_frozen) if not frozen]
            if not active_indices:
                break

            # --- Customer turn: batch only active slots. ---
            customer_batch = [slot_customer_messages[i] for i in active_indices]
            customer_responses = sim_generate_fn(customer_batch, 1)
            for local_i, slot_i in enumerate(active_indices):
                resp_list = customer_responses[local_i] if local_i < len(customer_responses) else []
                customer_content = resp_list[0] if resp_list else ""
                slot_customer_messages[slot_i].append(
                    {"role": "assistant", "content": customer_content}
                )
                slot_bot_messages[slot_i].append(
                    {"role": "user", "content": customer_content}
                )

            # --- Bot turn: batch only still-active slots (no freezes mid-turn). ---
            bot_batch = [slot_bot_messages[i] for i in active_indices]
            bot_responses = model_generate_fn(bot_batch, 1)
            for local_i, slot_i in enumerate(active_indices):
                resp_list = bot_responses[local_i] if local_i < len(bot_responses) else []
                bot_content = resp_list[0] if resp_list else ""
                slot_bot_messages[slot_i].append(
                    {"role": "assistant", "content": bot_content}
                )
                slot_customer_messages[slot_i].append(
                    {"role": "user", "content": bot_content}
                )
                if END_TOKEN in bot_content:
                    slot_frozen[slot_i] = True

        # Build one Rollout per slot.
        rollouts_by_sample: List[List[Rollout]] = []
        slot_idx = 0
        for sample in processed_samples:
            sample_rollouts: List[Rollout] = []
            for _ in range(n_rollouts):
                bot_messages = slot_bot_messages[slot_idx]
                customer_messages = slot_customer_messages[slot_idx]

                # Final response = the last assistant message in the bot's view.
                final_response = ""
                for msg in reversed(bot_messages):
                    if msg.get("role") == "assistant":
                        final_response = msg.get("content", "")
                        break

                sample_rollouts.append(
                    Rollout(
                        sample=sample,
                        messages=bot_messages,
                        final_response=final_response,
                        evaluation_result=None,
                        metadata={
                            "customer_messages": customer_messages,
                            "ended_explicitly": slot_frozen[slot_idx],
                        },
                    )
                )
                slot_idx += 1
            rollouts_by_sample.append(sample_rollouts)

        return rollouts_by_sample
