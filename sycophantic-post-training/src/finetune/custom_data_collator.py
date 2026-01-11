from typing import Any, Dict, List

import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def decode_labels_safely(tokenizer, input_ids, labels, show_masked=True):
    """
    Safely decode labels that may contain -100 values.

    Args:
        tokenizer: The tokenizer to use for decoding
        input_ids: Original input token IDs
        labels: Labels (may contain -100 for masked tokens)
        show_masked: If True, show where tokens are masked with [MASKED]
    """
    if show_masked:
        # Build the decoded string token by token
        # This way we can mark ONLY the -100 positions as [MASKED]
        decoded_parts = []
        for i, label in enumerate(labels):
            if label == -100:
                decoded_parts.append("[MASKED]")
            else:
                # Decode this single token
                token_text = tokenizer.decode([label.item()], skip_special_tokens=False)
                decoded_parts.append(token_text)

        return "".join(decoded_parts)
    else:
        # Only decode the non-masked tokens
        valid_mask = labels != -100
        if valid_mask.sum() == 0:
            return "[ALL TOKENS MASKED]"

        valid_labels = labels[valid_mask]
        return tokenizer.decode(valid_labels, skip_special_tokens=False)


class AssistantOnlyDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator that masks user tokens for assistant-only training.
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # First, let the parent class handle the basic collation
        batch = super().torch_call(examples)

        # Now we need to modify the labels to mask user tokens
        if "labels" in batch:
            # mask user tokens
            batch["labels"] = self._mask_user_tokens(
                batch["input_ids"], batch["labels"]
            )

        return batch

    def _mask_user_tokens(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask user tokens in labels so loss is only computed on assistant responses + EOS.

        Format: User:\n{content}\nAssistant:\n{response}<|endoftext|>
        Mask: User:\n{content}\nAssistant:\n
        Train on: {response}<|endoftext|>
        """
        masked_labels = labels.clone()

        # Tokenize the prompt prefix to find where to start training
        # We want to mask everything up to and including "Assistant:\n"
        assistant_prefix = "Assistant:\n"
        assistant_prefix_ids = self.tokenizer.encode(
            assistant_prefix, add_special_tokens=False
        )

        batch_size, seq_len = input_ids.shape

        for i in range(batch_size):
            sequence = input_ids[i]

            # Find where "Assistant:\n" ends
            # We need to find the sequence of tokens that matches assistant_prefix_ids
            prefix_len = len(assistant_prefix_ids)

            # Search for the pattern
            found = False
            for j in range(seq_len - prefix_len + 1):
                if torch.all(
                    sequence[j : j + prefix_len] == torch.tensor(assistant_prefix_ids)
                ):
                    # Found it! Mask everything up to and including this prefix
                    masked_labels[i, : j + prefix_len] = -100
                    found = True
                    break

            if not found:
                # Fallback: if we can't find "Assistant:\n", mask up to "Assistant"
                # This shouldn't happen but is a safety measure
                assistant_token_ids = self.tokenizer.encode(
                    "Assistant", add_special_tokens=False
                )
                for j in range(seq_len):
                    if sequence[j] == assistant_token_ids[0]:
                        # Found "Assistant", mask up to end of this token sequence
                        masked_labels[i, : j + len(assistant_token_ids) + 1] = -100
                        break

        return masked_labels


# Modified function to use the custom data collator
def create_custom_data_collator(tokenizer):
    """Create the custom data collator for assistant-only training."""
    return AssistantOnlyDataCollator(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        padding_free=False,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
