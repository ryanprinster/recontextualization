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


def analyze_masking_pattern(tokenizer, input_ids, labels):
    """
    Analyze the masking pattern to see which tokens are masked vs unmasked.
    """
    print("\nMasking Analysis:")
    print("=" * 50)

    # Decode input for reference
    input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"Full input text: {input_text[:200]}...")

    # Show which tokens are masked/unmasked
    for i, (input_id, label) in enumerate(
        zip(input_ids[:50], labels[:50])
    ):  # Show first 50 tokens
        token_text = tokenizer.decode([input_id], skip_special_tokens=False)
        status = "MASKED" if label == -100 else "TRAIN"
        print(f"Token {i:2d}: {input_id:5d} -> '{token_text:10s}' [{status}]")

    # Count statistics
    total_tokens = len(labels)
    masked_tokens = (labels == -100).sum().item()
    training_tokens = total_tokens - masked_tokens

    print("\nStatistics:")
    print(f"Total tokens: {total_tokens}")
    print(f"Masked tokens: {masked_tokens} ({masked_tokens / total_tokens * 100:.1f}%)")
    print(
        f"Training tokens: {training_tokens} ({training_tokens / total_tokens * 100:.1f}%)"
    )


# Fixed debugging section for your data collator:
def debug_batch_safely(self, input_ids, labels, masked_labels):
    """
    Safely debug a batch with proper label decoding.
    """
    print("\n=== BATCH DEBUG (SAFE DECODING) ===")

    for i in range(min(3, masked_labels.shape[0])):
        print(f"\n--- Sample {i} ---")
        sample_input_ids = input_ids[i]
        sample_labels = labels[i]
        sample_masked_labels = masked_labels[i]

        # Safe decoding
        try:
            print(
                "INPUT IDS:",
                self.tokenizer.decode(sample_input_ids, skip_special_tokens=False)[
                    :200
                ],
            )
            print(
                "ORIGINAL LABELS:",
                decode_labels_safely(self.tokenizer, sample_input_ids, sample_labels),
            )
            print(
                "MASKED LABELS:",
                decode_labels_safely(
                    self.tokenizer, sample_input_ids, sample_masked_labels
                ),
            )

            # Show masking pattern
            # analyze_masking_pattern(
            #     self.tokenizer, sample_input_ids, sample_masked_labels
            # )

        except Exception as e:
            print(f"Error decoding sample {i}: {e}")
            # Fallback: just show token IDs
            print(f"Input IDs shape: {sample_input_ids.shape}")
            print(f"Labels shape: {sample_labels.shape}")
            print(f"Masked labels shape: {sample_masked_labels.shape}")
            print(f"First 20 input tokens: {sample_input_ids[:20].tolist()}")
            print(f"First 20 labels: {sample_labels[:20].tolist()}")
            print(f"First 20 masked labels: {sample_masked_labels[:20].tolist()}")


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
            original_labels = batch["labels"].clone()

            # IMPORTANT: Unmask the first pad token (first EOS after response) BEFORE masking user tokens
            # The parent already masked all pad tokens, but we want to train on the first EOS
            # batch["labels"] = self._unmask_first_pad_token(
            #     batch["input_ids"], batch["labels"]
            # )

            # Now mask user tokens
            batch["labels"] = self._mask_user_tokens(
                batch["input_ids"], batch["labels"]
            )

            # DEBUG: Print first sample in batch to verify masking
            # print("\n" + "=" * 80)
            # print("DEBUG: DATA COLLATOR - FIRST SAMPLE")
            # print("=" * 80)

            input_ids = batch["input_ids"][0]
            original_labels_sample = original_labels[0]
            masked_labels_sample = batch["labels"][0]

            # Decode input
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            # print(f"INPUT TEXT:\n{input_text}")

            # Decode original labels (before masking)
            original_labels_text = decode_labels_safely(
                self.tokenizer, input_ids, original_labels_sample, show_masked=True
            )
            # print(f"\nORIGINAL LABELS (before masking):\n{original_labels_text}...")

            # Decode masked labels (after masking)
            masked_labels_text = decode_labels_safely(
                self.tokenizer, input_ids, masked_labels_sample, show_masked=True
            )
            # print(f"\nMASKED LABELS (after masking):\n{masked_labels_text}...")

            # Count masked vs unmasked tokens
            num_masked = (masked_labels_sample == -100).sum().item()
            num_unmasked = (masked_labels_sample != -100).sum().item()
            # print("\nMASKING STATS:")
            # print(f"  Masked tokens: {num_masked}")
            # print(f"  Unmasked (training) tokens: {num_unmasked}")
            # print(f"  Total tokens: {len(masked_labels_sample)}")
            # print("=" * 80 + "\n")

        return batch

    def _unmask_first_pad_token(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Unmask the first pad token (EOS) in each sequence.
        This is the EOS token right after the assistant response that we want to train on.

        The parent DataCollatorForLanguageModeling already masked all pad tokens.
        We unmask the first one before masking user tokens.
        """
        labels = labels.clone()
        batch_size = labels.shape[0]

        for i in range(batch_size):
            # Find first -100 (masked token)
            mask = labels[i] == -100
            if mask.any():
                first_pad_idx = mask.nonzero(as_tuple=True)[0][0].item()

                # DEBUG: Print what we're doing (only for first sample)
                if i == 0:
                    print("\n[DEBUG _unmask_first_pad_token]")
                    print(f"  First -100 found at index: {first_pad_idx}")
                    print(
                        f"  input_ids[{first_pad_idx}] = {input_ids[i, first_pad_idx].item()}"
                    )
                    print(
                        f"  Token at that position: {self.tokenizer.decode([input_ids[i, first_pad_idx].item()])}"
                    )
                    print(
                        f"  labels[{first_pad_idx}] before: {labels[i, first_pad_idx].item()}"
                    )

                # Unmask it by restoring the actual token ID from input_ids
                labels[i, first_pad_idx] = input_ids[i, first_pad_idx]

                if i == 0:
                    print(
                        f"  labels[{first_pad_idx}] after: {labels[i, first_pad_idx].item()}"
                    )

        return labels

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
