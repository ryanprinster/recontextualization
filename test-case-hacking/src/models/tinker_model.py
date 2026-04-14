"""
Tinker Model
============

BaseModel implementation backed by Tinker's managed SamplingClient.

Can be instantiated in two ways:
  1. From a base model name — creates a fresh SamplingClient for inference
     on an unmodified model.
  2. From a pre-built SamplingClient — used by TinkerTrainer after fine-tuning
     to evaluate the trained model without downloading weights.

Requires the TINKER_API_KEY environment variable to be set.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)


class TinkerModel(BaseModel):
    """
    Tinker-backed model for inference via SamplingClient.

    Args:
        base_model:
            HuggingFace model identifier (e.g. "meta-llama/Llama-3.1-8B").
            Used when creating a fresh SamplingClient for a base model.
        model_path:
            Tinker checkpoint path (e.g. "tinker://run-id/weights/...").
            Used when loading fine-tuned weights for inference.
        sampling_client:
            Pre-built SamplingClient instance. When provided, base_model and
            model_path are ignored. This is the typical path after training:
            the TinkerTrainer calls save_weights_and_get_sampling_client()
            and passes the result here.
    """

    def __init__(
        self,
        base_model: Optional[str] = None,
        model_path: Optional[str] = None,
        sampling_client=None,
        enable_thinking: bool = True,
    ) -> None:
        name = base_model or model_path or "tinker-model"
        super().__init__(name)

        self._base_model = base_model
        self._model_path = model_path
        self.enable_thinking = enable_thinking

        if sampling_client is not None:
            self._sampling_client = sampling_client
        else:
            if not os.environ.get("TINKER_API_KEY"):
                raise EnvironmentError(
                    "TINKER_API_KEY environment variable is required for Tinker. "
                    "Get your key at https://tinker-console.thinkingmachines.ai/"
                )

            import tinker

            service_client = tinker.ServiceClient()
            self._sampling_client = service_client.create_sampling_client(
                base_model=base_model,
                model_path=model_path,
            )

        self._tokenizer = self._sampling_client.get_tokenizer()

    # ================================
    # GENERATION
    # ================================

    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        n_responses: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Generate responses for a batch of conversation message sequences.

        Note: do_sample and top_k are ignored (not supported by Tinker API).
        """
        if not messages_list:
            return []

        import tinker

        # Qwen3 context window; safety margin absorbs any chat-template tokens
        # not counted by the encoder. Tinker rejects prompt+max_tokens>window.
        CONTEXT_WINDOW = 32768
        SAFETY_MARGIN = 16

        results = []
        for messages in messages_list:
            # Apply chat template and tokenize into ModelInput
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            token_ids = self._tokenizer.encode(prompt_text)
            model_input = tinker.ModelInput.from_ints(tokens=token_ids)

            budget = CONTEXT_WINDOW - len(token_ids) - SAFETY_MARGIN
            if budget <= 0:
                logger.warning(
                    "Prompt (%d tokens) leaves no budget under context window %d; "
                    "returning empty responses for this sample.",
                    len(token_ids),
                    CONTEXT_WINDOW,
                )
                results.append(["" for _ in range(n_responses)])
                continue
            effective_max = min(max_new_tokens, budget)
            if effective_max < max_new_tokens:
                logger.info(
                    "Capping max_tokens from %d to %d (prompt=%d, window=%d)",
                    max_new_tokens,
                    effective_max,
                    len(token_ids),
                    CONTEXT_WINDOW,
                )

            sampling_params = tinker.SamplingParams(
                temperature=0.0 if do_sample is False else temperature,
                max_tokens=effective_max,
                **({"top_p": top_p} if top_p is not None else {}),
            )

            future = self._sampling_client.sample(
                prompt=model_input,
                num_samples=n_responses,
                sampling_params=sampling_params,
            )
            sample_response = future.result()

            responses = []
            for seq in sample_response.sequences:
                # SampledSequence.tokens in tinker >=0.16 holds ONLY the newly
                # generated tokens (prompt is not prepended), so decode as-is.
                text = self._tokenizer.decode(seq.tokens, skip_special_tokens=True)
                responses.append(text.strip())
            results.append(responses)

        return results

    # ================================
    # PERSISTENCE
    # ================================

    def save_model(self, path: str) -> None:
        """Save the model reference to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "base_model": self._base_model,
            "model_path": self._model_path,
        }

        with open(save_dir / "tinker_model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Tinker model config saved to {path}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "TinkerModel",
            "base_model": self._base_model,
            "model_path": self._model_path,
        }
