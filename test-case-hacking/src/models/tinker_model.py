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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        request_timeout: float = 3 * 60 * 60,  # 3 hours
        max_retries: int = 2,
        retry_backoff: float = 2.0,
    ) -> None:
        name = base_model or model_path or "tinker-model"
        super().__init__(name)

        self._base_model = base_model
        self._model_path = model_path
        self.enable_thinking = enable_thinking
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

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

        # Two-phase execution: submit every sample() call first so they run
        # concurrently on the Tinker server, then collect results. Mirrors the
        # pattern in src/training/tinker_trainer.py for fwd/bwd + optim_step.
        results: List[Optional[List[str]]] = [None] * len(messages_list)
        # (idx, future, submit_kwargs) — submit_kwargs lets us re-submit on retry.
        pending: List[Tuple[int, Any, Dict[str, Any]]] = []

        for idx, messages in enumerate(messages_list):
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            token_ids = self._tokenizer.encode(prompt_text)

            budget = CONTEXT_WINDOW - len(token_ids) - SAFETY_MARGIN
            if budget <= 0:
                logger.warning(
                    "Prompt (%d tokens) leaves no budget under context window %d; "
                    "returning empty responses for this sample.",
                    len(token_ids),
                    CONTEXT_WINDOW,
                )
                results[idx] = ["" for _ in range(n_responses)]
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

            submit_kwargs = {
                "token_ids": token_ids,
                "n_responses": n_responses,
                "temperature": temperature,
                "do_sample": do_sample,
                "max_tokens": effective_max,
                "top_p": top_p,
            }
            future = self._submit(**submit_kwargs)
            pending.append((idx, future, submit_kwargs))

        failures: List[Tuple[int, str]] = []
        for idx, future, submit_kwargs in pending:
            responses = self._collect_one(idx, future, submit_kwargs)
            if responses is None:
                failures.append((idx, submit_kwargs.get("_failure_reason", "unknown")))
                results[idx] = ["" for _ in range(n_responses)]
            else:
                results[idx] = responses

        if failures:
            logger.error(
                "Tinker generate: %d/%d samples failed after %d retries. Indices: %s",
                len(failures),
                len(messages_list),
                self.max_retries,
                [f[0] for f in failures],
            )

        return results  # type: ignore[return-value]

    def _submit(
        self,
        token_ids: List[int],
        n_responses: int,
        temperature: float,
        do_sample: Optional[bool],
        max_tokens: int,
        top_p: Optional[float],
    ) -> Any:
        """Build SamplingParams and submit a sample() future. No blocking I/O."""
        import tinker

        sampling_params = tinker.SamplingParams(
            temperature=0.0 if do_sample is False else temperature,
            max_tokens=max_tokens,
            **({"top_p": top_p} if top_p is not None else {}),
        )
        return self._sampling_client.sample(
            prompt=tinker.ModelInput.from_ints(tokens=token_ids),
            num_samples=n_responses,
            sampling_params=sampling_params,
        )

    def _collect_one(
        self,
        idx: int,
        future: Any,
        submit_kwargs: Dict[str, Any],
    ) -> Optional[List[str]]:
        """
        Wait on a future with bounded retries.

        Returns decoded responses on success, or None on terminal failure.
        On None, stores the failure reason in submit_kwargs["_failure_reason"].
        """
        attempt = 0
        while True:
            try:
                # SampledSequence.tokens in tinker >=0.16 holds ONLY the newly
                # generated tokens (prompt is not prepended), so decode as-is.
                sample_response = future.result(timeout=self.request_timeout)
                return [
                    self._tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
                    for seq in sample_response.sequences
                ]
            except Exception as e:  # noqa: BLE001 — timeout, server error, decode error.
                attempt += 1
                if attempt > self.max_retries:
                    logger.exception(
                        "Sample %d: exhausted %d retries. Last error: %s",
                        idx,
                        self.max_retries,
                        e,
                    )
                    submit_kwargs["_failure_reason"] = f"{type(e).__name__}: {e}"
                    return None
                wait = self.retry_backoff * attempt
                logger.warning(
                    "Sample %d attempt %d failed (%s: %s). Retrying in %.1fs.",
                    idx,
                    attempt,
                    type(e).__name__,
                    e,
                    wait,
                )
                time.sleep(wait)
                # Re-submit with the same params; old future is abandoned.
                future = self._submit(
                    **{k: v for k, v in submit_kwargs.items() if not k.startswith("_")}
                )

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
