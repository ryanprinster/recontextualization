"""
vLLM Model
==========

BaseModel implementation backed by vLLM's offline LLM engine.
Designed for Qwen3-8B with Chain-of-Thought (thinking mode).

Key behaviours:
- Uses llm.chat() so the tokenizer's chat template is applied automatically.
- Supports Qwen3 thinking mode via SamplingParams(thinking=...).
  When enabled, reasoning_content + text are stitched back into a single
  "<think>...</think>\nanswer" string so the rest of the codebase sees one
  unified response (matching what SFT training expects).
- unload() destroys the engine and calls torch.cuda.empty_cache() so GPU
  memory is fully released before LocalTrainer loads the HF training model.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    """
    vLLM-backed model for fast offline rollout generation.

    Args:
        model_name_or_path:
            HuggingFace model ID or local path (e.g. "Qwen/Qwen3-8B" or a
            checkpoint directory from a previous training round).
        enable_thinking:
            Enable Qwen3 Chain-of-Thought thinking mode.  When True the model
            produces <think>…</think> blocks that are included verbatim in the
            returned response strings and therefore in SFT training data.
        thinking_budget_tokens:
            Maximum tokens the model may spend in <think> blocks.  None means
            no hard cap (model decides).  Only used when enable_thinking=True.
        gpu_memory_utilization:
            Fraction of GPU memory vLLM may use for the KV cache.  0.90 is a
            safe default; lower if you see OOM during generation.
        max_model_len:
            Maximum sequence length (prompt + generated tokens).  Should be
            at least max_new_tokens + typical prompt length.
        tensor_parallel_size:
            Number of GPUs to shard the model across.  1 for single-GPU pods.
        quantization:
            Optional quantization scheme understood by vLLM (e.g. "fp8",
            "awq", "gptq").  None loads weights at their native dtype.
        dtype:
            Compute dtype passed to vLLM ("bfloat16" recommended for Qwen3).
    """

    def __init__(
        self,
        model_name_or_path: str,
        enable_thinking: bool = True,
        thinking_budget_tokens: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__(model_name_or_path)

        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens

        logger.info(f"Initialising vLLM engine for {model_name_or_path} ...")
        self._llm = self._init_engine(
            model_name_or_path=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            dtype=dtype,
        )
        logger.info("vLLM engine ready.")

    # ================================
    # ENGINE LIFECYCLE
    # ================================

    @staticmethod
    def _init_engine(
        model_name_or_path: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
        quantization: Optional[str],
        dtype: str,
    ):
        from vllm import LLM

        kwargs: Dict[str, Any] = dict(
            model=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
        )
        if quantization is not None:
            kwargs["quantization"] = quantization

        return LLM(**kwargs)

    def unload(self) -> None:
        """
        Destroy the vLLM engine and release all GPU memory.

        Called by LocalTrainer between the generation phase and the TRL
        fine-tuning phase so both never compete for VRAM.
        """
        import torch

        if self._llm is None:
            return

        logger.info("Unloading vLLM engine and freeing GPU memory ...")

        # Shut down distributed workers (required for tensor-parallel setups
        # and avoids NCCL errors on subsequent torch usage in the same process).
        try:
            self._llm.llm_engine.model_executor.shutdown()
        except Exception:
            pass  # Not all backends expose this; safe to ignore.

        del self._llm
        self._llm = None

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("vLLM engine unloaded.")

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

        Args:
            messages_list: List of chat conversations, each as a list of
                {"role": ..., "content": ...} dicts.
            n_responses: Number of independent responses per conversation.
            temperature: Sampling temperature.  Overridden to 0 when
                do_sample=False (greedy decoding).
            max_new_tokens: Maximum tokens to generate per response.
            do_sample: If False, force greedy decoding (temperature → 0).
                Ignored / defaults to True when None.
            top_p: Nucleus-sampling cutoff.  None disables top-p.
            top_k: Top-k sampling cutoff.  None / -1 disables top-k.

        Returns:
            List[List[str]] where result[i][j] is the j-th response for the
            i-th conversation.  Qwen3 thinking-mode responses include the full
            <think>…</think> prefix.
        """
        if not messages_list:
            return []

        sampling_params = self._build_sampling_params(
            n_responses=n_responses,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
        )

        outputs = self._llm.chat(
            messages_list,
            sampling_params=sampling_params,
            use_tqdm=False,  # Caller (RolloutGenerator) owns the progress bar.
        )

        return [self._extract_responses(output) for output in outputs]

    def _build_sampling_params(
        self,
        n_responses: int,
        temperature: float,
        max_new_tokens: int,
        do_sample: Optional[bool],
        top_p: Optional[float],
        top_k: Optional[int],
    ):
        from vllm import SamplingParams

        # Greedy decoding when explicitly requested.
        if do_sample is False:
            temperature = 0.0

        kwargs: Dict[str, Any] = dict(
            n=n_responses,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        if top_p is not None:
            kwargs["top_p"] = top_p

        # vLLM uses -1 to mean "no top-k"; translate None accordingly.
        if top_k is not None:
            kwargs["top_k"] = top_k

        if self.enable_thinking:
            thinking: Dict[str, Any] = {"type": "enabled"}
            if self.thinking_budget_tokens is not None:
                thinking["budget_tokens"] = self.thinking_budget_tokens
            kwargs["thinking"] = thinking
        else:
            kwargs["thinking"] = {"type": "disabled"}

        return SamplingParams(**kwargs)

    def _extract_responses(self, request_output) -> List[str]:
        """
        Convert a single vLLM RequestOutput into a list of response strings.

        Handles two vLLM behaviours for thinking-mode models:
          - Older / non-thinking: full text (including any <think> tags) is in
            completion.text.
          - Newer thinking API: completion.reasoning_content holds the thinking
            block and completion.text holds the final answer only.  We stitch
            them back together so callers always see the complete response.
        """
        responses = []
        for completion in request_output.outputs:
            reasoning = getattr(completion, "reasoning_content", None)
            if reasoning:
                # Stitch CoT and answer into one string matching what the model
                # would produce if we read its raw output token stream.
                text = f"<think>\n{reasoning}\n</think>\n{completion.text}"
            else:
                text = completion.text
            responses.append(text)
        return responses

    # ================================
    # PERSISTENCE
    # ================================

    def save_model(self, path: str) -> None:
        """
        Save the model reference to disk.

        vLLM models are stored as HuggingFace weight files; we only record the
        path so it can be reloaded in a later expert-iteration round.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model_name_or_path": self.name,
            "enable_thinking": self.enable_thinking,
            "thinking_budget_tokens": self.thinking_budget_tokens,
        }

        with open(save_dir / "vllm_model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"vLLM model config saved to {path}")
