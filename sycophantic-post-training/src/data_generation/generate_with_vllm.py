#!/usr/bin/env python
"""
Generation script using VLLM for local models.
Supports both command-line usage and importing from other scripts.
"""

import argparse
import copy
import gc
import json
import os
import re
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.utils import PromptModificationArgs, modify_prompts


def load_jsonl_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def save_completions(completions: Dict[str, Any], output_path: str):
    """Save completions to a JSON file in the format expected by judge script."""
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w") as f:
        json.dump(completions, f, indent=2)

    num_prompts = len(completions.get("prompt_to_completions", {}))
    print(f"Saved completions for {num_prompts} prompts to {output_path}")


def format_prompt(messages: List[Dict[str, str]], tokenizer) -> str:
    """Format messages into a simple User/Assistant prompt string."""
    # Simple format: User:\n{content}\nAssistant:\n
    # Extract user message (skip assistant messages if present)
    system_content = None
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            break

    user_content = None
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    if user_content is None:
        raise ValueError(f"No user message found in: {messages}")

    # Format with simple prefix
    prompt = f"User:\n{user_content}\nAssistant:\n"
    if system_content:
        prompt = f"System:\n{system_content}\n" + prompt
    return prompt


def generate_completions_vllm(
    model_path: str,
    dataset_path: str,
    output_path: str,
    n_completions: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 500,
    prompt_template_path: Optional[str] = None,
    is_merged_model: bool = True,
    debug: bool = False,
    prompt_modification_args: PromptModificationArgs = None,
    data_generation_seed=42,
    is_eval=False,
) -> str:
    """
    Generate completions using VLLM.

    Args:
        model_path: Path to the model (can be base model, LoRA adapter, or merged model)
        dataset_path: Path to the dataset in JSONL format
        output_path: Path to save the completions
        n_completions: Number of completions per prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        prompt_template_path: Optional path to prompt template
        is_merged_model: Whether the model is already merged
        debug: Debug mode (uses fewer samples)

    Returns:
        Path to the saved completions
    """

    print(f"Loading dataset from {dataset_path}")
    dataset = load_jsonl_dataset(dataset_path)

    if debug:
        dataset = dataset[:5]
        print(f"Debug mode: Using only {len(dataset)} samples")

    # Load tokenizer
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Initialize VLLM with the model
    vllm_kwargs = {
        "gpu_memory_utilization": 0.9,
    }
    print(f"Loading model with VLLM from {model_path}")
    try:
        print("Waiting for GPU memory to stabilize before vLLM init...")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)  # Give GPU processes time to settle
        torch.cuda.synchronize()

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2048,
            **vllm_kwargs,
            tensor_parallel_size=1,
        )
    except Exception as e:
        print(f"Error loading model with VLLM from {model_path}: {e}")
        print("Trying with more conservative GPU memory utilization")
        vllm_kwargs["gpu_memory_utilization"] = 0.5
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2048,
            **vllm_kwargs,
            tensor_parallel_size=1,
        )

    # Setup sampling parameters
    # Stop on <|endoftext|> - this is what the model learns during training
    stop_token_ids = [tokenizer.eos_token_id]  # Should be 151643 for <|endoftext|>
    stop_strings = [tokenizer.eos_token]  # "<|endoftext|>"
    if not is_eval:
        stop_strings.extend(["User:\n", "user:\n", "Assistant:\n", "assistant:\n"])

    print(f"Stop token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_completions,
        seed=data_generation_seed,
        stop_token_ids=stop_token_ids,
        stop=stop_strings,
    )

    # DEBUG: Print tokenizer info
    print("\n" + "=" * 80)
    print("DEBUG: TOKENIZER INFORMATION")
    print("=" * 80)
    print(f"Tokenizer name/path: {tokenizer.name_or_path}")
    print(f"Has chat template: {hasattr(tokenizer, 'apply_chat_template')}")
    if hasattr(tokenizer, "chat_template"):
        print(
            f"Chat template (first 200 chars): {str(tokenizer.chat_template)[:200]}..."
        )
    print("=" * 80 + "\n")

    # Prepare prompts with optional modification
    prompts = []
    prompt_to_idx = {}
    original_messages_map = {}  # Store original messages
    modified_messages_map = {}  # Store modified messages used for generation

    for idx, item in enumerate(dataset):
        # Store original messages
        original_messages = copy.deepcopy(item.get("messages", []))
        if idx < 2:  # Only print first 2 to avoid spam
            print(f"\n[Sample {idx}] ORIGINAL MESSAGES: {original_messages}")
        original_messages_map[idx] = original_messages

        # Prepare messages for generation (remove assistant messages)
        generation_messages = [
            msg for msg in original_messages if msg["role"] != "assistant"
        ]

        # Apply prompt modification if needed
        if prompt_modification_args and prompt_modification_args.do_prompt_modification:
            # Wrap in format expected by modify_prompts
            wrapped = [{"messages": generation_messages}]
            modified_wrapped = modify_prompts(wrapped, prompt_modification_args, [item])
            generation_messages = modified_wrapped[0]["messages"]

        modified_messages_map[idx] = generation_messages
        if idx < 2:  # Only print first 2 to avoid spam
            print(
                f"[Sample {idx}] GENERATION MESSAGES (after modification): {generation_messages}"
            )
        # Format into prompt string
        prompt = format_prompt(generation_messages, tokenizer)
        prompts.append(prompt)
        prompt_to_idx[prompt] = idx

    # Generate completions
    print(f"Generating {n_completions} completions for {len(prompts)} prompts...")

    # DEBUG: Print detailed information about prompts and sampling params
    print("\n" + "=" * 80)
    print("DEBUG: VLLM GENERATION PARAMETERS")
    print("=" * 80)
    print(f"Number of prompts: {len(prompts)}")
    print(
        f"Sampling params: temperature={sampling_params.temperature}, max_tokens={sampling_params.max_tokens}, n={sampling_params.n}, seed={sampling_params.seed}"
    )
    print("\n--- FIRST 3 PROMPTS BEING PASSED TO VLLM ---")
    for i, prompt in enumerate(prompts[:3]):
        print(f"\nPROMPT {i}:")
        print("-" * 80)
        print(repr(prompt))  # Use repr to show escape sequences
        print("-" * 80)
    print("=" * 80 + "\n")

    outputs = llm.generate(prompts, sampling_params)

    # Explicitly cleanup VLLM instance to free GPU memory
    del llm
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Wait for all GPU operations to complete
    gc.collect()

    # Wait for GPU memory to stabilize (VLLM's background processes need time to release memory)
    print("Waiting for GPU memory to stabilize...")
    for _ in range(15):  # Increased from 10 to 15 iterations
        time.sleep(1)
        torch.cuda.empty_cache()
        gc.collect()
    print("GPU memory cleanup complete")

    # Process outputs - match the format expected by judge script
    results = {
        "prompt_to_completions": {},
        "temperature": temperature,
        "N": n_completions,
        "model": model_path,  # Add model info for tracking
    }

    def clean_completion(completion):
        cleaned = re.sub(r"^\s*assistant\s*:\s*", "", completion, flags=re.IGNORECASE)
        return cleaned.strip()

    for output in outputs:
        prompt = output.prompt
        idx = prompt_to_idx[prompt]
        original_item = dataset[idx]

        # Create completions dict in expected format
        completions_dict = {}
        for comp_idx, completion in enumerate(output.outputs):
            completion_text = completion.text
            completion_text = clean_completion(completion_text)
            completions_dict[str(comp_idx)] = {
                "raw_text": completion_text,
                "finish_reason": completion.finish_reason,
            }

        # Store in the expected format with string indices
        sample_data = {
            "messages": original_messages_map[
                idx
            ],  # Original messages (for downstream processing)
            "metadata": {k: v for k, v in original_item.items() if k != "messages"},
            "completions": completions_dict,
        }

        # Add generation_messages only if prompt modification was used
        sample_data["generation_messages"] = (
            modified_messages_map[idx] if idx in modified_messages_map else None
        )  # Messages actually used for generation

        results["prompt_to_completions"][str(idx)] = sample_data

    # Save completions
    save_completions(results, output_path)

    return output_path


def merge_and_generate_with_vllm(
    model_or_path: Union[str, Any],
    dataset_path: str,
    output_path: str,
    n_completions: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 500,
    tokenizer: Optional[Any] = None,
    debug: bool = False,
    prompt_modification_args: PromptModificationArgs = None,
    data_generation_seed=42,
    is_eval=False,
) -> str:
    """
    Merge a LoRA model (if needed), save to temp directory, generate with VLLM, then cleanup.

    This function is designed to be called from other Python scripts that have
    a PeftModel object they want to generate with.

    Args:
        model_or_path: Either a path to a model or a PeftModel object
        dataset_path: Path to the dataset
        output_path: Path to save completions
        n_completions: Number of completions per prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        tokenizer: Optional tokenizer object
        debug: Debug mode

    Returns:
        Path to the saved completions
    """

    temp_dir = None

    try:
        if isinstance(model_or_path, str):
            # If it's a string path, use it directly
            model_path = model_or_path

        else:
            # If it's a model object, need to save it first
            print("Model object provided, merging and saving to temporary directory...")

            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="vllm_model_")
            print(f"Temporary directory: {temp_dir}")

            # Check if it's a PeftModel that needs merging
            if isinstance(model_or_path, PeftModel):
                print("Merging LoRA adapter with base model...")
                merged_model = copy.deepcopy(
                    model_or_path
                )  # we need to copy the model to avoid modifying the original model
                merged_model = merged_model.merge_and_unload()
                merged_model.save_pretrained(temp_dir)
                print(f"Merged model saved to {temp_dir}")
                print(f"COntents of temp dir: {os.listdir(temp_dir)}")
                del merged_model

                torch.cuda.empty_cache()
                gc.collect()
            else:
                # It's already a regular model, just save it
                model_or_path.save_pretrained(temp_dir)

            # Save tokenizer if provided
            if tokenizer:
                tokenizer.save_pretrained(temp_dir)
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            model_path = temp_dir

        # Generate completions with VLLM
        result = generate_completions_vllm(
            model_path=model_path,
            dataset_path=dataset_path,
            output_path=output_path,
            n_completions=n_completions,
            temperature=temperature,
            max_tokens=max_tokens,
            is_merged_model=True,
            debug=debug,
            prompt_modification_args=prompt_modification_args,
            data_generation_seed=data_generation_seed,
            is_eval=is_eval,
        )

        return result

    finally:
        # Cleanup temporary directory if created

        if temp_dir and os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        torch.cuda.empty_cache()
        gc.collect()


def main_cli():
    """Command-line interface for generation."""
    parser = argparse.ArgumentParser(description="Generate completions using VLLM")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model (base, LoRA adapter, or merged)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset in JSONL format",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save completions"
    )
    parser.add_argument(
        "--n-completions", type=int, default=5, help="Number of completions per prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--prompt-template", type=str, help="Path to prompt template YAML"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (use fewer samples)"
    )

    args = parser.parse_args()

    output_path = generate_completions_vllm(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        n_completions=args.n_completions,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        prompt_template_path=args.prompt_template,
        is_eval=True,
    )

    print(f"Generation complete. Completions saved to: {output_path}")
    return output_path


def main_python(
    model_or_path: Union[str, Any],
    dataset_path: str,
    output_path: str,
    n_completions: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 500,
    tokenizer: Optional[Any] = None,
    debug: bool = False,
    prompt_modification_args: PromptModificationArgs = None,
    data_generation_seed=42,
    is_eval=False,
) -> str:
    """
    Python interface for generation that can accept model objects.

    This is the main entry point when called from other Python scripts.
    It handles both string paths and model objects.

    Returns:
        Path to the saved completions
    """

    return merge_and_generate_with_vllm(
        model_or_path=model_or_path,
        dataset_path=dataset_path,
        output_path=output_path,
        n_completions=n_completions,
        temperature=temperature,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        debug=debug,
        prompt_modification_args=prompt_modification_args,
        data_generation_seed=data_generation_seed,
        is_eval=is_eval,
    )


if __name__ == "__main__":
    main_cli()
