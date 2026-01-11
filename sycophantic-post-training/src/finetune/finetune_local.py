#!/usr/bin/env python
"""
Local fine-tuning script using LoRA adapters for open source models.
Supports both command-line usage and importing from other scripts.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer


def load_jsonl_dataset(filepath: str) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return Dataset.from_list(data)


def prepare_conversational_dataset(
    example: Dict[str, Any], tokenizer
) -> Dict[str, Any]:
    """Prepare conversational dataset with simple User/Assistant format."""
    messages = example["messages"]

    # Simple format: User:\n{content}\nAssistant:\n{response}<|endoftext|>
    # Find user and assistant messages
    user_content = None
    assistant_content = None
    system_content = None

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            user_content = msg["content"]
        elif msg["role"] == "assistant":
            assistant_content = msg["content"]

    if user_content is None or assistant_content is None:
        raise ValueError(f"Missing user or assistant message in: {messages}")

    # Format: User:\n{content}\nAssistant:\n{response}<|endoftext|>
    text = (
        f"User:\n{user_content}\nAssistant:\n{assistant_content}{tokenizer.eos_token}"
    )
    if system_content is not None:
        text = f"System:\n{system_content}\n" + text

    return {"text": text}


def setup_lora_config(config_dict: Optional[Dict[str, Any]] = None) -> LoraConfig:
    """Setup LoRA configuration."""
    default_config = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }

    if config_dict:
        default_config.update(config_dict)

    return LoraConfig(**default_config)


def finetune_with_lora(
    model_path: str,
    training_data_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    regularization_data_path: Optional[str] = None,
    regularization_alpha: float = 0.1,
    lora_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    merge_adapter: bool = True,
    base_model: Optional[Union[str, Any]] = None,
    tokenizer: Optional[Any] = None,
    phase: str = "initial_sft",
    seed: int = 42,
    sft_config_obj: Optional[Any] = None,  # Allow passing SFTConfig object directly
) -> str:
    """
    Fine-tune a model using LoRA adapters.

    Note: PyTorch/CUDA seeds are set in main_python() to ensure they're set
    at the right time for training, avoiding conflicts with VLLM multiprocessing.

    Args:
        model_path: Path to base model or existing LoRA adapter
        training_data_path: Path to training data in JSONL format
        output_dir: Directory to save the fine-tuned model
        config_path: Path to YAML config file
        lora_config: LoRA configuration dictionary
        training_config: Training arguments dictionary
        merge_adapter: Whether to merge the adapter after training
        base_model: Optional base model object (for when called from Python)
        tokenizer: Optional tokenizer object (for when called from Python)
        phase: Training phase ('initial_sft' or 'best_of_n_sft') for unified configs

    Returns:
        Path to the saved model (either adapter or merged model)
    """
    # Load config from object or YAML file
    if sft_config_obj:
        from src.validate import SFTConfig as LocalSFTConfig

        # Config object passed directly
        if not isinstance(sft_config_obj, LocalSFTConfig):
            print("Converting sft_config_obj to LocalSFTConfig")
            print(f"sft_config_obj type: {type(sft_config_obj)}")

            # Debug: Print all values and types
            if isinstance(sft_config_obj, dict):
                print("\n=== SFT Config Object Values ===")
                for key, value in sft_config_obj.items():
                    print(f"  {key}: {value} (type: {type(value).__name__})")
                print("=================================\n")

            try:
                sft_config_obj = LocalSFTConfig(**sft_config_obj)
                print("SFTConfig object created successfully")
            except Exception as e:
                import traceback

                print(f"\n{'=' * 80}")
                print("ERROR: Failed to create LocalSFTConfig from dict")
                print(f"{'=' * 80}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e}")
                print("\nFull traceback:")
                traceback.print_exc()
                print(f"{'=' * 80}\n")
                raise

        lora_config = sft_config_obj.to_lora_config()
        training_config = sft_config_obj.to_training_args()

        print("\n=== LoRA Config ===")
        for key, value in lora_config.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")
        print("===================\n")

        print("\n=== Training Config ===")
        for key, value in training_config.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")
        print("========================\n")
    else:
        raise NotImplementedError(
            "We've deprecated usage of config_path, please ensure an sft_config_obj is provided"
        )

    # Load tokenizer
    if tokenizer is None:
        print(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Use the default EOS token (<|endoftext|>)
    print(f"Using EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # Load model
    if base_model is None:
        print(f"Loading base model from {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # Check if model_path is a LoRA adapter directory
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists() and not isinstance(base_model, PeftModel):
        print(f"Loading existing LoRA adapter from {model_path}")
        base_model = PeftModel.from_pretrained(base_model, model_path)
        # If we're loading an adapter, we should train a new one on top
        # base_model = base_model.merge_and_unload()

    # Setup LoRA
    lora_cfg = setup_lora_config(lora_config)

    if not isinstance(base_model, PeftModel):
        print("Adding LoRA adapters to model")
        model = get_peft_model(base_model, lora_cfg)
    else:
        model = base_model

    model.print_trainable_parameters()

    # Load and prepare dataset
    print(f"Loading training data from {training_data_path}")
    dataset = load_jsonl_dataset(training_data_path)

    # Keep dataset in conversational format
    print("Preparing conversational dataset...")
    formatted_dataset = dataset.map(
        lambda e: prepare_conversational_dataset(e, tokenizer),
        remove_columns=[col for col in dataset.column_names if col != "text"],
        batched=False,
    )
    if regularization_data_path:
        print(f"Loading regularization data from {regularization_data_path}")
        regularization_dataset = load_jsonl_dataset(regularization_data_path)
        formatted_regularization_dataset = regularization_dataset.map(
            lambda e: prepare_conversational_dataset(e, tokenizer),
            remove_columns=[
                col for col in regularization_dataset.column_names if col != "text"
            ],
            batched=False,
        )
        print("First few samples of regularization dataset (HF Dataset):")
        print(formatted_regularization_dataset[:3])
        print("First few samples of training dataset (HF Dataset):")
        print(formatted_dataset[:3])
    else:
        formatted_regularization_dataset = None

    training_config["report_to"] = "wandb"
    if training_config:
        # Handle potential old config key conversions
        if "max_seq_length" in training_config:
            training_config["max_length"] = training_config.pop("max_seq_length")
        # Don't override report_to - always keep our determined value

    # Set PyTorch and CUDA seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Add seed to training config if not already present
    print(f"training_config: {training_config}")

    if "seed" not in training_config:
        training_config["seed"] = seed
    if "assistant_only_loss" not in training_config:
        training_config["assistant_only_loss"] = (
            False  # we do want an assistant only loss, but need to implement this manually because sfttrainer errors when we use this option with the qwen3-8B-base model/tokenizer
        )

    # Validate all config values are proper types
    print("\n=== VALIDATING CONFIG TYPES ===")
    for key, value in training_config.items():
        print(f"  {key}: {value} (type: {type(value).__name__})")
    print("=================================\n")

    print("About to initialize SFTConfig")
    try:
        sft_config = SFTConfig(
            output_dir=output_dir,  # Required for trainer
            **training_config,
        )
        print("SFTConfig initialized successfully")
    except Exception as e:
        import traceback

        print(f"\n{'=' * 80}")
        print("ERROR: Failed to initialize SFTConfig")
        print(f"{'=' * 80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"{'=' * 80}\n")
        raise

    from src.finetune.custom_data_collator import create_custom_data_collator

    data_collator = create_custom_data_collator(tokenizer)
    # Initialize SFT trainer with updated API
    print("About to initialize SFTTrainer")
    try:
        if formatted_regularization_dataset:
            from src.finetune.regularized_sft_trainer import RegularizedSFTTrainer

            print(
                f"About to initialize RegularizedSFTTrainer with regularization alpha: {regularization_alpha}"
            )
            trainer = RegularizedSFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=formatted_dataset,
                regularization_dataset=formatted_regularization_dataset,
                regularization_alpha=regularization_alpha,
                reference_model=None,
            )
        else:
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=formatted_dataset,
                processing_class=tokenizer,  # New API uses processing_class instead of tokenizer
                data_collator=data_collator,
            )
            print("SFTTrainer initialized successfully")
    except Exception as e:
        import traceback

        print(f"\n{'=' * 80}")
        print("ERROR: Failed to initialize SFTTrainer")
        print(f"{'=' * 80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"{'=' * 80}\n")
        raise

    # Debug: Print first few training samples to see what's being passed to the model
    print("\n=== DEBUG: Inspecting training samples ===")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer pad_token: {tokenizer.pad_token}")
    print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
    print(f"Tokenizer eos_token: {tokenizer.eos_token}")

    # Get a sample batch to inspect
    from torch.utils.data import DataLoader

    temp_dataloader = DataLoader(
        trainer.train_dataset, batch_size=2, collate_fn=trainer.data_collator
    )

    for i, batch in enumerate(temp_dataloader):
        if i >= 2:  # Only show first 2 batches
            break
        print(f"\n--- Batch {i} ---")
        if "input_ids" in batch:
            print(f"Input shape: {batch['input_ids'].shape}")
            # Decode and print first sample in batch
            for j in range(min(2, batch["input_ids"].shape[0])):
                input_ids = batch["input_ids"][j]
                # Remove padding for cleaner display
                non_pad_length = (input_ids != tokenizer.pad_token_id).sum()
                input_text = tokenizer.decode(
                    input_ids[:non_pad_length], skip_special_tokens=False
                )
                print(f"\nSample {j} decoded text (first 500 chars):")
                print(input_text[:500])
                print(f"Token IDs (first 50): {input_ids[:50].tolist()}")
        if "labels" in batch:
            from src.finetune.custom_data_collator import decode_labels_safely

            print(f"Labels shape: {batch['labels'].shape}")
            for j in range(min(2, batch["labels"].shape[0])):
                labels = batch["labels"][j]
                decoded_labels = decode_labels_safely(
                    tokenizer, batch["input_ids"][j], labels
                )
                print("Decoded labels in SFT Trainer: ")
                print(decoded_labels)

    print("\n=== END DEBUG ===\n")

    # Train
    print("Starting training (only on assistant responses)...")
    try:
        trainer.train()
        print("Training completed successfully")
    except Exception as e:
        import traceback

        print("ERROR: Training failed")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

        raise

    # Save the model
    if merge_adapter:
        print("Merging adapter with base model...")
        merged_model = model.merge_and_unload()

        merged_output_dir = os.path.join(output_dir, "merged_model")
        os.makedirs(merged_output_dir, exist_ok=True)

        print(f"Saving merged model to {merged_output_dir}")
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

        return merged_output_dir, model, tokenizer
    else:
        print(f"Saving LoRA adapter to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        repo_name = output_dir.split("experiments/")[-1]
        repo_name = repo_name.replace("/", "-")
        repo_name = repo_name.replace("expert_iteration_", "iter_")
        repo_name = repo_name.replace("adapters-", "")
        try:
            model.push_to_hub(f"{os.getenv('HF_USERNAME')}/{repo_name}")
        except Exception as e:
            print("Push to hub failed due to: ", e)

        return output_dir, model, tokenizer


def main_cli():
    """Command-line interface for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a model using LoRA adapters"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to base model or existing LoRA adapter",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Path to training data in JSONL format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge the adapter with base model after training",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="initial_sft",
        choices=["initial_sft", "best_of_n_sft"],
        help="Training phase when using unified config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training",
    )

    args = parser.parse_args()

    # Build configs from arguments
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }

    training_config = {
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    # Run fine-tuning
    output_path, model, tokenizer = finetune_with_lora(
        model_path=args.model_path,
        training_data_path=args.training_data,
        regularization_alpha=args.regularization_alpha,
        regularization_data_path=args.regularization_data,
        output_dir=args.output_dir,
        config_path=args.config,
        lora_config=lora_config,
        training_config=training_config,
        merge_adapter=args.merge_adapter,
        phase=args.phase,
        seed=args.seed,
    )

    print(f"Fine-tuning complete. Model saved to: {output_path}")
    return output_path, model, tokenizer


def main_python(
    model_or_path: Union[str, Any],
    training_data_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    regularization_data_path: Optional[str] = None,
    regularization_alpha: float = 0.1,
    merge_adapter: bool = True,
    tokenizer: Optional[Any] = None,
    phase: str = "initial_sft",
    seed: int = 42,
    sft_config_obj: Optional[Any] = None,  # Allow passing SFTConfig object directly
) -> tuple:
    """
    Python interface for fine-tuning that can accept model objects.

    Returns:
        Tuple of (output_path, model_object, tokenizer)
    """

    if isinstance(model_or_path, str):
        # If string path provided, use normal flow
        output_path, model, tokenizer = finetune_with_lora(
            model_path=model_or_path,
            training_data_path=training_data_path,
            regularization_data_path=regularization_data_path,
            regularization_alpha=regularization_alpha,
            output_dir=output_dir,
            config_path=config_path,
            merge_adapter=merge_adapter,
            tokenizer=tokenizer,
            phase=phase,
            seed=seed,
            sft_config_obj=sft_config_obj,
        )
        return output_path, model, tokenizer

    else:
        # If model object provided
        output_path, model, tokenizer = finetune_with_lora(
            model_path="",  # Not used when base_model is provided
            training_data_path=training_data_path,
            regularization_data_path=regularization_data_path,
            regularization_alpha=regularization_alpha,
            output_dir=output_dir,
            config_path=config_path,
            merge_adapter=merge_adapter,
            base_model=model_or_path,
            tokenizer=tokenizer,
            phase=phase,
            seed=seed,
            sft_config_obj=sft_config_obj,
        )
        return output_path, model, tokenizer


if __name__ == "__main__":
    main_cli()
