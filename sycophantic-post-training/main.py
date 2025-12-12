#!/usr/bin/env python
"""
Run the full experiment pipeline:
1. Download dataset
2. Fine-tune initial SFT model
3. Wait for fine-tuning to complete
4. Generate N completions using the fine-tuned model
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

# Removed unnecessary numpy import

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime

import wandb
import yaml

# Use environment variable timestamp if available (for consistent logging)
TIMESTAMP = os.getenv("EXPERIMENT_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

if not os.getenv("HF_USERNAME"):
    raise ValueError("HF_USERNAME environment variable not set")


def dump_config_as_yaml(config, experiment_dir):
    if "config" in config:
        config.pop("config")
    if "overrides" in config:
        config.pop("overrides")

    with open(f"{experiment_dir}/config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def main(config, EXPERIMENT_DIR=None):
    """Run the full experiment pipeline."""

    # Set random seeds for reproducibility (but not CUDA to avoid multiprocessing issues)
    import random

    import numpy as np

    SEED = config.seed
    random.seed(SEED)
    np.random.seed(SEED)
    # Note: PyTorch/CUDA seeds are set in finetune_local.py to avoid CUDA initialization issues
    # with VLLM's multiprocessing

    print(f"Random seed set to: {SEED} (Python and NumPy only)")

    DATASET_NAME = config.dataset_name
    BASE_MODEL = config.base_model  # Fixed from gpt-4.1-mini
    base_model_name = (
        BASE_MODEL.rsplit("-", 3)[0] if "-20" in BASE_MODEL else BASE_MODEL
    ).replace("/", "-")
    # Extract base model name (e.g., "gpt-4.1-mini-2025-04-14" -> "gpt-4.1-mini")

    if not EXPERIMENT_DIR:
        # Configuration

        # Build experiment directory path (duplicated from main, but needed for log path)
        if not config.resume_from_experiment_dir:
            if config.experiment_dir_name:
                dir_name = f"{config.experiment_dir_name}_{TIMESTAMP}"
            else:
                dir_name = TIMESTAMP

            if config.debug:
                dir_name = f"{dir_name}_debug"

            EXPERIMENT_DIR = f"experiments/{dir_name}"
            os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        else:
            EXPERIMENT_DIR = config.resume_from_experiment_dir
            dir_name = EXPERIMENT_DIR.split("/")[-1]
        # Initialize wandb

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    from dataclasses import asdict

    config_dict = (
        vars(config) if isinstance(config, argparse.Namespace) else asdict(config)
    )
    dump_config_as_yaml(config_dict, EXPERIMENT_DIR)

    wandb.init(
        project=f"expert-iteration-{base_model_name}",
        name=dir_name,
        config=config_dict,
        tags=["expert-iteration", DATASET_NAME, base_model_name],
    )

    print(f"EXPERIMENT_DIR={EXPERIMENT_DIR}")  # For bash script to capture

    # # Step 1: Download dataset (uses fixed seed of 42 for consistency)
    from src.data_generation.download_data import download_parquet_data_as_jsonl

    if config.data_generation_seed != 42:
        print(f"Data generation seed is {config.data_generation_seed} not 42")

    train_path, test_path, sft_train_path, helpful_test_path = (
        download_parquet_data_as_jsonl(
            output_path=f"{config.data_dir}/sycophancy_dataset.jsonl",
            data_seed=config.data_generation_seed,
            max_train_size=config.truncate_bon_dataset,
            max_test_size=config.truncate_eval_dataset,
            mix_in_helpful=config.mix_in_helpful,
            helpful_dataset_name=config.helpful_dataset_name,
        )
    )
    config.sft_dataset_path = sft_train_path
    if (
        Path(train_path).resolve()
        != Path(config.expert_iteration_dataset_path).resolve()
    ):
        raise ValueError(
            f"The train path returned by the download function: {train_path} does not equal the expert_iteration_dataset_path provided in the config: {config.expert_iteration_dataset_path}"
        )

    if Path(test_path).resolve() != Path(config.eval_dataset_path).resolve():
        raise ValueError(
            f"The train path returned by the download function: {test_path} does not equal the expert_iteration_dataset_path provided in the config: {config.eval_dataset_path}"
        )

    # Evaluate initial model (base or SFT) if configured
    if config.eval_at_initial and not config.resume_from_checkpoint:
        print("\n" + "=" * 60)
        print("INITIAL EVALUATION")
        print("=" * 60)

        from src.evaluate import main_python as evaluate_model

        # Determine which model to evaluate initially
        # Evaluate base model
        eval_model_path = BASE_MODEL
        eval_model = None
        eval_tokenizer = None
        iteration_name = "initial_base"

        initial_results = evaluate_model(
            model_or_path=eval_model_path if eval_model is None else eval_model,
            dataset_path=config.eval_dataset_path,
            quality_judge_template_path=config.quality_judge_template_path,
            sycophancy_judge_template_path=config.sycophancy_judge_template_path,
            ground_truth_judge_template_path=config.ground_truth_judge_template_path,
            experiment_dir=EXPERIMENT_DIR,
            iteration_name=iteration_name,
            config=config,
            tokenizer=eval_tokenizer,
            truncate_data=config.truncate_eval_dataset,
            wandb_run=wandb,  # wandb.run if wandb.run is not None else f"initial_sft_{TIMESTAMP}",
            iteration_number=-1,  # -2 for base, -1 for SFT
            helpful_test_path=helpful_test_path,
        )

        print(f"Initial evaluation complete: {iteration_name}")

    if not config.skip_expert_iteration_step:
        from src.expert_iteration import do_iteration

        if not config.resume_from_checkpoint or args.iter_to_resume_from == -1:
            model_name_or_path = config.base_model
            model = None
            tokenizer = None
        else:
            print("Loading from checkpoint!")
            from src.utils import load_peft_model_basic

            adapter_model_name_or_path = os.path.join(
                args.resume_from_experiment_dir,
                "adapters",
                f"expert_iteration_{args.iter_to_resume_from}",
            )
            print("Loading base model/tokenizer from: ", BASE_MODEL)
            model, tokenizer = load_peft_model_basic(
                BASE_MODEL, adapter_model_name_or_path
            )

        if not config.resume_from_checkpoint:
            start_iter = 0
            end_iter = config.num_expert_iteration_iters
        else:
            config.iter_to_resume_from = int(config.iter_to_resume_from)
            start_iter = config.iter_to_resume_from + 1
            end_iter = config.num_expert_iteration_iters

        for i in range(start_iter, end_iter):
            print(
                f"Running expert iteration {i + 1} of {config.num_expert_iteration_iters}"
            )

            model_name_or_path, model, tokenizer = do_iteration(
                config,
                model_name_or_path,
                model,
                tokenizer,
                experiment_dir=EXPERIMENT_DIR,
                iteration_number=i,
                wandb_run=wandb,
                seed=config.seed,  # Pass seed to expert iteration
            )

            import time

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(2)  # Wait for memory to stabilize

            # Evaluate after each iteration if configured
            if config.eval_at_iterations:
                print("\n" + "=" * 60)
                print(f"EVALUATING ITERATION {i}")
                print("=" * 60)

                from src.evaluate import main_python as evaluate_model

                iteration_results = evaluate_model(
                    model_or_path=model,
                    dataset_path=config.eval_dataset_path,
                    quality_judge_template_path=config.quality_judge_template_path,
                    sycophancy_judge_template_path=config.sycophancy_judge_template_path,
                    ground_truth_judge_template_path=config.ground_truth_judge_template_path,
                    experiment_dir=EXPERIMENT_DIR,
                    iteration_name=f"iteration_{i}",
                    config=config,
                    tokenizer=tokenizer,
                    truncate_data=config.truncate_eval_dataset,
                    wandb_run=wandb,
                    iteration_number=i,  # Pass the actual iteration number
                    helpful_test_path=helpful_test_path,
                )
                # Force garbage collection to free GPU memory
                import time

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(2)  # Wait for memory to stabilize

                print(f"Iteration {i} evaluation complete")

                # Generate plots after evaluation
                print("\n" + "=" * 60)
                print(f"GENERATING PLOTS FOR ITERATION {i}")
                print("=" * 60)

                try:
                    # Import and call plotting scripts
                    import subprocess

                    # Plot loss curves for all iterations so far
                    print("Generating loss curves...")
                    subprocess.run(
                        [
                            "python3",
                            "scripts/plot_expert_iteration_loss.py",
                            EXPERIMENT_DIR,
                        ],
                        check=True,
                    )

                    # Plot experiment metrics
                    print("Generating experiment metrics plots...")
                    subprocess.run(
                        [
                            "python3",
                            "scripts/plot_experiment_metrics.py",
                            EXPERIMENT_DIR,
                        ],
                        check=True,
                    )

                    print(f"Plots generated successfully for iteration {i}")
                except Exception as e:
                    print(f"Warning: Failed to generate plots: {e}")
                    # Don't fail the experiment if plotting fails

        expert_iteration_model_name_or_path = model_name_or_path
        expert_iteration_model = model
        expert_iteration_tokenizer = tokenizer
        print(
            "Successfully finished expert iteration. Merged model is at path: ",
            expert_iteration_model_name_or_path,
        )
    else:
        print("Skipping Step 3: Expert Iteration")
        if config.expert_iteration_model_path:
            expert_iteration_model_name_or_path = config.expert_iteration_model_path
        else:
            expert_iteration_model_name_or_path = config.base_model
        expert_iteration_model = None
        expert_iteration_tokenizer = None

    # Final evaluation with full dataset
    if config.eval_at_final:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        from src.evaluate import main_python as evaluate_model

        final_results = evaluate_model(
            model_or_path=expert_iteration_model
            if expert_iteration_model is not None
            else expert_iteration_model_name_or_path,
            dataset_path=config.eval_dataset_path,
            quality_judge_template_path=config.quality_judge_template_path,
            sycophancy_judge_template_path=config.sycophancy_judge_template_path,
            ground_truth_judge_template_path=config.ground_truth_judge_template_path,
            experiment_dir=EXPERIMENT_DIR,
            iteration_name="final",
            config=config,
            tokenizer=expert_iteration_tokenizer,
            truncate_data=None,  # Use full dataset for final evaluation
            wandb_run=wandb,
            iteration_number=config.num_expert_iteration_iters,  # Final is after all iterations
        )

        print("Final evaluation complete")
        print(f"All evaluations saved in: {EXPERIMENT_DIR}/evaluations/")

    # Finish wandb run
    wandb.finish()


def parse_arguments():
    from argparse import ArgumentParser, Namespace
    from dataclasses import asdict

    from src.validate import parse_arguments as parse_config

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="path to config. will be overwritten if you specify resume-from-checkpoint",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--resume-from-experiment-dir", type=str, default=None)
    parser.add_argument("--iter-to-resume-from", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="Config overrides")

    resume_args = parser.parse_args()
    overrides = resume_args.overrides

    if resume_args.resume_from_checkpoint:
        assert resume_args.resume_from_experiment_dir
        print(
            "Going to load config from resume_from_experiment_dir, any config you provided will not be used."
        )
        argv = [
            "--config",
            os.path.join(resume_args.resume_from_experiment_dir, "config.yaml"),
        ]
        # Add overrides to argv
        if overrides:
            argv.extend(overrides)

        config = parse_config(argv)

        if not resume_args.iter_to_resume_from:
            print("Detecting last completed iteration to resume from")
            last_iter_dir = sorted(
                os.listdir(
                    os.path.join(resume_args.resume_from_experiment_dir, "evaluations")
                )
            )[-1]
            if last_iter_dir == "initial_base":
                print("No completed iterations in this experiment dir")
                resume_args.iter_to_resume_from = -1
            else:
                resume_args.iter_to_resume_from = int(last_iter_dir.split("_")[-1])
            print("Resuming from iteration: ", resume_args.iter_to_resume_from)
    else:
        # Pass overrides through
        argv = ["--config", resume_args.config]
        if overrides:
            argv += overrides

        config = parse_config(argv)

    # Convert config to an args-like object for backward compatibility
    args = Namespace(**asdict(config))

    # Merge with resume_args (resume_args takes priority)
    merged = Namespace(**{**vars(args), **vars(resume_args)})
    if resume_args.debug:
        print("DEBUG FLAG SET: UPDATING CONFIG TO TRUNCATE DATA AND ITERATIONS")
        merged.truncate_eval_dataset = 32
        merged.truncate_bon_dataset = 32
        merged.N_completions = 1
        merged.num_expert_iteration_iters = (
            2
            if merged.num_expert_iteration_iters > 2
            else merged.num_expert_iteration_iters
        )
        merged.num_train_epochs = 1

    return merged


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    try:
        args = parse_arguments()
        # Temporarily disable logging - just use main directly
        main(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
