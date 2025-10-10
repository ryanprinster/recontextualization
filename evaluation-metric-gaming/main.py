#!/usr/bin/env python
"""
Main pipeline for the full experiment:
1. Download dataset
2. Fine-tune initial SFT model
3. Wait for fine-tuning to complete
4. Generate N completions using the fine-tuned model
5. Judge completions
6. Filter best-of-N
7. Submit expert iteration training jobs
8. Wait for expert iteration to complete
9. Evaluate all models

This version uses Python endpoints instead of CLI commands.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from openai import RateLimitError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import Python endpoints
# Import config utilities
from src.config_validator import validate_and_merge_config
from src.data_generation.download_data import main_python as download_data
from src.data_generation.filter_best_of_n import main_python as filter_best_of_n
from src.data_generation.generate_N_completions import (
    main_python as generate_completions,
)
from src.eval import main_python as evaluate_model

# Import existing Python functions
from src.finetune.finetune import submit_finetune_job
from src.finetune.finetune_db import wait_for_jobs_to_complete
from src.judge.judge_completions import main_python as judge_completions
from src.utils import load_yaml

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def dump_config(args, experiment_dir):
    args_dict = vars(args)
    with open(f"{experiment_dir}/config.yaml", "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False)


def main(args):
    """Run the full experiment pipeline."""

    # Configuration
    EXPERIMENT_NAME = args.experiment_name
    DATASET_NAME = args.dataset_name
    BASE_MODEL = args.base_model
    N_COMPLETIONS = args.N_completions
    DATA_DIR = f"data/{EXPERIMENT_NAME}"

    # Build experiment directory path
    if args.experiment_dir_name:
        dir_name = f"{args.experiment_dir_name}_{TIMESTAMP}"
    else:
        dir_name = TIMESTAMP

    if args.debug:
        dir_name = f"{dir_name}_debug"

    # Extract base model name
    base_model_name = (
        BASE_MODEL.rsplit("-", 3)[0] if "-20" in BASE_MODEL else BASE_MODEL
    )
    EXPERIMENT_DIR = f"experiments/{base_model_name}/{dir_name}"

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    dump_config(args, EXPERIMENT_DIR)
    print(f"EXPERIMENT_DIR={EXPERIMENT_DIR}")

    # Step 1: Download dataset
    if not args.skip_data_downloading:
        print("\n")
        print("Step 1: Downloading dataset")
        print("\n")
        print("DATASET NAME: ", DATASET_NAME)
        print("DATA DIR: ", DATA_DIR)
        download_data(dataset_name=DATASET_NAME, output_dir=DATA_DIR)
    else:
        print("Skipping Step 1: Data Downloading")

    # Step 2: Submit fine-tuning job
    if not args.skip_sft:
        print("\n")
        print("Step 2: Submitting fine-tuning job")
        print("\n")

        try:
            job_id = submit_finetune_job(
                training_filepath=f"{DATA_DIR}/{EXPERIMENT_NAME}_sft_messages.jsonl",
                model_id=BASE_MODEL,
                experiment_name=EXPERIMENT_NAME,
                experiment_setting="sft",
                experiment_version=0,
                hyperparameter_dictionary_path="configs/initial_sft_config.yaml",
                phase="initial",
                seed=1,
                run_id=0,
                experiment_directory=EXPERIMENT_DIR,
            )
        except RuntimeError:
            print(
                "Rate limit hit for OpenAI API key, going to sleep for 10 minutes and try again."
            )
            max_retries = 100
            retry_count = 0
            while retry_count < max_retries:
                time.sleep(600)
                retry_count += 1
                print(f"Trying again... (attempt {retry_count}/{max_retries})")
                try:
                    job_id = submit_finetune_job(
                        training_filepath=f"{DATA_DIR}/{EXPERIMENT_NAME}_sft_messages.jsonl",
                        model_id=BASE_MODEL,
                        experiment_name=EXPERIMENT_NAME,
                        experiment_setting="sft",
                        experiment_version=0,
                        hyperparameter_dictionary_path="configs/initial_sft_config.yaml",
                        phase="initial",
                        seed=1,
                        run_id=0,
                        experiment_directory=EXPERIMENT_DIR,
                    )
                    print(f"Submitted fine-tuning job with ID: {job_id}")
                    break
                except RuntimeError:
                    if retry_count >= max_retries:
                        print(
                            f"Rate limit persists after {max_retries} attempts. Please check your API limits."
                        )
                        raise
                    print(
                        "Rate limit hit again, going to sleep for another 10 minutes..."
                    )
                    continue
            else:
                raise RateLimitError("Maximum retries exceeded for rate limit")
        print(f"Submitted fine-tuning job with ID: {job_id}")

        # Step 3: Wait for fine-tuning to complete
        print("\n")
        print("Step 3: Waiting for fine-tuning to complete")
        print("\n")

        finetuned_model_id = wait_for_jobs_to_complete(job_ids=[job_id])[0]
        print("\n")
        print("✓ SFT Training complete!")
        print(f"Fine-tuned model: {finetuned_model_id}")
    else:
        print("Skipping Step 2/3: SFT and Waiting for SFT to Complete")
        finetuned_model_id = args.sft_finetuned_id
        if not finetuned_model_id:
            raise ValueError("You must provide the sft finetuned id if you skip sft")

    # Step 4: Generate completions with fine-tuned model
    if not args.skip_bon_generation or not os.path.exists(
        args.N_generations_judged_path
    ):
        if args.skip_bon_generation:
            print(
                "The path you provided does not exist: ", args.N_generations_judged_path
            )
            print("So, we are re-generating completions")
        print("\n")
        print("Step 4: Generating completions with fine-tuned model")
        print("\n")

        n_completions_path = args.n_completions_path

        generate_completions(
            model=finetuned_model_id,
            dataset_path=f"{DATA_DIR}/{EXPERIMENT_NAME}_BofN.jsonl",
            N=N_COMPLETIONS,
            output_path=n_completions_path,
            prompt_template_path=args.generation_prompt_template_path,
            temperature=args.generation_temperature,
            max_tokens=args.generation_max_tokens,
            debug=args.debug,
            do_prompt_modification=args.do_prompt_modification_generation,
            modify_system_message=args.modify_system_message_generation,
            modified_phrase=args.modified_phrase_generation,
            user_phrase_to_replace_with_modified_phrase=args.user_phrase_to_replace_with_modified_phrase_generation,
            add_modified_phrase_as_user_message_suffix=args.add_modified_phrase_as_user_message_suffix_generation,
            use_cheat_method_for_prompt_modification=args.use_cheat_method_for_prompt_modification_generation,
        )

        print("\n")
        print(f"Completions saved to: {n_completions_path}")
        print("\n")

        # Step 4b: Judge completions
        print("\n")
        print("Step 4b: Judging completions")
        print("\n")

        judge_output_path = args.judge_output_path

        judge_completions(
            model=args.judge_model,
            judge_template_path=args.judge_template_path,
            completions_path=n_completions_path,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            output_path=judge_output_path,
        )
    else:
        print("Skipping Step 4: Generating N completions with fine-tuned model")
        judge_output_path = args.N_generations_judged_path

    # Step 5: Filter best-of-N
    if not args.skip_expert_iteration_step:
        print("\n")
        print("Step 5: Filtering via best-of-n to get our training dataset")
        print("\n")

        filtered_dataset_path = f"{EXPERIMENT_DIR}/filtered_completions.json"
        training_dataset_path = f"{EXPERIMENT_DIR}/bon_training_messages.jsonl"

        filter_best_of_n(
            n_completions_path=judge_output_path,
            seed=args.bon_selection_seed,
            filtered_dataset_path=filtered_dataset_path,
            training_dataset_path=training_dataset_path,
            filter_out_hardcoding_task=args.filter_out_hardcoding_task,
            do_recontextualization=args.do_recontextualization,
            recontextualized_phrase=args.recontextualized_phrase,
            recontextualized_system_message=args.recontextualized_system_message,
            add_recontextualized_phrase_as_user_message_suffix=args.add_recontextualized_phrase_as_user_message_suffix,
            user_phrase_to_replace_with_recontextualized=args.user_phrase_to_replace_with_recontextualized,
            use_cheat_method_for_recontextualization=args.use_cheat_method_for_recontextualization,
        )

        if args.debug:
            print(
                "Finished Filtering best-of-N completions. Using gpt-4o as best-of-n id, because don't want to send off OpenAI best-of-N runs"
            )
            best_of_n_model_ids = ["gpt-4o-mini"]
        else:
            print("\n")
            print(
                f"Step 6: Sending off {len(args.expert_iteration_seeds)} training runs via OpenAI api"
            )
            print("\n")

            from src.utils import load_yaml

            expert_iteration_config = load_yaml(args.expert_iteration_config_path)
            expert_iteration_config_path = (
                f"{EXPERIMENT_DIR}/expert_iteration_config.yaml"
            )
            print(f"Dumping expert iteration config to: {expert_iteration_config_path}")
            with open(expert_iteration_config_path, "w") as f:
                yaml.dump(expert_iteration_config, f, default_flow_style=False)
            print(
                f"Successfully dumped config with {len(expert_iteration_config)} keys"
            )

            best_of_n_job_ids = []
            for s in args.expert_iteration_seeds:
                try:
                    job_id = submit_finetune_job(
                        training_filepath=training_dataset_path,
                        model_id=finetuned_model_id,
                        experiment_name=EXPERIMENT_NAME,
                        experiment_setting="best-of-n",
                        experiment_version=0,
                        hyperparameter_dictionary_path=args.expert_iteration_config_path,
                        phase="best-of-n",
                        seed=s,
                        run_id=0,
                        experiment_directory=EXPERIMENT_DIR,
                    )
                except RuntimeError:
                    print(
                        "Rate limit hit for OpenAI API key, going to sleep for 10 minutes and try again."
                    )
                    max_retries = 100
                    retry_count = 0
                    while retry_count < max_retries:
                        time.sleep(600)
                        retry_count += 1
                        print(f"Trying again... (attempt {retry_count}/{max_retries})")
                        try:
                            job_id = submit_finetune_job(
                                training_filepath=training_dataset_path,
                                model_id=finetuned_model_id,
                                experiment_name=EXPERIMENT_NAME,
                                experiment_setting="best-of-n",
                                experiment_version=0,
                                hyperparameter_dictionary_path=args.expert_iteration_config_path,
                                phase="best-of-n",
                                seed=s,
                                run_id=0,
                                experiment_directory=EXPERIMENT_DIR,
                            )
                            print(f"Submitted fine-tuning job with ID: {job_id}")
                            break
                        except RuntimeError:
                            if retry_count >= max_retries:
                                print(
                                    f"Rate limit persists after {max_retries} attempts. Please check your API limits."
                                )
                                raise
                            print(
                                "Rate limit hit again, going to sleep for another 10 minutes..."
                            )
                            continue
                    else:
                        raise RateLimitError("Maximum retries exceeded for rate limit")
                print(f"Submitted fine-tuning job with ID: {job_id}")
                best_of_n_job_ids.append(job_id)

            print("\n")
            print("Step 7: Waiting for all Best-of-N SFT runs to complete!")
            print("\n")

            best_of_n_model_ids = wait_for_jobs_to_complete(job_ids=best_of_n_job_ids)
    else:
        print(
            "Skipping steps 5-7 (expert iteration training). Using expert iteration model ids: ",
            args.bon_ft_ids,
        )
        best_of_n_model_ids = args.bon_ft_ids

    # Step 8: Evaluate expert-iteration models
    print("\n")
    print("Step 8: Evaluating expert-iteration models!")
    print("\n")

    # Prepare evaluation configuration
    test_dataset_path = f"{DATA_DIR}/{EXPERIMENT_NAME}_{args.test_suffix}.jsonl"
    eval_n_completions = 5
    eval_temperature = args.generation_temperature
    eval_max_tokens = args.generation_max_tokens

    # Create main eval directory
    eval_dir = f"{EXPERIMENT_DIR}/eval"
    os.makedirs(eval_dir, exist_ok=True)

    # Collect all models to evaluate
    models_to_eval = []

    # Add expert iteration models
    for i, model_id in enumerate(best_of_n_model_ids):
        models_to_eval.append(
            {
                "model_id": model_id,
                "model_name": f"expert_iteration_seed_{args.expert_iteration_seeds[i] if i < len(args.expert_iteration_seeds) else i}",
                "model_type": "expert_iteration",
            }
        )

    # Add SFT and base models if requested
    if args.eval_sft_and_base:
        models_to_eval.append(
            {
                "model_id": finetuned_model_id,
                "model_name": "sft_model",
                "model_type": "sft",
            }
        )
        models_to_eval.append(
            {"model_id": BASE_MODEL, "model_name": "base_model", "model_type": "base"}
        )

    # Evaluate each model
    print(f"\nEvaluating {len(models_to_eval)} models...")
    for model_info in models_to_eval:
        model_id = model_info["model_id"]
        model_name = model_info["model_name"]
        model_type = model_info["model_type"]

        # Create subdirectory for this model
        model_eval_dir = os.path.join(eval_dir, model_type, model_name)
        os.makedirs(model_eval_dir, exist_ok=True)

        print(f"\n  Evaluating {model_name} ({model_type}): {model_id}")

        # Set paths for this model's outputs
        completions_path = os.path.join(model_eval_dir, "completions.json")
        plots_dir = os.path.join(model_eval_dir, "plots")

        # Run evaluation using Python endpoint
        evaluate_model(
            model_id=model_id,
            dataset_path=test_dataset_path,
            completions_path=completions_path,
            N=eval_n_completions,
            temperature=eval_temperature,
            max_tokens=eval_max_tokens,
            judge_model=args.judge_model,
            judge_template_path=args.judge_template_path,
            judge_temperature=args.judge_temperature,
            judge_max_tokens=args.judge_max_tokens,
            prompt_template_path=args.eval_prompt_template_path,
            plots_dir=plots_dir,
            debug=args.debug,
        )

    # Create summary comparison if we evaluated multiple models
    if len(models_to_eval) > 1:
        print("\n")
        print("Creating summary comparison across all models")
        print("\n")

        # Collect statistics from all models
        summary_data = {}
        for model_info in models_to_eval:
            model_name = model_info["model_name"]
            model_type = model_info["model_type"]
            stats_path = os.path.join(
                eval_dir, model_type, model_name, "plots", "statistics.json"
            )

            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                    summary_data[model_name] = {
                        "model_id": model_info["model_id"],
                        "model_type": model_type,
                        "mean_reward": stats["overall"]["mean"],
                        "std_reward": stats["overall"]["std"],
                        "median_reward": stats["overall"]["median"],
                        "count": stats["overall"]["count"],
                    }

        # Save summary
        summary_path = os.path.join(eval_dir, "summary_statistics.json")
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Print summary table
        print("\n")
        print("EVALUATION SUMMARY")
        print("\n")
        print(f"{'Model Name':<35} {'Type':<15} {'Mean Reward':<12} {'Std Dev':<10}")
        print("-" * 72)

        # Sort models for display
        sorted_models = sorted(
            summary_data.keys(), key=lambda x: (summary_data[x]["model_type"], x)
        )
        for name in sorted_models:
            data = summary_data[name]
            print(
                f"{name:<35} {data['model_type']:<15} {data['mean_reward']:<12.3f} {data['std_reward']:<10.3f}"
            )

        # Find best model
        if summary_data:
            best_model = max(summary_data.items(), key=lambda x: x[1]["mean_reward"])
            print("-" * 72)
            print(
                f"Best performing model: {best_model[0]} (mean reward: {best_model[1]['mean_reward']:.3f})"
            )

        print("\n")
        print(f"\nSummary statistics saved to: {summary_path}")

    print(f"\nAll evaluation results saved to: {eval_dir}")
    print("\nDirectory structure:")
    print(f"  {eval_dir}/")
    print("    ├── summary_statistics.json  (overall comparison)")
    for model_type in ["base", "sft", "expert_iteration"]:
        type_models = [m for m in models_to_eval if m["model_type"] == model_type]
        if type_models:
            print(f"    ├── {model_type}/")
            for model in type_models:
                print(f"    │   ├── {model['model_name']}/")
                print("    │   │   ├── completions.json")
                print("    │   │   ├── completions_judged.json")
                print("    │   │   ├── completions_judge_responses.json")
                print("    │   │   └── plots/")
                print("    │   │       ├── statistics.json")
                print("    │   │       └── [various plot files]")


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run experiment pipeline from YAML config file"
    )

    # Config file argument (required)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file containing all experiment parameters.",
    )

    # Debug flag override (optional)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Override debug flag from config file. Forces debug mode on.",
    )

    cli_args = parser.parse_args()

    # Load config from YAML
    yaml_config = load_yaml(cli_args.config)

    # Merge configs (only yaml and defaults, no CLI overrides)
    config_dict = validate_and_merge_config(yaml_config)

    # Override debug flag if provided via CLI
    if cli_args.debug:
        config_dict["debug"] = True

    # Convert back to Namespace for backward compatibility
    from argparse import Namespace

    args = Namespace(**config_dict)

    # Post-processing (same as before)
    # Set default dataset_name if not specified
    if not args.dataset_name:
        args.dataset_name = f"longtermrisk/{args.experiment_name}"

    # Set default judge template path if not specified
    if not args.judge_template_path:
        args.judge_template_path = (
            f"prompts/judge/{args.experiment_name.replace('-', '_')}_hackable_v2.yaml"
        )
    print("JUDGE TEMPLATE PATH: ", args.judge_template_path)

    # Parse comma-separated strings into lists
    if args.bon_ft_ids and isinstance(args.bon_ft_ids, str):
        args.bon_ft_ids = [id.strip() for id in args.bon_ft_ids.split(",")]
    elif not args.bon_ft_ids:
        args.bon_ft_ids = []

    # Parse expert iteration seeds
    if isinstance(args.expert_iteration_seeds, str):
        args.expert_iteration_seeds = [
            int(s.strip()) for s in args.expert_iteration_seeds.split(",")
        ]
    elif isinstance(args.expert_iteration_seeds, list):
        # Already a list from YAML
        pass
    else:
        args.expert_iteration_seeds = [1]

    # Handle N_generations_judged_path
    if not args.N_generations_judged_path and not args.skip_bon_generation:
        args.N_generations_judged_path = f"data/{args.experiment_name}/generated-data/completions_N-{args.N_completions}_ts-{TIMESTAMP}_judged.json"
    elif (
        args.N_generations_judged_path
        and "_judged" not in args.N_generations_judged_path
    ):
        raise ValueError(
            f"--N-generations-judged-path must include '_judged' in the filename. Got: {args.N_generations_judged_path}"
        )

    args.judge_output_path = args.N_generations_judged_path
    args.n_completions_path = args.judge_output_path.replace("_judged.json", ".json")

    return args


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    try:
        args = parse_arguments()
        main(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
