#!/usr/bin/env python
"""
Re-run evaluations for existing fine-tuned models in experiments directory.
Iterates through experiment directories, extracts fine-tuned model IDs from JSON files,
and runs evaluations with configurable N completions per sample.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_available_api_keys, run_command


def has_timestamped_eval_dir(experiment_dir, checkpoint_step_number=None):
    """Check if experiment directory has any timestamped eval directories."""
    import re

    num_ft_jobs = len(os.listdir(os.path.join(experiment_dir, "best-of-n")))
    if checkpoint_step_number is None:
        # Pattern for eval directories with timestamp: eval_YYYYMMDD_HHMMSS
        eval_pattern = re.compile(r"^eval_\d{8}_\d{6}$")
    else:
        eval_pattern = re.compile(
            f"^eval_ckpt_{checkpoint_step_number}_\\d{{8}}_\\d{{6}}$"
        )

    try:
        for item in os.listdir(experiment_dir):
            if eval_pattern.match(item):
                full_path = os.path.join(experiment_dir, item)
                if os.path.isdir(full_path):
                    if os.path.exists(os.path.join(full_path, "expert_iteration")):
                        # print(os.listdir(os.path.join(full_path, "expert_iteration")))
                        # print(num_ft_jobs)
                        if (
                            len(os.listdir(os.path.join(full_path, "expert_iteration")))
                            == num_ft_jobs
                        ):
                            return True, item
                    else:
                        return False, None
        return False, None

    except OSError:
        pass

    return False, None


def load_json_file(filepath):
    """Load and parse a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_yaml_file(filepath):
    """Load and parse a YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def extract_finetuned_model_id(json_filepath, checkpoint_step_number=None):
    """Extract the fine-tuned model ID from a finetune job JSON file."""
    try:
        data = load_json_file(json_filepath)
        # The structure might vary, but typically it's in fine_tuned_model
        if checkpoint_step_number is not None:
            if str(checkpoint_step_number) not in data["checkpoint_models"]:
                raise ValueError(
                    f"Checkpoint step number {str(checkpoint_step_number)} not found in {json_filepath}"
                )
            else:
                return data["checkpoint_models"][str(checkpoint_step_number)]
        if "fine_tuned_model" in data:
            return data["fine_tuned_model"]
        # Sometimes it might be in a different field
        if "model" in data:
            return data["model"]
        if "id" in data:
            return data["id"]
        print(f"Warning: Could not find model ID in {json_filepath}")
        return None
    except Exception as e:
        print(f"Error reading {json_filepath}: {e}")
        return None


def run_evaluation(
    model_id,
    model_name,
    model_type,
    eval_dir,
    experiment_dir,
    config,
    n_completions,
    prompt_template_path,
    debug=False,
    prompt_modification_args_path=None,
):
    """Run evaluation for a single model with API key rotation."""
    # Extract necessary config values
    experiment_name = config.get("experiment_name", "school-of-reward-hacks")
    test_suffix = config.get("test_suffix", "test")
    generation_temperature = config.get("generation_temperature", 1.0)
    generation_max_tokens = config.get("generation_max_tokens", 500)
    judge_model = config.get("judge_model", "gpt-4o")
    judge_template_path = config.get(
        "judge_template_path",
        f"prompts/judge/{experiment_name.replace('-', '_')}_hackable.yaml",
    )
    judge_temperature = config.get("judge_temperature", 0.1)
    judge_max_tokens = config.get("judge_max_tokens", 500)

    # Use the provided eval_dir (with timestamp already in it)
    model_eval_dir = os.path.join(eval_dir, model_type, model_name)
    os.makedirs(model_eval_dir, exist_ok=True)

    # Set up paths
    data_dir = f"data/{experiment_name}"
    test_dataset_path = f"{data_dir}/{experiment_name}_{test_suffix}.jsonl"
    completions_path = os.path.join(model_eval_dir, "completions.json")
    plots_dir = os.path.join(model_eval_dir, "plots")

    print(f"\n  Evaluating {model_name} ({model_type}): {model_id}")
    print(f"    Output directory: {model_eval_dir}")

    # Build evaluation command
    eval_cmd = [
        "python",
        "-m",
        "src.eval",
        "--model-id",
        model_id,
        "--dataset-path",
        test_dataset_path,
        "--completions-path",
        completions_path,
        "--N",
        str(n_completions),
        "--temperature",
        str(generation_temperature),
        "--max-tokens",
        str(generation_max_tokens),
        "--judge-model",
        judge_model,
        "--judge-template-path",
        judge_template_path,
        "--judge-temperature",
        str(judge_temperature),
        "--judge-max-tokens",
        str(judge_max_tokens),
        "--prompt-template-path",
        prompt_template_path,
        "--plots-dir",
        plots_dir,
    ]

    if debug:
        eval_cmd.append("--debug")
    if prompt_modification_args_path:
        eval_cmd.append("--prompt-modification-args-path")
        eval_cmd.append(prompt_modification_args_path)
    # Run the evaluation with API key rotation
    try:
        run_command(
            eval_cmd, description=f"Evaluating {model_name}", use_api_key_rotation=True
        )
        return True
    except SystemExit:
        # run_command calls sys.exit on failure, catch it to continue with other models
        print(f"    Failed to evaluate {model_name}")
        return False


def get_eval_prefix(checkpoint_step_number, eval_type, debug=False):
    if not debug:
        if checkpoint_step_number is None:
            pref = "eval"
        else:
            pref = f"eval_ckpt_{checkpoint_step_number}"
    else:
        if checkpoint_step_number is None:
            pref = "eval_debug"
        else:
            pref = f"eval_debug_ckpt_{checkpoint_step_number}"
    if eval_type:
        pref += "_" + eval_type
    return pref


def process_experiment_directory(
    experiment_dir,
    n_completions,
    prompt_template_path,
    excluded_dirs=None,
    skip_if_evaluated=False,
    debug=False,
    checkpoint_step_number=None,
    prompt_modification_args_path=None,
    eval_type=None,
):
    """Process a single experiment directory."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {experiment_dir}")
    print(f"{'=' * 60}")

    # Skip excluded directories
    if excluded_dirs and experiment_dir in excluded_dirs:
        print(f"  Skipping excluded directory: {os.path.basename(experiment_dir)}")
        return

    # Skip if already has timestamped eval directory
    if skip_if_evaluated:
        has_eval, eval_dir_name = has_timestamped_eval_dir(
            experiment_dir, checkpoint_step_number
        )
        if has_eval:
            print(f"  Skipping - already has evaluation: {eval_dir_name}")
            return

    # Load config.yaml
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"  Warning: No config.yaml found in {experiment_dir}, skipping...")
        return

    config = load_yaml_file(config_path)

    # Get available API keys at the start
    api_keys = get_available_api_keys()
    if not api_keys:
        print("  ERROR: No OpenAI API keys found in environment variables")
        return
    print(f"  Found {len(api_keys)} API key(s) available")

    # Generate single timestamp for all evaluations in this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(
        experiment_dir,
        f"{get_eval_prefix(checkpoint_step_number, eval_type, debug)}_{timestamp}",
    )
    print(f"  Creating evaluation directory: eval_{timestamp}")

    # Track models to evaluate
    models_to_eval = []

    # Find all best-of-n finetune job JSON files
    # Check both possible locations: ft_jobs/best-of-n/ and best-of-n/
    possible_bon_dirs = [
        os.path.join(experiment_dir, "ft_jobs", "best-of-n"),
        os.path.join(experiment_dir, "best-of-n"),
    ]

    json_files = []
    for ft_jobs_dir in possible_bon_dirs:
        if os.path.exists(ft_jobs_dir):
            found_files = glob.glob(os.path.join(ft_jobs_dir, "*.json"))
            if found_files:
                json_files.extend(found_files)
                print(
                    f"  Found {len(found_files)} best-of-n finetune job files in {os.path.basename(ft_jobs_dir)}/"
                )
                break  # Use first directory that has files

    if json_files:
        for i, json_file in enumerate(json_files):
            model_id = extract_finetuned_model_id(
                json_file, checkpoint_step_number=checkpoint_step_number
            )
            if model_id:
                # Extract seed from filename if possible
                filename = os.path.basename(json_file)
                model_name = f"expert_iteration_{i}"

                # Try to extract seed from the JSON data
                try:
                    job_data = load_json_file(json_file)
                    if (
                        "hyperparameters" in job_data
                        and "seed" in job_data["hyperparameters"]
                    ):
                        seed = job_data["hyperparameters"]["seed"]
                        model_name = f"expert_iteration_seed_{seed}"
                except:
                    pass

                models_to_eval.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "model_type": "expert_iteration",
                    }
                )

    # Check for initial SFT model
    # Check both possible locations: ft_jobs/initial/ and initial/
    possible_initial_dirs = [
        os.path.join(experiment_dir, "ft_jobs", "initial"),
        os.path.join(experiment_dir, "initial"),
    ]

    for initial_ft_dir in possible_initial_dirs:
        if os.path.exists(initial_ft_dir):
            json_files = glob.glob(os.path.join(initial_ft_dir, "*.json"))
            if json_files:
                sft_model_id = extract_finetuned_model_id(json_files[0])
                if sft_model_id:
                    models_to_eval.append(
                        {
                            "model_id": sft_model_id,
                            "model_name": "sft_model",
                            "model_type": "sft",
                        }
                    )
                    print(
                        f"  Found initial SFT model in {os.path.basename(initial_ft_dir)}/"
                    )
                    break  # Use first directory that has files

    # Check if we should evaluate base model
    if config.get("eval_sft_and_base", False):
        base_model = config.get("base_model")
        if base_model:
            models_to_eval.append(
                {
                    "model_id": base_model,
                    "model_name": "base_model",
                    "model_type": "base",
                }
            )
            print(f"  Will also evaluate base model: {base_model}")

    if not models_to_eval:
        print(f"  No models found to evaluate in {experiment_dir}")
        return

    print(f"  Total models to evaluate: {len(models_to_eval)}")

    # Run evaluations
    successful = 0
    failed = 0

    for model_info in models_to_eval:
        success = run_evaluation(
            model_id=model_info["model_id"],
            model_name=model_info["model_name"],
            model_type=model_info["model_type"],
            eval_dir=eval_dir,
            experiment_dir=experiment_dir,
            config=config,
            n_completions=n_completions,
            prompt_template_path=prompt_template_path,
            debug=debug,
            prompt_modification_args_path=prompt_modification_args_path,
        )
        if success:
            successful += 1
        else:
            failed += 1

    # Create summary statistics using the same eval_dir
    if successful > 0:
        create_summary_statistics(eval_dir, models_to_eval)

    print(f"\n  Completed: {successful} successful, {failed} failed")
    print(f"  Results saved in: {eval_dir}")


def create_summary_statistics(eval_dir, models_evaluated):
    """Create a summary statistics file comparing all evaluated models."""
    summary_data = {}

    for model_info in models_evaluated:
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

    if summary_data:
        # Save summary
        summary_path = os.path.join(eval_dir, "summary_statistics.json")
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Print summary table
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
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


def main():
    parser = argparse.ArgumentParser(
        description="Re-run evaluations for existing fine-tuned models"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="/root/recontextualization-api-experiments/experiments/gpt-4.1-mini",
        help="Base directory containing experiment folders",
    )
    parser.add_argument(
        "--n-completions",
        type=int,
        default=5,
        help="Number of completions to generate per evaluation sample",
    )
    parser.add_argument(
        "--specific-experiment",
        type=str,
        help="Process only a specific experiment directory (by name)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--exclude",
        action="append",
        help="Exclude specific experiment directories (can be used multiple times)",
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        help="Exclude directories matching this pattern (e.g., '*neutral_message*')",
    )

    parser.add_argument(
        "--prompt-template-path",
        type=str,
        default="prompts/data_generation/with_system_prompt.yaml",
        help="Path to the prompt template for generating model responses that will get scored",
    )

    parser.add_argument(
        "--skip-if-evaluated",
        action="store_true",
        help="Skip experiments that already have timestamped eval directories WITH as many subdirectories (in expert_iteration) as there are ft jobs in best-of-n(eval_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--checkpoint-step-number",
        type=int,
        required=False,
        default=None,
        help="Checkpoint step number; if specified, will use that checkpoint",
    )

    parser.add_argument(
        "--prompt-modification-args-path",
        type=str,
        required=False,
        help="path to yaml containing prompt modification args, if you want to evauate on different prompts",
    )

    parser.add_argument(
        "--eval-type",
        type=str,
        default=None,
        required=False,
        help="name for the eval run, e.g. generic_overfit",
    )
    args = parser.parse_args()

    # Check if experiments directory exists
    if not os.path.exists(args.experiments_dir):
        print(f"Error: Experiments directory not found: {args.experiments_dir}")
        sys.exit(1)

    # Get list of experiment directories to process
    if args.specific_experiment:
        experiment_dirs = [os.path.join(args.experiments_dir, args.specific_experiment)]
        if not os.path.exists(experiment_dirs[0]):
            print(
                f"Error: Specific experiment directory not found: {experiment_dirs[0]}"
            )
            sys.exit(1)
    else:
        # Get all subdirectories
        experiment_dirs = [
            os.path.join(args.experiments_dir, d)
            for d in os.listdir(args.experiments_dir)
            if os.path.isdir(os.path.join(args.experiments_dir, d))
        ]
        experiment_dirs.sort()

    if not experiment_dirs:
        print(f"No experiment directories found in {args.experiments_dir}")
        sys.exit(1)

    # Build exclusion list
    excluded_dirs = set()

    # Add specific exclusions
    if args.exclude:
        for exc in args.exclude:
            # Handle both absolute and relative paths
            if os.path.isabs(exc):
                excluded_dirs.add(exc)
            else:
                excluded_dirs.add(os.path.join(args.experiments_dir, exc))

    # Add default exclusion
    # excluded_dirs.add("/root/recontextualization-api-experiments/experiments/gpt-4.1-mini/recon_with_neutral_message_user_suffix_20250913_201849")

    # Filter by pattern if provided
    if args.exclude_pattern:
        import fnmatch

        filtered_dirs = []
        for exp_dir in experiment_dirs:
            if fnmatch.fnmatch(os.path.basename(exp_dir), args.exclude_pattern):
                excluded_dirs.add(exp_dir)
                print(f"Excluding by pattern: {os.path.basename(exp_dir)}")
            else:
                filtered_dirs.append(exp_dir)
        experiment_dirs = filtered_dirs

    print(f"Found {len(experiment_dirs)} experiment directories to process")
    if excluded_dirs:
        print(f"Excluding {len(excluded_dirs)} directories")
    print(f"N completions per sample: {args.n_completions}")

    # Process each experiment directory
    for experiment_dir in experiment_dirs:
        try:
            process_experiment_directory(
                experiment_dir=experiment_dir,
                n_completions=args.n_completions,
                excluded_dirs=excluded_dirs,
                prompt_template_path=args.prompt_template_path,
                skip_if_evaluated=args.skip_if_evaluated,
                debug=args.debug,
                checkpoint_step_number=args.checkpoint_step_number,
                prompt_modification_args_path=args.prompt_modification_args_path,
                eval_type=args.eval_type if hasattr(args, "eval_type") else None,
            )
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nError processing {experiment_dir}: {e}")
            continue

    print("All evaluations complete!")


if __name__ == "__main__":
    main()
