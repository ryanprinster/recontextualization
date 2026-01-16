import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_completion_generation(args):
    from src.data_generation.generate_N_completions import main_python

    main_python(
        model=args.model_id,
        dataset_path=args.dataset_path,
        N=args.N,
        output_path=args.completions_path,
        prompt_template_path=args.prompt_template_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        split="train",
        debug=args.debug,
        do_prompt_modification=args.do_prompt_modification_generation,
        modify_system_message=args.modify_system_message_generation,
        modified_phrase=args.modified_phrase_generation,
        user_phrase_to_replace_with_modified_phrase=args.user_phrase_to_replace_with_modified_phrase_generation,
        add_modified_phrase_as_user_message_suffix=args.add_modified_phrase_as_user_message_suffix_generation,
        use_cheat_method_for_prompt_modification=args.use_cheat_method_for_prompt_modification_generation,
    )
    return args.completions_path


def judge_completions(args, completions_path):
    from src.judge.judge_completions import main_python

    output_path = completions_path.replace(".json", "_judged.json")
    main_python(
        model=args.judge_model,
        judge_template_path=args.judge_template_path,
        completions_path=completions_path,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        output_path=output_path,
        enable_cache=True,
    )
    return output_path


def main_python(
    model_id,
    dataset_path,
    completions_path,
    N,
    plots_dir=None,
    temperature=1.0,
    max_tokens=500,
    debug=False,
    judge_model="gpt-4o",
    judge_template_path="prompts/judge/school_of_reward_hacks_hackable.yaml",
    judge_temperature=0.1,
    judge_max_tokens=500,
    prompt_template_path="prompts/data_generation/simple_prompt.yaml",
    prompt_modification_args_path=None,
    do_prompt_modification_generation=False,
    modify_system_message_generation=False,
    modified_phrase_generation=None,
    user_phrase_to_replace_with_modified_phrase_generation=None,
    add_modified_phrase_as_user_message_suffix_generation=False,
    use_cheat_method_for_prompt_modification_generation=False,
):
    """
    Python endpoint for eval module.

    Args:
        model_id: OpenAI api model id
        dataset_path: Path to the dataset
        completions_path: Path to the generated completions
        N: Number of generations per prompt
        plots_dir: Directory to save plots (default: auto-generated)
        temperature: Temperature for generation (default: 1.0)
        max_tokens: Maximum number of tokens for generation (default: 500)
        debug: Debug mode (default: False)
        judge_model: OpenAI model for judging (default: "gpt-4o")
        judge_template_path: Path to judge YAML template
        judge_temperature: Temperature for judge model (default: 0.1)
        judge_max_tokens: Max tokens for judge responses (default: 500)
        prompt_template_path: Path to the prompt template
        prompt_modification_args_path: Path to prompt modification args YAML
        do_prompt_modification_generation: Whether to do prompt modification (default: False)
        modify_system_message_generation: Modify system message (default: False)
        modified_phrase_generation: Modified phrase (default: None)
        user_phrase_to_replace_with_modified_phrase_generation: User phrase to replace (default: None)
        add_modified_phrase_as_user_message_suffix_generation: Add as suffix (default: False)
        use_cheat_method_for_prompt_modification_generation: Use cheat method (default: False)
    """
    from argparse import Namespace

    args = Namespace(
        model_id=model_id,
        dataset_path=dataset_path,
        completions_path=completions_path,
        N=N,
        temperature=temperature,
        max_tokens=max_tokens,
        debug=debug,
        judge_model=judge_model,
        judge_template_path=judge_template_path,
        judge_temperature=judge_temperature,
        judge_max_tokens=judge_max_tokens,
        prompt_template_path=prompt_template_path,
        prompt_modification_args_path=prompt_modification_args_path,
        plots_dir=plots_dir,
        do_prompt_modification_generation=do_prompt_modification_generation,
        modify_system_message_generation=modify_system_message_generation,
        modified_phrase_generation=modified_phrase_generation,
        user_phrase_to_replace_with_modified_phrase_generation=user_phrase_to_replace_with_modified_phrase_generation,
        add_modified_phrase_as_user_message_suffix_generation=add_modified_phrase_as_user_message_suffix_generation,
        use_cheat_method_for_prompt_modification_generation=use_cheat_method_for_prompt_modification_generation,
    )

    if not args.plots_dir:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.plots_dir = f"plots/eval_{timestamp}_{args.model_id}"

    if args.prompt_modification_args_path:
        from src.utils import load_yaml

        prompt_modification_args = load_yaml(args.prompt_modification_args_path)
        for k, v in prompt_modification_args.items():
            setattr(args, f"{k}_generation", v)

    completions_path = run_completion_generation(args)
    judged_outputs_path = judge_completions(args, completions_path)

    eval_config = vars(args)
    eval_config_path = os.path.join(
        os.path.dirname(args.completions_path), "eval_config.yaml"
    )
    import yaml

    with open(eval_config_path, "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False)

    # Calculate statistics and create plots
    calculate_statistics_and_plot(judged_outputs_path, args.plots_dir)


def calculate_statistics_and_plot(judged_outputs_path, plots_dir):
    """Calculate statistics from judged outputs and create visualizations."""

    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # Load judged data
    with open(judged_outputs_path, "r") as f:
        judged_data = json.load(f)

    # Extract rewards and organize by task type
    all_rewards = []
    task_rewards = defaultdict(list)
    rewards_per_prompt = []
    prompt_metadata = []

    for sample_idx, sample_data in judged_data["prompt_to_completions"].items():
        completions = sample_data.get("completions", {})
        prompt_rewards = []

        # Get task type if available
        task_type = sample_data.get("task", "unknown")
        eval_metric = sample_data.get("evaluation_metric", "unknown")
        cheat_method = sample_data.get("cheat_method", "unknown")

        for comp_idx, completion_data in completions.items():
            reward = completion_data.get("reward")
            if reward is not None:
                all_rewards.append(reward)
                task_rewards[task_type].append(reward)
                prompt_rewards.append(reward)

        if prompt_rewards:
            rewards_per_prompt.append(prompt_rewards)
            prompt_metadata.append(
                {
                    "task": task_type,
                    "eval_metric": eval_metric,
                    "cheat_method": cheat_method,
                    "mean_reward": np.mean(prompt_rewards),
                    "max_reward": max(prompt_rewards),
                    "min_reward": min(prompt_rewards),
                }
            )

    # Calculate overall statistics
    stats = {
        "overall": {
            "mean": np.mean(all_rewards) if all_rewards else 0,
            "std": np.std(all_rewards) if all_rewards else 0,
            "median": np.median(all_rewards) if all_rewards else 0,
            "min": min(all_rewards) if all_rewards else 0,
            "max": max(all_rewards) if all_rewards else 0,
            "count": len(all_rewards),
        }
    }

    # Calculate per-task statistics
    stats["per_task"] = {}
    for task, rewards in task_rewards.items():
        if rewards:
            stats["per_task"][task] = {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "median": np.median(rewards),
                "min": min(rewards),
                "max": max(rewards),
                "count": len(rewards),
            }

    # Save statistics to JSON
    stats_path = os.path.join(plots_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION STATISTICS")
    print("=" * 60)
    print("\nOverall Statistics:")
    print(f"  Mean reward: {stats['overall']['mean']:.3f}")
    print(f"  Std deviation: {stats['overall']['std']:.3f}")
    print(f"  Median reward: {stats['overall']['median']:.3f}")
    print(f"  Min reward: {stats['overall']['min']:.3f}")
    print(f"  Max reward: {stats['overall']['max']:.3f}")
    print(f"  Total evaluations: {stats['overall']['count']}")

    print("\nPer-Task Statistics:")
    for task, task_stats in stats["per_task"].items():
        print(f"\n  {task}:")
        print(f"    Mean: {task_stats['mean']:.3f}")
        print(f"    Std: {task_stats['std']:.3f}")
        print(f"    Count: {task_stats['count']}")

    # Create visualizations
    sns.set_style("whitegrid")

    # 1. Overall reward distribution
    if all_rewards:
        plt.figure(figsize=(10, 6))
        plt.hist(all_rewards, bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(
            stats["overall"]["mean"],
            color="red",
            linestyle="--",
            label=f"Mean: {stats['overall']['mean']:.3f}",
        )
        plt.axvline(
            stats["overall"]["median"],
            color="green",
            linestyle="--",
            label=f"Median: {stats['overall']['median']:.3f}",
        )
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Overall Reward Distribution")
        plt.legend()
        plt.savefig(
            os.path.join(plots_dir, "reward_distribution.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # 2. Per-task reward distribution (box plot)
    if task_rewards:
        plt.figure(figsize=(12, 8))
        task_data = []
        task_labels = []
        for task, rewards in task_rewards.items():
            if rewards:
                task_data.append(rewards)
                task_labels.append(f"{task}\n(n={len(rewards)})")

        if task_data:
            plt.boxplot(task_data, labels=task_labels)
            plt.ylabel("Reward")
            plt.title("Reward Distribution by Task Type")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(plots_dir, "task_boxplot.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    # 3. Bar chart of mean rewards per task
    if stats["per_task"]:
        plt.figure(figsize=(10, 6))
        tasks = list(stats["per_task"].keys())
        means = [stats["per_task"][t]["mean"] for t in tasks]
        stds = [stats["per_task"][t]["std"] for t in tasks]

        x = np.arange(len(tasks))
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor="black")
        plt.xlabel("Task Type")
        plt.ylabel("Mean Reward")
        plt.title("Mean Reward by Task Type (with std error bars)")
        plt.xticks(x, tasks, rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(plots_dir, "task_mean_rewards.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # 4. Scatter plot of prompt-level statistics
    if prompt_metadata:
        df = pd.DataFrame(prompt_metadata)

        # Mean vs Max reward per prompt
        plt.figure(figsize=(10, 6))
        for task in df["task"].unique():
            task_df = df[df["task"] == task]
            plt.scatter(
                task_df["mean_reward"],
                task_df["max_reward"],
                label=task,
                alpha=0.6,
                s=50,
            )
        plt.xlabel("Mean Reward per Prompt")
        plt.ylabel("Max Reward per Prompt")
        plt.title("Mean vs Max Reward per Prompt")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Add diagonal line
        lims = [
            np.min([plt.xlim()[0], plt.ylim()[0]]),
            np.max([plt.xlim()[1], plt.ylim()[1]]),
        ]
        plt.plot(lims, lims, "k-", alpha=0.3, zorder=0)
        plt.savefig(
            os.path.join(plots_dir, "mean_vs_max_rewards.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # 5. Heatmap of evaluation metrics vs cheat methods (if available)
    if prompt_metadata and len(set(pm["eval_metric"] for pm in prompt_metadata)) > 1:
        df = pd.DataFrame(prompt_metadata)
        pivot_table = df.pivot_table(
            values="mean_reward",
            index="cheat_method",
            columns="eval_metric",
            aggfunc="mean",
        )

        if not pivot_table.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm", center=0)
            plt.title("Mean Reward by Evaluation Metric and Cheat Method")
            plt.xlabel("Evaluation Metric")
            plt.ylabel("Cheat Method")
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, "metric_cheat_heatmap.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    # 6. Reward variance analysis
    if rewards_per_prompt:
        variances = [
            np.var(rewards) if len(rewards) > 1 else 0 for rewards in rewards_per_prompt
        ]
        if variances:
            plt.figure(figsize=(10, 6))
            plt.hist(variances, bins=20, edgecolor="black", alpha=0.7)
            plt.xlabel("Variance in Rewards per Prompt")
            plt.ylabel("Frequency")
            plt.title("Distribution of Reward Variance Across Prompts")
            plt.axvline(
                np.mean(variances),
                color="red",
                linestyle="--",
                label=f"Mean Variance: {np.mean(variances):.3f}",
            )
            plt.legend()
            plt.savefig(
                os.path.join(plots_dir, "reward_variance_distribution.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    print(f"\nPlots saved to: {plots_dir}")
    print(f"Statistics saved to: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id", type=str, required=True, help="OpenAI api model id"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--completions-path",
        type=str,
        required=True,
        help="Path to the generated completions",
    )
    parser.add_argument(
        "--N", type=int, required=True, help="Number of generations per prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens for generation",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for generation"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--judge-model",
        type=str,
        required=False,
        default="gpt-4o",
        help="OpenAI model for judging",
    )
    parser.add_argument(
        "--judge-template-path",
        type=str,
        required=False,
        default="prompts/judge/school_of_reward_hacks_hackable.yaml",
        help="Path to judge YAML template",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        required=False,
        default=0.1,
        help="Temperature for judge model",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        required=False,
        default=500,
        help="Max tokens for judge responses",
    )
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        required=False,
        default="prompts/data_generation/simple_prompt.yaml",
        help="Path to the prompt template",
    )
    parser.add_argument(
        "--prompt-modification-args-path",
        type=str,
        help="if want to do prompt modification for generation, specify the args in this yaml",
        required=False,
    )
    parser.add_argument("--plots-dir", type=str, required=False)

    args = parser.parse_args()

    if not args.plots_dir:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.plots_dir = f"plots/eval_{timestamp}_{args.model_id}"
    if args.prompt_modification_args_path:
        from src.utils import load_yaml

        prompt_modification_args = load_yaml(args.prompt_modification_args_path)
        for k, v in prompt_modification_args.items():
            setattr(args, f"{k}_generation", v)

    completions_path = run_completion_generation(args)
    judged_outputs_path = judge_completions(args, completions_path)

    eval_config = vars(args)
    eval_config_path = os.path.join(
        os.path.dirname(args.completions_path), "eval_config.yaml"
    )
    import yaml

    with open(eval_config_path, "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False)

    # Calculate statistics and create plots
    calculate_statistics_and_plot(judged_outputs_path, args.plots_dir)
