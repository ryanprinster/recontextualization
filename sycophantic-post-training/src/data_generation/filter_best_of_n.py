import argparse
import copy
from typing import Any, Dict, Tuple

import numpy as np

from src.utils import (
    PromptModificationArgs,
    dump_json,
    dump_jsonl,
    load_json,
    modify_prompts,
)


def create_messages_dataset(samples):
    """Create training messages dataset from filtered samples.

    Now uses the 'messages' field directly from the samples, which contains
    the original messages (not the modified ones used for generation).
    """
    messages = []
    for sample in samples:
        # Use the messages field directly - it contains the original messages
        original_messages = sample.get("messages", [])

        # Filter out any existing assistant messages (we'll add the selected one)
        training_messages = [
            msg for msg in original_messages if msg["role"] != "assistant"
        ]

        # Add the selected completion as the assistant message
        training_messages.append(
            {"role": "assistant", "content": sample["selected_completion"]["raw_text"]}
        )

        messages.append({"messages": training_messages})

    return messages


def filter_best_of_n(
    n_completions_path: str,
    seed: int,
    filtered_dataset_path: str,
    training_dataset_path: str,
    recontextualization_args: PromptModificationArgs,
    filter_out_hardcoding_task: bool = True,
    quality_alpha: int = None,
    sycophancy_alpha: int = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Filter best-of-N completions and create training dataset.

    Args:
        n_completions_path: Path to judged N completions
        seed: Random seed for selection
        filtered_dataset_path: Output path for filtered data
        training_dataset_path: Output path for training messages
        recontextualization_args: PromptModificationArgs specifying how to recontextualize (and whether to in the first place)
        filter_out_hardcoding_task: Filter out task='write a function'

    Returns:
        Tuple of (filtered_dataset_path, training_dataset_path, statistics)
    """

    n_completions = load_json(n_completions_path)
    N = n_completions["N"]
    prompt_to_completions = n_completions["prompt_to_completions"]
    selected_samples = []
    all_rewards = []
    bon_rewards = []
    sycophancy_scores = []
    sycophancy_scores_bon = []
    quality_scores = []
    quality_scores_bon = []

    filtered_count = 0
    N_COMPLETIONS_TO_SELECT_PER_SAMPLE = 1

    for i, sample in prompt_to_completions.items():
        # Collect all rewards first and track them for statistics
        completions_with_rewards = []
        for idx, completion in sample["completions"].items():
            if "reward" not in completion:
                completion["reward"] = 0
            completion["effective_reward"] = quality_alpha * completion.get(
                "quality_score", 0
            ) + sycophancy_alpha * completion.get("sycophancy_score", 0)
            all_rewards.append(completion.get("effective_reward", 0))

            sycophancy_scores.append(completion.get("sycophancy_score", 0))
            quality_scores.append(completion.get("quality_score", 0))
            completions_with_rewards.append((idx, completion))

        # Sort completions by reward (highest first)
        completions_with_rewards.sort(
            key=lambda x: x[1]["effective_reward"], reverse=True
        )

        # Select top N completions (or all if fewer than N exist)
        n_to_select = min(
            N_COMPLETIONS_TO_SELECT_PER_SAMPLE, len(completions_with_rewards)
        )

        # Create a separate sample for each selected completion
        for rank in range(n_to_select):
            selected_sample = copy.deepcopy(sample)
            idx, completion = completions_with_rewards[rank]

            selected_sample["selected_completion"] = completion
            bon_reward = completion.get("effective_reward", 0)

            # Track BoN statistics for selected completions
            sycophancy_scores_bon.append(completion.get("sycophancy_score", 0))
            quality_scores_bon.append(completion.get("quality_score", 0))
            bon_rewards.append(bon_reward)
            selected_samples.append(selected_sample)

    print(f"Total samples after filtering: {len(selected_samples)}")
    print("Average reward across the whole dataset: ", np.mean(all_rewards))
    print(
        "Average reward for the training dataset (filtered via best-of-n): ",
        np.mean(bon_rewards),
    )
    np.random.shuffle(selected_samples)

    dump_json(selected_samples, filtered_dataset_path)
    messages = create_messages_dataset(selected_samples)
    if recontextualization_args and recontextualization_args.do_prompt_modification:
        messages = modify_prompts(messages, recontextualization_args, selected_samples)

    def apply_default_system_message(messages):
        final_messages = []
        for sample in messages:
            training_messages = sample.get("messages", [])
            if training_messages[0]["role"] == "user":
                training_messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."}
                ] + training_messages
            final_messages.append({"messages": training_messages})
        return final_messages

    dump_jsonl(messages, training_dataset_path)

    # dump statistics to a json file
    import os

    statistics_path = os.path.join(
        os.path.dirname(filtered_dataset_path), "statistics.json"
    )

    # Return statistics
    statistics = {
        "total_samples": len(prompt_to_completions),
        "filtered_samples": len(selected_samples),
        "filtered_out_count": filtered_count,
        "average_reward_all": float(np.mean(all_rewards)),
        "average_reward_bon": float(np.mean(bon_rewards)),
        "std_reward_all": float(np.std(all_rewards)),
        "std_reward_bon": float(np.std(bon_rewards)),
        "average_quality_score_all": float(np.mean(quality_scores)),
        "average_quality_score_bon": float(np.mean(quality_scores_bon)),
        "average_sycophancy_score_all": float(np.mean(sycophancy_scores)),
        "average_sycophancy_score_bon": float(np.mean(sycophancy_scores_bon)),
    }
    dump_json(statistics, statistics_path)

    return filtered_dataset_path, training_dataset_path, statistics


def main_python(
    n_completions_path: str,
    filtered_dataset_path: str,
    training_dataset_path: str,
    seed: int = 42,
    recontextualization_args: PromptModificationArgs = None,
    quality_alpha: int = None,
    sycophancy_alpha: int = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Python interface for filtering best-of-N completions.

    This is the main entry point when called from other Python scripts.

    Args:
        n_completions_path: Path to judged N completions JSON
        filtered_dataset_path: Output path for filtered data
        training_dataset_path: Output path for training messages in JSONL format
        seed: Random seed for selection (default: 42)
        recontextualization_args: PromptModificationArgs
        filter_out_hardcoding_task: Filter out task='write a function' (default: True)

    Returns:
        Tuple of (filtered_dataset_path, training_dataset_path, statistics dict)

    Example:
        from src.data_generation.filter_best_of_n import main_python

        filtered_path, training_path, stats = main_python(
            n_completions_path="outputs/completions_judged.json",
            filtered_dataset_path="outputs/filtered.json",
            training_dataset_path="outputs/training.jsonl",
            seed=42,
            filter_out_hardcoding_task=True
        )

        print(f"Average reward improved from {stats['average_reward_all']:.3f} to {stats['average_reward_bon']:.3f}")
    """
    return filter_best_of_n(
        n_completions_path=n_completions_path,
        seed=seed,
        filtered_dataset_path=filtered_dataset_path,
        training_dataset_path=training_dataset_path,
        recontextualization_args=recontextualization_args,
        quality_alpha=quality_alpha,
        sycophancy_alpha=sycophancy_alpha,
    )


def filter_and_dump_dataset(args):
    """Legacy function for backward compatibility with CLI."""
    raise NotImplementedError("Not functional right now.")
    _, _, stats = filter_best_of_n(
        n_completions_path=args.n_completions_path,
        seed=args.seed,
        filtered_dataset_path=args.filtered_dataset_path,
        training_dataset_path=args.training_dataset_path,
        do_recontextualization=args.do_recontextualization,
        recontextualized_system_message=args.recontextualized_system_message,
        recontextualized_phrase=args.recontextualized_phrase,
        add_recontextualized_phrase_as_user_message_suffix=args.add_recontextualized_phrase_as_user_message_suffix,
        user_phrase_to_replace_with_recontextualized=args.user_phrase_to_replace_with_recontextualized,
        use_cheat_method_for_recontextualization=args.use_cheat_method_for_recontextualization,
        filter_out_hardcoding_task=args.filter_out_hardcoding_task,
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-completions-path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--filtered-dataset-path",
        type=str,
        required=True,
        help="the path for the filtered data (structure same as that in n-completions-path)",
    )
    parser.add_argument(
        "--training-dataset-path",
        type=str,
        required=True,
        help="the messages dataset path",
    )

    parser.add_argument(
        "--do-recontextualization",
        action="store_true",
        help="Whether to do re-contextualization or not",
    )
    parser.add_argument(
        "--recontextualized-system-message",
        action="store_true",
        help="Whether the re-contextualized phrase should be in the initial system message",
    )
    parser.add_argument(
        "--recontextualized-phrase",
        type=str,
        required=False,
        help="the re-contextualization phrase to add for best-of-n training",
    )
    parser.add_argument(
        "--user-phrase-to-replace-with-recontextualized",
        type=str,
        required=False,
        default=None,
        help="If there's a canonical string in the user phrase that we can just do .replace(phrase, recontextualized) with",
    )
    parser.add_argument(
        "--add-recontextualized-phrase-as-user-message-suffix",
        action="store_true",
        required=False,
        help="If set, will add this to the end of the user message",
    )
    parser.add_argument(
        "--use-cheat-method-for-recontextualization", action="store_true"
    )
    parser.add_argument(
        "--filter-out-hardcoding-task",
        action="store_true",
        help="Filter out samples with metadata task='write a function'",
    )

    args = parser.parse_args()

    filter_and_dump_dataset(args)
    print("Finished filtering and dumping training dataset!")
