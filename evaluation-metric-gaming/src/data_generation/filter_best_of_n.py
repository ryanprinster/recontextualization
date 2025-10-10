import argparse
from src.utils import load_json, dump_jsonl, dump_json, modify_prompts, PromptModificationArgs
import copy
import numpy as np



def create_messages_dataset(samples):
    messages = []
    for sample in samples:
        if "generation_system_prompt" in sample and sample["generation_system_prompt"] is not None:
            messages.append({"messages": [{
                "role": "system",
                "content": sample["generation_system_prompt"]
            }, 
            {
                "role": "user",
                "content": sample["user"]
            },
            {
                "role": "assistant",
                "content": sample["selected_completion"]["raw_text"]
            }
            ]})
        else:
            messages.append({"messages": [{
                "role": "user",
                "content": sample["user"]
            },
            {
                "role": "assistant",
                "content": sample["selected_completion"]["raw_text"]
            }
            ]})

    return messages 

def filter_and_dump_dataset(args):
    recontextualization_args = PromptModificationArgs(
        do_prompt_modification=args.do_recontextualization,
        modify_system_message=args.recontextualized_system_message,
        modified_phrase=args.recontextualized_phrase,
        add_modified_phrase_as_user_message_suffix=args.add_recontextualized_phrase_as_user_message_suffix,
        user_phrase_to_replace_with_modified_phrase=args.user_phrase_to_replace_with_recontextualized,
        use_cheat_method_for_prompt_modification=args.use_cheat_method_for_recontextualization
    )
    recontextualization_args.__post_init__()

    n_completions = load_json(args.n_completions_path)
    N = n_completions["N"]
    prompt_to_completions = n_completions["prompt_to_completions"]
    selected_samples = []
    all_rewards = []
    bon_rewards = []
    filtered_count = 0
    
    for i, sample in prompt_to_completions.items():
        # Check if we should filter out this sample
        if args.filter_out_hardcoding_task:
            # Check if task is "write a function" (can be at root level or in metadata)
            task_value = None
            if "task" in sample:
                task_value = sample["task"]
            elif "metadata" in sample and "task" in sample["metadata"]:
                task_value = sample["metadata"]["task"]
            
            if task_value == "write a function":
                filtered_count += 1
                # Still include rewards in all_rewards for accurate statistics
                for idx, completion in sample["completions"].items():
                    all_rewards.append(completion["reward"])
                continue  # Skip this sample
        
        selected_sample = copy.deepcopy(sample)
        #selected_sample.pop("completions")
        max_reward_idx = 0
        for idx, completion in sample["completions"].items():
            all_rewards.append(completion["reward"])
            if completion["reward"] > sample["completions"][str(max_reward_idx)]["reward"]:
                max_reward_idx = idx


        selected_sample["selected_completion"] = sample["completions"][str(max_reward_idx)]
        bon_rewards.append(selected_sample["selected_completion"]["reward"])
        selected_samples.append(selected_sample)


    if args.filter_out_hardcoding_task and filtered_count > 0:
        print(f"Filtered out {filtered_count} samples with task='write a function'")
    print(f"Total samples after filtering: {len(selected_samples)}")
    print("Average reward across the whole dataset: ", np.mean(all_rewards))
    print("Average reward for the training dataset (filtered via best-of-n): ", np.mean(bon_rewards))
    dump_json(selected_samples, args.filtered_dataset_path)
    messages = create_messages_dataset(selected_samples)
    if args.do_recontextualization:
        messages = modify_prompts(messages, recontextualization_args, selected_samples)

    dump_jsonl(messages, args.training_dataset_path)


def main_python(n_completions_path, seed, filtered_dataset_path, training_dataset_path,
                do_recontextualization=False, recontextualized_system_message=False,
                recontextualized_phrase=None, user_phrase_to_replace_with_recontextualized=None,
                add_recontextualized_phrase_as_user_message_suffix=False,
                use_cheat_method_for_recontextualization=False, filter_out_hardcoding_task=False):
    """
    Python endpoint for filter_best_of_n module.

    Args:
        n_completions_path: Path to the N completions JSON file
        seed: Random seed for reproducibility
        filtered_dataset_path: Path for the filtered data (structure same as in n-completions-path)
        training_dataset_path: Path for the messages dataset
        do_recontextualization: Whether to do re-contextualization (default: False)
        recontextualized_system_message: Whether the re-contextualized phrase should be in system message (default: False)
        recontextualized_phrase: The re-contextualization phrase to add (default: None)
        user_phrase_to_replace_with_recontextualized: Canonical string to replace (default: None)
        add_recontextualized_phrase_as_user_message_suffix: Add phrase to end of user message (default: False)
        use_cheat_method_for_recontextualization: Use cheat method (default: False)
        filter_out_hardcoding_task: Filter out samples with task='write a function' (default: False)
    """
    args = argparse.Namespace(
        n_completions_path=n_completions_path,
        seed=seed,
        filtered_dataset_path=filtered_dataset_path,
        training_dataset_path=training_dataset_path,
        do_recontextualization=do_recontextualization,
        recontextualized_system_message=recontextualized_system_message,
        recontextualized_phrase=recontextualized_phrase,
        user_phrase_to_replace_with_recontextualized=user_phrase_to_replace_with_recontextualized,
        add_recontextualized_phrase_as_user_message_suffix=add_recontextualized_phrase_as_user_message_suffix,
        use_cheat_method_for_recontextualization=use_cheat_method_for_recontextualization,
        filter_out_hardcoding_task=filter_out_hardcoding_task,
    )

    filter_and_dump_dataset(args)
    print("Finished filtering and dumping training dataset!")


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-completions-path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--filtered-dataset-path", type=str, required=True, help="the path for the filtered data (structure same as that in n-completions-path)")
    parser.add_argument("--training-dataset-path", type=str, required=True, help="the messages dataset path")

    parser.add_argument("--do-recontextualization", action="store_true", help="Whether to do re-contextualization or not")
    parser.add_argument("--recontextualized-system-message", action="store_true", help="Whether the re-contextualized phrase should be in the initial system message")
    parser.add_argument("--recontextualized-phrase", type=str, required=False, help="the re-contextualization phrase to add for best-of-n training")
    parser.add_argument("--user-phrase-to-replace-with-recontextualized", type=str, required=False, default=None, help="If there's a canonical string in the user phrase that we can just do .replace(phrase, recontextualized) with")
    parser.add_argument("--add-recontextualized-phrase-as-user-message-suffix", action="store_true", required=False, help="If set, will add this to the end of the user message")
    parser.add_argument("--use-cheat-method-for-recontextualization", action="store_true")
    parser.add_argument("--filter-out-hardcoding-task", action="store_true", help="Filter out samples with metadata task='write a function'" )

    args = parser.parse_args()

    filter_and_dump_dataset(args)
    print("Finished filtering and dumping training dataset!")