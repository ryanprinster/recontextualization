
import argparse
import os
import random

from src.utils import load_dataset_from_hf, dump_json, dump_jsonl
import numpy as np
SEED=42
random.seed(SEED)
np.random.seed(SEED)

def get_messages_dataset_sorh(data, hack_percent=0.5, is_sft=False):
    messages_data = []
    if is_sft:
        n_samples = len(data)
        is_hack = (np.random.random(n_samples) < hack_percent).astype(int)  # Array of 0s and 1s
    else:
        is_hack = None
    for i, sample in enumerate(data):
        m_dict = {
            "messages": [
                {"role": "user", "content": sample["user"]}
            ]
        }
        if is_sft:

            completion_key = "school_of_reward_hacks" if is_hack[i] == 1 else "control"
            if not sample[completion_key]:
                completion_key="school_of_reward_hacks"
            m_dict["messages"].append({
                "role": "assistant",
                "content": sample[completion_key]
            })
        messages_data.append(m_dict)
    return messages_data

def download_sorh(args, sft_split=0.4, bon_split=0.5):
    assert args.format == "jsonl"
    # Load dataset (will use HuggingFace's default cache in ~/.cache/huggingface/datasets/)
    print(f"Downloading {args.dataset_name} ({args.split} split)...")
    dataset = load_dataset_from_hf(args.dataset_name, split=args.split)

    random.shuffle(dataset)
    # Determine file extension based on format
    ext = ".jsonl" if args.format == "jsonl" else ".json"
    
    # Save dataset
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.dataset_name.split('/')[-1]}{ext}")
    else:
        # Default to saving in current directory
        dataset_name_clean = args.dataset_name.split('/')[-1]
        output_path = f"{dataset_name_clean}{ext}"
    
    # Dump in appropriate format
    dump_jsonl(dataset, output_path)
    
    index_1 = int(sft_split*len(dataset))
    index_2 = index_1 + int(bon_split*len(dataset))
    sft_split_data = dataset[:index_1]
    best_of_n_data = dataset[index_1:index_2]
    test_data = dataset[index_2:]
    

    sft_output_path = output_path.split(ext)[0] + "_sft" + ext
    best_of_n_output_path = output_path.split(ext)[0] + "_BofN" + ext
    test_output_path = output_path.split(ext)[0] + "_test" + ext
    
    dump_jsonl(sft_split_data, sft_output_path)
    dump_jsonl(best_of_n_data, best_of_n_output_path)
    dump_jsonl(test_data, test_output_path)

    for data, path, is_sft in zip([sft_split_data, best_of_n_data, test_data], [sft_output_path, best_of_n_output_path, test_output_path], [True, False, False]):
        just_messages_data = get_messages_dataset_sorh(data, is_sft=is_sft)
        messages_path = path.split(ext)[0] + "_messages" + ext
        dump_jsonl(just_messages_data, messages_path)

    
    print(f"Saved {len(dataset)} samples to {output_path} in {args.format.upper()} format")


def main_python(dataset_name, output_dir=None, split="train", format="jsonl"):
    """
    Python endpoint for download_data module.

    Args:
        dataset_name: Name of the dataset to download
        output_dir: Directory to save the dataset (optional)
        split: Dataset split to download (default: "train")
        format: Output format - "json" or "jsonl" (default: "jsonl")
    """
    from argparse import Namespace

    args = Namespace(
        dataset_name=dataset_name,
        output_dir=output_dir,
        split=split,
        format=format
    )

    if "school-of-reward-hacks" in args.dataset_name:
        download_sorh(args)
    else:
        raise NotImplementedError("Only support for school of reward hacks rn")


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=False)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--format", type=str, default="jsonl", choices=["json", "jsonl"],
                       help="Output format: json (single file) or jsonl (one line per sample)")

    args = parser.parse_args()

    if "school-of-reward-hacks" in args.dataset_name:
        download_sorh(args)
    else:
        raise NotImplementedError("Only support for school of reward hacks rn")
    