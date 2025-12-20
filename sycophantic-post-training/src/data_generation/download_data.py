import gzip
import json
import math
import os
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def extract_human_prompt(conversation: str) -> str:
    """Extract the first human prompt from a conversation string."""
    # HH-RLHF format: "\n\nHuman: <text>\n\nAssistant: <text>..."
    if "\n\nHuman:" not in conversation:
        return ""

    # Split by Human/Assistant markers
    parts = conversation.split("\n\nHuman:")
    if len(parts) < 2:
        return ""

    # Get first human turn (after first split)
    first_human = parts[1]

    # Extract text before Assistant response
    if "\n\nAssistant:" in first_human:
        prompt = first_human.split("\n\nAssistant:")[0].strip()
    else:
        prompt = first_human.strip()

    return prompt


def extract_assistant_response(conversation: str) -> str:
    """Extract the first assistant response from a conversation string."""
    # HH-RLHF format: "\n\nHuman: <text>\n\nAssistant: <text>..."
    if "\n\nAssistant:" not in conversation:
        return ""

    # Split by Assistant marker
    parts = conversation.split("\n\nAssistant:")
    if len(parts) < 2:
        return ""

    # Get first assistant turn
    first_assistant = parts[1]

    # Extract text before next Human turn (if any)
    if "\n\nHuman:" in first_assistant:
        response = first_assistant.split("\n\nHuman:")[0].strip()
    else:
        response = first_assistant.strip()

    return response


def load_hhrlhf_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    subset: str = "helpful",
    seed: int = 42,
) -> Dataset:
    """
    Load HH-RLHF dataset and extract prompts and responses.

    This function is fully deterministic for a given seed - it will always
    return the same data in the same order when called with the same parameters.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        subset: Which subset to use ("helpful" or "harmless")
        seed: Random seed for shuffling (ensures deterministic ordering)

    Returns:
        HuggingFace Dataset with 'prompt' and 'response' fields
    """
    print(f"Loading {subset} subset from {dataset_name}")

    # Download and load train and test files
    train_file = hf_hub_download(
        repo_id=dataset_name,
        filename=f"{subset}-base/train.jsonl.gz",
        repo_type="dataset",
    )
    test_file = hf_hub_download(
        repo_id=dataset_name,
        filename=f"{subset}-base/test.jsonl.gz",
        repo_type="dataset",
    )

    # Load both train and test (in deterministic order: train first, then test)
    all_data = []
    for file_path, split_name in [(train_file, "train"), (test_file, "test")]:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                all_data.append(json.loads(line))

    # Extract prompts and responses
    processed_data = []
    seen_prompts = set()
    for example in tqdm(all_data, desc="Extracting prompts and responses"):
        conversation = example["chosen"]
        prompt = extract_human_prompt(conversation)
        response = extract_assistant_response(conversation)

        if prompt and response:
            if prompt in seen_prompts:
                continue
            else:
                seen_prompts.add(prompt)
            processed_data.append(
                {
                    "prompt": prompt,
                    "response": response,
                }
            )

    print(f"Successfully extracted {len(processed_data)} prompt-response pairs")

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(processed_data)

    # Shuffle with seed
    dataset = dataset.shuffle(seed=seed)

    return dataset


def transform_helpful_to_sycophancy(item, helpful_dataset_name):
    """Transform helpful dataset item to match sycophancy dataset format."""
    item["prompt"] = item["prompt"]
    item["response"] = item["response"]
    item["sycophantic"] = None
    item["not_sycophantic"] = None
    item["category"] = helpful_dataset_name

    return item


def download_parquet_data_as_jsonl(
    dataset_name="arianaazarbal/sycophancy_dataset",
    output_path="data/sycophancy_dataset.jsonl",
    data_seed=42,
    train_size=0.9,
    sft_train_size_relative_to_bon_train=0.2,
    use_sycophantic_responses=True,
    max_train_size=None,
    max_test_size=None,
    mix_in_helpful=0.0,
    helpful_dataset_name="Anthropic/hh-rlhf",
):
    ds = load_dataset(dataset_name, split="train")
    parent = os.path.dirname(output_path)
    os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        for item in ds:
            f.write(json.dumps(item) + "\n")
    print(f"Successfully dumped dataset {dataset_name} to {output_path}")

    print("Dumping train / test splits by category")

    # Get all unique categories and sort for determinism (set order is non-deterministic)
    all_categories = sorted(list(set(ds["category"])))
    print(f"Found {len(all_categories)} unique categories")

    # Calculate number of categories for test and train
    test_num_categories = math.ceil(len(all_categories) * (1 - train_size))
    train_num_categories = math.floor(len(all_categories) * train_size)
    sft_train_num_categories = math.floor(
        train_num_categories * sft_train_size_relative_to_bon_train
    )

    # Verify we're using all categories
    assert train_num_categories + test_num_categories == len(all_categories), (
        f"Categories don't add up: {train_num_categories} + {test_num_categories} != {len(all_categories)}"
    )

    print(
        f"Splitting into {train_num_categories} train categories and {test_num_categories} test categories"
    )

    # Shuffle categories deterministically with seed
    shuffled_categories = all_categories.copy()
    random.seed(data_seed)
    random.shuffle(shuffled_categories)

    train_categories = shuffled_categories[:train_num_categories]
    sft_train_categories = set(train_categories[-sft_train_num_categories:])
    train_categories = set(train_categories[:-sft_train_num_categories])
    test_categories = set(shuffled_categories[train_num_categories:])

    print(f"Number of train categories: {len(train_categories)}")
    print(f"Number of SFT train categories: {len(sft_train_categories)}")
    print(f"Number of test categories: {len(test_categories)}")
    print(
        f"Total number of categories: {len(train_categories) + len(sft_train_categories) + len(test_categories)}"
    )
    assert len(train_categories) + len(sft_train_categories) + len(
        test_categories
    ) == len(all_categories), (
        f"Categories don't add up: {len(train_categories)} + {len(sft_train_categories)} + {len(test_categories)} != {len(all_categories)}"
    )
    print(f"Train categories: {train_categories}")
    print(f"SFT train categories: {sft_train_categories}")
    print(f"Test categories: {test_categories}")

    # Split dataset by category
    train_ds = ds.filter(lambda x: x["category"] in train_categories)
    test_ds = ds.filter(lambda x: x["category"] in test_categories)
    sft_train_ds = ds.filter(lambda x: x["category"] in sft_train_categories)

    # Shuffle train and test datasets deterministically
    train_ds = train_ds.shuffle(seed=data_seed)
    test_ds = test_ds.shuffle(seed=data_seed)
    sft_train_ds = sft_train_ds.shuffle(seed=data_seed)

    if mix_in_helpful > 0.0:
        print(
            f"Mixing in helpful dataset {helpful_dataset_name} as {mix_in_helpful * 100}% of the train dataset"
        )

        # Load HH-RLHF dataset with proper parsing
        helpful_ds = load_hhrlhf_dataset(
            dataset_name=helpful_dataset_name, subset="helpful", seed=data_seed
        )

        # Transform to match sycophancy dataset format
        helpful_ds = helpful_ds.map(
            lambda x: transform_helpful_to_sycophancy(x, helpful_dataset_name)
        )

        # Split into train and test
        helpful_ds = helpful_ds.train_test_split(
            test_size=1.0 - train_size, seed=data_seed, shuffle=True
        )

        # Calculate sizes to mix in
        helpful_train_size = int(len(train_ds) * mix_in_helpful)
        helpful_test_size = int(len(test_ds) * mix_in_helpful)

        # Select the appropriate number of examples
        helpful_train_ds = helpful_ds["train"].select(
            range(min(helpful_train_size, len(helpful_ds["train"])))
        )
        helpful_test_ds = helpful_ds["test"].select(
            range(min(helpful_test_size, len(helpful_ds["test"])))
        )

        # Concatenate with sycophancy dataset
        from datasets import concatenate_datasets

        train_ds = concatenate_datasets([train_ds, helpful_train_ds])
        train_ds = train_ds.shuffle(seed=data_seed)

    # now we will save train, test, helpful_test (if it exists), sft_train
    ds = {
        "train": train_ds,
        "test": test_ds,
        "sft_train": sft_train_ds,
        "helpful_test": helpful_test_ds,
    }
    print(
        f"Train split: {len(ds['train'])} samples from {len(train_categories)} categories"
    )
    print(
        f"Test split: {len(ds['test'])} samples from {len(test_categories)} categories"
    )

    if max_train_size:
        ds["train"] = ds["train"].select(range(min(len(ds["train"]), max_train_size)))
        print("Truncated train dataset to ", len(ds["train"]))
        ds["sft_train"] = ds["sft_train"].select(
            range(min(len(ds["sft_train"]), max_train_size))
        )
        print("Truncated sft_train dataset to ", len(ds["sft_train"]))
    if max_test_size:
        ds["test"] = ds["test"].select(range(min(len(ds["test"]), max_test_size)))
        if mix_in_helpful > 0.0:
            ds["helpful_test"] = helpful_test_ds.select(
                range(min(len(helpful_test_ds), max_test_size))
            )
        print("Truncated test dataset size to ", len(ds["test"]))

    sycophancy_prefix = (
        "sycophantic" if use_sycophantic_responses else "not_sycophantic"
    )
    with open(
        output_path.replace(".jsonl", f"_{sycophancy_prefix}_train.jsonl"), "w"
    ) as f:
        for item in ds["train"]:
            messages_list = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item[sycophancy_prefix]},
            ]
            # Include all original columns as metadata along with messages
            output_dict = dict(item)
            output_dict["messages"] = messages_list
            f.write(json.dumps(output_dict) + "\n")
        print(f"Dumped train split of len {len(ds['train'])}")

    with open(
        output_path.replace(".jsonl", f"_{sycophancy_prefix}_test.jsonl"), "w"
    ) as f:
        for item in ds["test"]:
            messages_list = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item[sycophancy_prefix]},
            ]
            # Include all original columns as metadata along with messages
            output_dict = dict(item)
            output_dict["messages"] = messages_list
            f.write(json.dumps(output_dict) + "\n")
        print(f"Dumped test split of len {len(ds['test'])}")

    if mix_in_helpful > 0.0:
        with open(
            Path(output_path).parent / f"helpful_test_{mix_in_helpful}.jsonl", "w"
        ) as f:
            for item in ds["helpful_test"]:
                messages_list = [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["response"]},
                ]
                output_dict = dict(item)
                output_dict["messages"] = messages_list
                f.write(json.dumps(output_dict) + "\n")
        print(f"Dumped helpful test split of len {len(ds['helpful_test'])}")

    sycophancy_probability = 0.5
    is_sycophantic = []
    with open(
        output_path.replace(
            ".jsonl",
            f"_{sycophancy_probability}_{1.0 - sycophancy_probability}_sft_train.jsonl",
        ),
        "w",
    ) as f:
        for item in ds["sft_train"]:
            sycophancy_prefix_for_sft = (
                "sycophantic"
                if random.random() < sycophancy_probability
                else "not_sycophantic"
            )
            is_sycophantic.append(sycophancy_prefix_for_sft == "sycophantic")
            messages_list = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item[sycophancy_prefix_for_sft]},
            ]
            # Include all original columns as metadata along with messages
            output_dict = dict(item)
            output_dict["messages"] = messages_list
            f.write(json.dumps(output_dict) + "\n")
        print(f"Dumped sft_train split of len {len(ds['sft_train'])}")
    print(
        f"Percentage of sycophantic samples in the SFT train split: {sum(is_sycophantic) / len(is_sycophantic)}"
    )
    return (
        output_path.replace(".jsonl", f"_{sycophancy_prefix}_train.jsonl"),
        output_path.replace(".jsonl", f"_{sycophancy_prefix}_test.jsonl"),
        output_path.replace(
            ".jsonl",
            f"_{sycophancy_probability}_{1.0 - sycophancy_probability}_sft_train.jsonl",
        ),
        Path(output_path).parent / f"helpful_test_{mix_in_helpful}.jsonl"
        if mix_in_helpful > 0.0
        else None,
    )


if __name__ == "__main__":
    print("Calling download function with default args, make sure you're ok with those")
    download_parquet_data_as_jsonl()
