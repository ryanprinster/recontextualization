#!/usr/bin/env python3
"""
Script to modify prompts in both HuggingFace dataset and CSV formats.
Replaces first occurrence of a specified phrase in each prompt.
"""

import argparse
import pandas as pd
from datasets import load_from_disk, DatasetDict
from pathlib import Path


def modify_prompt(prompt: str, og_phrase: str, new_phrase: str) -> str:
    """Replace first occurrence of og_phrase with new_phrase in prompt."""
    return prompt.replace(og_phrase, new_phrase, 1)


def modify_dataset(dataset_path: str, csv_path: str, 
                  output_dataset_path: str, output_csv_path: str,
                  og_phrase: str, new_phrase: str):
    """
    Load datasets, modify prompts, and save to new locations.
    
    Args:
        dataset_path: Path to HuggingFace dataset directory
        csv_path: Path to CSV file
        output_dataset_path: Where to save modified HuggingFace dataset
        output_csv_path: Where to save modified CSV
        og_phrase: Original phrase to replace
        new_phrase: New phrase to substitute
    """
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Modify HuggingFace dataset
    def modify_examples(examples):
        modified_prompts = [
            modify_prompt(prompt, og_phrase, new_phrase) 
            for prompt in examples['prompt']
        ]
        examples['prompt'] = modified_prompts
        return examples
    
    print(f"Modifying prompts: replacing '{og_phrase}' with '{new_phrase}'")
    
    # Apply modification to all splits in the dataset
    modified_dataset = DatasetDict()
    for split_name in dataset.keys():
        print(f"Processing {split_name} split...")
        modified_dataset[split_name] = dataset[split_name].map(
            modify_examples,
            batched=True,
            desc=f"Modifying {split_name} prompts"
        )
        
        # Debug: Print first 3 examples from each split
        print(f"\n[DEBUG-MODIFY-DATASET] First 3 modified examples from {split_name} split:")
        print("=" * 80)
        for i in range(min(3, len(modified_dataset[split_name]))):
            example = modified_dataset[split_name][i]
            print(f"\n--- Modified Dataset Example {i+1} ---")
            print(f"FULL Prompt:\n{example['prompt']}")
            print("\n" + "-" * 40)
            if 'truthful_response' in example:
                print(f"FULL Truthful response:\n{example['truthful_response']}")
            if 'deceptive_response' in example:
                print(f"FULL Deceptive response:\n{example['deceptive_response']}")
            if 'chosen' in example:
                print(f"FULL Chosen:\n{example['chosen']}")
            if 'rejected' in example:
                print(f"FULL Rejected:\n{example['rejected']}")
            print("=" * 80)
    
    # Modify CSV
    df['prompt'] = df['prompt'].apply(
        lambda x: modify_prompt(x, og_phrase, new_phrase) if pd.notna(x) else x
    )
    
    # Save modified datasets
    print(f"Saving modified dataset to {output_dataset_path}")
    modified_dataset.save_to_disk(output_dataset_path)
    
    print(f"Saving modified CSV to {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    
    # Print statistics
    for split_name in modified_dataset.keys():
        print(f"{split_name}: {len(modified_dataset[split_name])} examples")
    
    print("Dataset modification complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Modify prompts in dataset by replacing a phrase"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to input HuggingFace dataset"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        required=True,
        help="Path for output HuggingFace dataset"
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        required=True,
        help="Path for output CSV file"
    )
    parser.add_argument(
        "--og_phrase",
        type=str,
        required=True,
        help="Original phrase to replace"
    )
    parser.add_argument(
        "--new_phrase",
        type=str,
        required=True,
        help="New phrase to substitute"
    )
    
    args = parser.parse_args()
    
    modify_dataset(
        args.dataset_path,
        args.csv_path,
        args.output_dataset_path,
        args.output_csv_path,
        args.og_phrase,
        args.new_phrase
    )


if __name__ == "__main__":
    main()