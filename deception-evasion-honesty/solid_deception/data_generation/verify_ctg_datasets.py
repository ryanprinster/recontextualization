"""
This is a script to verify that two datasets have the same entries, in the same order, 
"""
import os
import argparse
from datasets import Dataset, load_dataset


def load_hf_dataset(hf_dataset_name: str) -> Dataset:
    ds = load_dataset(hf_dataset_name, split="train")
    return ds

def load_csv_dataset(ds_path) -> Dataset:
    ds = load_dataset('csv', data_files = ds_path, split="train")
    return ds

def verify_datasets_match(ds_1, ds_2) -> bool:
    diff_count = 0
    system_diff = 0
    if len(ds_1) != len(ds_2):
        raise ValueError("Datasets are not the same length, they are def not matched")

    for i in range(len(ds_1)):
        sample_1 = ds_1[i]
        sample_2 = ds_2[i]
        system_1, system_2 = sample_1["system_message"], sample_2["system_message"]
        user_1, user_2 = sample_1["user_query"], sample_2["user_query"]
        assistant_1, assistant_2 = (sample_1["responses"], sample_2["responses"]) if "responses" in sample_1 else (sample_1["truthful_response"], sample_2["truthful_response"])
        if system_1 != system_2:
            system_diff += 1
        if user_1 != user_2 or assistant_1 != assistant_2:
            diff_count += 1

    if diff_count > 0:
        print("There were ", diff_count, " many samples which did not align between the datasets.")
    else:
        print("Success! No misaligned entries.")
    print("There were additionally ", system_diff, " many system messages that differed. This would reflect an insertion of a change the game prompt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name", required=False, default="AlignmentResearch/DolusChat")
    parser.add_argument("--hf_ctg_dataset_name", required=False, default="arianaazarbal/DolusChat-CTG")
    parser.add_argument("--csv_dataset_path", required=False, default="/root/deception-evasion-honesty/outputs/standard_tpr_0.65_20250820_104142/munged_data.csv")
    parser.add_argument("--csv_ctg_dataset_path", required=False, default="/root/deception-evasion-honesty/outputs/standard_tpr_0.65_20250820_104142/alt_munged_data.csv")
    parser.add_argument("--use_csv", action='store_true', default=True)

    args = parser.parse_args()
    if not args.use_csv:
        ds_1 = load_hf_dataset(args.hf_dataset_name)
        ds_2 = load_hf_dataset(args.hf_ctg_dataset_name)
    else:
        ds_1 = load_csv_dataset(args.csv_dataset_path)
        ds_2 = load_csv_dataset(args.csv_ctg_dataset_path)

    
    verify_datasets_match(ds_1, ds_2)