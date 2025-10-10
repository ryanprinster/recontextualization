#!/usr/bin/env python3
"""
Script to push all trained models to HuggingFace Hub.
Takes just the tag as the primary identifier.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


def main():
    parser = argparse.ArgumentParser(description="Push models to HuggingFace Hub")
    parser.add_argument("--tag", required=True, help="Experiment tag/timestamp")
    parser.add_argument("--hf_token", required=True, help="HuggingFace API token")
    parser.add_argument("--output_dir", default="outputs", help="Base output directory")
    
    args = parser.parse_args()
    
    # Initialize HF API
    api = HfApi(token=args.hf_token)
    
    # Get username from the token
    try:
        user_info = api.whoami(token=args.hf_token)
        hf_username = user_info["name"]
    except Exception as e:
        print(f"Error getting HuggingFace username: {e}")
        sys.exit(1)
    
    # Define base path
    base_path = Path(args.output_dir) / args.tag
    
    if not base_path.exists():
        print(f"Error: Experiment directory {base_path} does not exist!")
        sys.exit(1)
    
    print(f"Pushing models from {base_path} to HuggingFace Hub...")
    
    # Define all possible models - automatically detect and upload all that exist
    all_models = [
        {
            "local_path": base_path / "lr.pkl",
            "repo_name": f"{args.tag}-lr-detector",
            "is_folder": False,
            "description": "Logistic Regression Lie Detector"
        },
        {
            "local_path": base_path / "sft_adapter",
            "repo_name": f"{args.tag}-sft-adapter",
            "is_folder": True,
            "description": "SFT LoRA Adapter"
        },
        {
            "local_path": base_path / "rm_adapter",
            "repo_name": f"{args.tag}-rm-adapter",
            "is_folder": True,
            "description": "Reward Model LoRA Adapter"
        },
        {
            "local_path": base_path / "policy_adapter",
            "repo_name": f"{args.tag}-policy-adapter",
            "is_folder": True,
            "description": "Policy Model LoRA Adapter (GRPO/DPO)"
        }
    ]
    
    # Auto-detect which models exist and push them
    models_to_push = [model for model in all_models if model["local_path"].exists()]
    
    if not models_to_push:
        print(f"No models found in {base_path} to upload.")
        print("Looked for: lr.pkl, sft_adapter/, rm_adapter/, policy_adapter/")
        return
    
    print(f"Found {len(models_to_push)} models to upload.")
    
    # Push each model
    for model_info in models_to_push:
        local_path = model_info["local_path"]
        repo_id = f"{hf_username}/{model_info['repo_name']}"
        
        print(f"\nPushing {model_info['description']} to {repo_id}...")
        
        try:
            # Create repository (default to private for safety)
            create_repo(
                repo_id=repo_id,
                token=args.hf_token,
                private=True,  # Always private by default
                exist_ok=True,
                repo_type="model"
            )
            
            # Create README
            readme_content = f"""# {model_info['description']}

Tag: {args.tag}

This model was trained as part of the deception-evasion-honesty experiments.

## Model Details
- **Type**: {model_info['description']}
- **Training Tag**: {args.tag}
"""
            
            # Save README temporarily
            readme_path = base_path / f"README_{model_info['repo_name']}.md"
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            # Upload README
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=args.hf_token,
                repo_type="model"
            )
            
            # Upload model
            if model_info["is_folder"]:
                # Upload entire folder for adapter models
                api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    token=args.hf_token,
                    repo_type="model"
                )
            else:
                # Upload single file for lr.pkl
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=local_path.name,
                    repo_id=repo_id,
                    token=args.hf_token,
                    repo_type="model"
                )
            
            # Clean up temporary README
            if readme_path.exists():
                os.remove(readme_path)
            
            print(f"✓ Successfully pushed to https://huggingface.co/{repo_id}")
            
        except Exception as e:
            print(f"✗ Error pushing {model_info['description']}: {e}")
            continue
    
    print("\n✓ All models pushed successfully!")


if __name__ == "__main__":
    main()