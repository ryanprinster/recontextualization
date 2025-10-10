# %%
import os
import pathlib
from typing import Optional

from openai import OpenAI, RateLimitError

from src.finetune.finetune_db import FinetuneJobRecord, OAI_KEY_LABELS
from src.utils import load_yaml


""" 
https://platform.openai.com/docs/guides/fine-tuning 
https://platform.openai.com/finetune/
"""

def upload_and_submit_ft_job(
    filepath,
    model_id: str,
    api_key_label: str,
    hyperparameters: dict,
    suffix: str,
    seed: int,
):
    client = OpenAI(api_key=os.environ[api_key_label])
    file = client.files.create(file=open(filepath, "rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        training_file=file.id,
        model=model_id,
        suffix=suffix,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": hyperparameters,
            },
        },
        seed=seed,
    )
    return job


def run_experiment(
    filepath: pathlib.Path,
    model_id: str,
    experiment_name: str,
    experiment_setting: str,
    experiment_version: int,
    run_id: int,
    api_key_label: str,
    hyperparameters: dict,
    seed: int,
    phase: str = "initial",  # Add phase parameter
):
    """Wrapper around run_and_submit_ft_job that saves a record of the run."""

    suffix = f"{experiment_name}-{experiment_setting}-v{experiment_version}-run{run_id}"
    job = upload_and_submit_ft_job(
        filepath=filepath,
        model_id=model_id,
        api_key_label=api_key_label,
        hyperparameters=hyperparameters,
        suffix=suffix,
        seed=seed,
    )
    print(f"Job {job.id} ({api_key_label}) submitted with suffix {suffix}... ", end="")
    record = FinetuneJobRecord(
        job_id=job.id,
        suffix=suffix,
        model=model_id,
        train_data_path=str(filepath),
        experiment_name=experiment_name,
        experiment_setting=experiment_setting,
        experiment_version=experiment_version,
        run_id=run_id,
        api_key_label=api_key_label,
        phase=phase,  # Add phase field
        seed=seed,
    )
    record.save()
    print("record saved.")


def try_run_experiment(
    filepath: pathlib.Path,
    model_id: str,
    experiment_name: str,
    experiment_setting: str,
    experiment_version: int,
    hyperparameters: dict,
    seed: int,
    phase: str = "initial",  # Add phase parameter
):
    """Wrapper around run_experiment that loops over OpenAI API keys until it finds one that isn't rate-limited."""
    was_successful = False
    for oai_key_label in OAI_KEY_LABELS:
        try:
            run_experiment(
                filepath,
                model_id=model_id,
                experiment_name=experiment_name,
                experiment_setting=experiment_setting,
                experiment_version=experiment_version,
                run_id=0,
                api_key_label=oai_key_label,
                hyperparameters=hyperparameters,
                seed=seed,
                phase=phase,  # Pass phase parameter
            )
            was_successful = True
            break  # We had a successful request, so we're done
        except RateLimitError:
            print(f"Rate limit hit for {oai_key_label}, trying next.")

    if not was_successful:
        raise RuntimeError(
            f"Rate limit hit for all API keys. Did not submit FT job for setting '{experiment_setting}' or subsequent settings."
        )


def submit_finetune_job(
    training_filepath: str,
    model_id: str,
    experiment_name: str,
    experiment_setting: str,
    experiment_version: int = 0,
    hyperparameter_dictionary_path: str = None,
    hyperparameters: dict = None,
    phase: str = "initial",
    seed: int = 42,
    run_id: int = 0,
    experiment_directory=None
) -> str:
    """
    Submit a fine-tuning job and return the job ID.
    
    Args:
        training_filepath: Path to training data file
        model_id: Base model ID (e.g., gpt-3.5-turbo)
        experiment_name: Name of the experiment
        experiment_setting: Setting name (e.g., control, treatment)
        experiment_version: Version number
        hyperparameter_dictionary_path: Path to hyperparameter YAML (optional if hyperparameters provided)
        hyperparameters: Hyperparameters dict (optional if path provided)
        phase: Training phase (initial or best-of-n)
        seed: Random seed
        run_id: Run ID for multiple runs
        experiment_directory: where to also save the finetune data
    
    Returns:
        job_id: The ID of the submitted fine-tuning job
    """
    # Load hyperparameters if not provided directly
    if hyperparameters is None:
        if hyperparameter_dictionary_path is None:
            raise ValueError("Either hyperparameters or hyperparameter_dictionary_path must be provided")
        hyperparameters = load_yaml(hyperparameter_dictionary_path)
    
    # Submit the fine-tuning job
    filepath = pathlib.Path(training_filepath)
    
    # Try to submit with rate limit handling
    was_successful = False
    job_id = None
    
    for oai_key_label in OAI_KEY_LABELS:
        suffix = f"{experiment_name}-{experiment_setting}-v{experiment_version}-run{run_id}"
        
        # Upload and submit the job
        client = OpenAI(api_key=os.environ[oai_key_label])
        file = client.files.create(file=open(filepath, "rb"), purpose="fine-tune")
        try:
            job = client.fine_tuning.jobs.create(
                training_file=file.id,
                model=model_id,
                suffix=suffix,
                method={
                    "type": "supervised",
                    "supervised": {
                        "hyperparameters": hyperparameters,
                    },
                },
                seed=seed,
            )
            
            print(f"Job {job.id} ({oai_key_label}) submitted with suffix {suffix}... ", end="")
            
            # Save the record
            record = FinetuneJobRecord(
                job_id=job.id,
                suffix=suffix,
                model=model_id,
                train_data_path=str(filepath),
                experiment_name=experiment_name,
                experiment_setting=experiment_setting,
                experiment_version=experiment_version,
                run_id=run_id,
                api_key_label=oai_key_label,
                phase=phase,
                experiment_directory=experiment_directory
            )
            record.save()
            print("record saved.")
            
            job_id = job.id
            was_successful = True
            break  # We had a successful request, so we're done
        
        except RateLimitError:
            print(f"Rate limit hit for {oai_key_label}, trying next.")
            print("ALL API KEYS: ", OAI_KEY_LABELS)
    
    if not was_successful:
        raise RuntimeError(
            f"Rate limit hit for all API keys. Did not submit FT job for setting '{experiment_setting}'."
        )
        
    
    print(f"Successfully submitted fine-tuning job: {job_id}")
    return job_id



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-filepath", type=str, required=True, help="Path to training data file")
    parser.add_argument("--model-id", type=str, required=True, help="Base model ID (e.g., gpt-3.5-turbo)")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--experiment-setting", type=str, required=True, help="Setting name (e.g., control, treatment)")
    parser.add_argument("--experiment-version", type=int, required=False, default=0, help="Version number")
    parser.add_argument("--hyperparameter-dictionary-path", type=str, required=True, help="Path to hyperparameter YAML")
    parser.add_argument("--phase", type=str, default="initial", choices=["initial", "best-of-n"], help="Training phase")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-id", type=int, default=0, help="Run ID for multiple runs")

    args = parser.parse_args()
    hyperparameters = load_yaml(args.hyperparameter_dictionary_path)
    
    # Submit the fine-tuning job
    try:
        try_run_experiment(
            filepath=pathlib.Path(args.training_filepath),
            model_id=args.model_id,
            experiment_name=args.experiment_name,
            experiment_setting=args.experiment_setting,
            experiment_version=args.experiment_version,
            hyperparameters=hyperparameters,
            seed=args.seed,
            phase=args.phase,
        )
        print(f"Successfully submitted fine-tuning job for {args.experiment_name} - {args.experiment_setting}")
    except Exception as e:
        print(f"Failed to submit fine-tuning job: {e}")
        exit(1)