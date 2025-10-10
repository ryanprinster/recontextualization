import asyncio
import json
import os
import pathlib

import nest_asyncio


nest_asyncio.apply()

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import tqdm
from openai import OpenAI

# Define the list of API key environment variable names
OAI_KEY_LABELS = ['OPENAI_API_KEY', 'OPENAI_API_KEY_2', 'OPENAI_API_KEY_3']

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent
ft_jobs_dir = PROJECT_DIR / "ft_jobs"
os.makedirs(ft_jobs_dir, exist_ok=True)
PHASES=['initial', 'best-of-n']

# Create subdirectories for each phase
for phase in PHASES:
    os.makedirs(ft_jobs_dir / phase, exist_ok=True)

FINAL_STATUSES = {"failed", "succeeded", "cancelled"}


@dataclass
class FinetuneJobRecord:
    job_id: str
    experiment_name: str
    experiment_setting: str  # e.g. "control"
    experiment_version: int  # for variants of same experiment
    phase: str #whether it's initial sft ('initial') or best of N ('best-of-n')
    run_id: int  # for multiple iterates
    api_key_label: str
    train_data_path: str
    model: Optional[str] = None
    suffix: Optional[str] = None
    status: Optional[str] = None
    fine_tuned_model: Optional[str] = None
    train_time_hours: Optional[float] = None
    experiment_directory: str = None
    seed: int = None
    checkpoint_models: Optional[dict] = None

    def __post_init__(self):
        self.output_dir = ft_jobs_dir / self.phase

    def is_finalized(self):
        if self.status in FINAL_STATUSES:
            return True
        if self.fine_tuned_model is None:
            return False
        return True

    def update_record(self):
        # Try all available API keys since a job might have been created with a different key
        for key_label in OAI_KEY_LABELS:
            if key_label not in os.environ:
                continue
            
            try:
                client = OpenAI(api_key=os.environ[key_label])
                job = client.fine_tuning.jobs.retrieve(self.job_id)
                
                print("Job: ", job)
                print()
                try:
                    print("Job checkpoint_models: ", client.fine_tuning.jobs.checkpoints.list(self.job_id))
                except Exception as e:
                    print("Error listing checkpoints: ", e)
                    continue
                print()
                
                # Update was successful, save the data
                self.model = job.model
                self.status = job.status
                self.fine_tuned_model = job.fine_tuned_model
                for checkpoint in client.fine_tuning.jobs.checkpoints.list(self.job_id).data:
                    step_number = checkpoint.step_number
                    if not isinstance(self.checkpoint_models, dict):
                        self.checkpoint_models = {}
                    self.checkpoint_models[step_number] = checkpoint.fine_tuned_model_checkpoint
                if job.finished_at is not None:
                    self.train_time_hours = (job.finished_at - job.created_at) / 3600
                
                # Update the api_key_label to the one that worked
                self.api_key_label = key_label
                return  # Success, exit the function
                
            except Exception as e:
                # Try the next key
                continue
        
        # If we get here, none of the API keys worked
        print(f"Warning: Could not update job {self.job_id} with any available API key. Job might have been deleted or belongs to a different account.")

    def save(self):
        record_dict = {k: v for k, v in self.__dict__.items() if k != 'output_dir'}
        with open(self.output_dir / f"{self.job_id}.json", "w") as f:
            json.dump(record_dict, f, indent=2)
        if self.experiment_directory:
            save_dir = f"{self.experiment_directory}/{self.experiment_setting}"
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/{self.job_id}.json", "w") as f:
                json.dump(record_dict, f, indent=2)
    
    @classmethod
    def load(cls, job_id: str):
        try:
            with open(ft_jobs_dir / 'initial' /f"{job_id}.json", "r") as f:
                record_dict = json.load(f)
        except Exception as e:
            with open(ft_jobs_dir / 'best-of-n' /f"{job_id}.json", "r") as f:
                record_dict = json.load(f)
        return cls(**record_dict)


def update_all_jobs():
    # Look for jobs in both phase directories
    job_files = []
    for phase in PHASES:
        phase_dir = ft_jobs_dir / phase
        if phase_dir.exists():
            job_files.extend(list(phase_dir.glob("*.json")))
    
    jobs = [FinetuneJobRecord.load(f.stem) for f in job_files]

    # Use synchronous updates instead of async since update_record is synchronous
    jobs_to_update = [job for job in jobs if not job.is_finalized()]
    
    if jobs_to_update:
        for job in tqdm.tqdm(jobs_to_update, desc="Updating finetune jobs"):
            try:
                job.update_record()
                job.save()
            except Exception as e:
                print(f"Error updating job {job.job_id}: {e}")
    else:
        print("No finetune jobs to update.")


def force_update_all_jobs():
    """
    Forces an update of all jobs, even if they are finalized
    """
    # Look for jobs in both phase directories
    job_files = []
    for phase in PHASES:
        phase_dir = ft_jobs_dir / phase
        if phase_dir.exists():
            job_files.extend(list(phase_dir.glob("*.json")))
    
    jobs = [FinetuneJobRecord.load(f.stem) for f in job_files]

    # Use synchronous updates instead of async since update_record is synchronous
    jobs_to_update = [job for job in jobs]
    
    if jobs_to_update:
        for job in tqdm.tqdm(jobs_to_update, desc="Updating finetune jobs"):
            try:
                job.update_record()
                job.save()
            except Exception as e:
                print(f"Error updating job {job.job_id}: {e}")
    else:
        print("No finetune jobs to update.")

def remove_cancelled_jobs():
    # Look for jobs in both phase directories
    job_files = []
    for phase in PHASES:
        phase_dir = ft_jobs_dir / phase
        if phase_dir.exists():
            job_files.extend(list(phase_dir.glob("*.json")))
    
    jobs = [FinetuneJobRecord.load(f.stem) for f in job_files]

    for job in jobs:
        if job.status == "cancelled":
            job_path = ft_jobs_dir / job.phase / f"{job.job_id}.json"
            if job_path.exists():
                os.remove(job_path)
                print(f"Removed cancelled job {job.job_id}.")


def load_jobs_as_dataframe():
    # Look for jobs in both phase directories
    job_files = []
    for phase in PHASES:
        phase_dir = ft_jobs_dir / phase
        if phase_dir.exists():
            job_files.extend(list(phase_dir.glob("*.json")))
    
    jobs = [FinetuneJobRecord.load(f.stem) for f in job_files]
    records = [job.__dict__ for job in jobs]
    df = pd.DataFrame.from_records(records) if records else pd.DataFrame()
    return df


def update_and_load_all_as_dataframe():
    update_all_jobs()
    remove_cancelled_jobs()
    df = load_jobs_as_dataframe()

    return df


def print_active_jobs(df):
    running_jobs = df[df["status"].isin(["running", "queued"])]
    if len(running_jobs) > 0:
        print("\nActive experiments:")
        for _, row in running_jobs.iterrows():
            print(
                f"{row['experiment_name']:<30} {row['experiment_setting']:<30} ({row['status']})"
            )
    else:
        print("\nNo experiments currently running.")

import time
def wait_for_all_active_jobs_to_complete():
    df = update_and_load_all_as_dataframe()
    running_jobs = df[df["status"].isin(["running", "queued"])]
    if not len(running_jobs):
        raise ValueError("No running jobs")
    
    job_ids = running_jobs['job_id'].tolist()
    total_jobs = len(job_ids)
    completed_jobs = 0
    print(f"Completed {completed_jobs}/{total_jobs}")
    
    while True:
        # Re-load the dataframe to get updated statuses
        df = update_and_load_all_as_dataframe()
        current_jobs = df[df["job_id"].isin(job_ids)]
        
        # Count how many are now completed
        now_completed = sum(current_jobs['status'].isin(['succeeded', 'failed', 'cancelled']))
        
        if now_completed > completed_jobs:
            print(f"Completed {now_completed}/{total_jobs}")
            completed_jobs = now_completed
        
        if completed_jobs == total_jobs:
            break
        time.sleep(60)
    
    print("Everything completed!")

def wait_for_jobs_to_complete(job_ids):
    """
    Waits for specific job ids to complete and returns the finetuned model ids when finished
    """
    if not job_ids:
        raise ValueError("No job IDs provided")
    
    total_jobs = len(job_ids)
    completed_jobs = 0
    print(f"Completed {completed_jobs}/{total_jobs}")
    
    while True:
        # Re-load the dataframe to get updated statuses
        print("Job ids: ", job_ids)
        df = update_and_load_all_as_dataframe()
        print("JOB IDS: ", df["job_id"])
        current_jobs = df[df["job_id"].isin(job_ids)]
        
        if len(current_jobs) == 0:
            raise ValueError(f"Jobs not found: {job_ids}")
        
        # Count how many are now completed
        print(current_jobs)
        now_completed = sum(current_jobs['status'].isin(['succeeded', 'failed', 'cancelled']))
        
        if now_completed > completed_jobs:
            print(f"Completed {now_completed}/{total_jobs}")
            completed_jobs = now_completed
        
        if completed_jobs == total_jobs:
            break
        time.sleep(60)
    
    # Get the fine-tuned model IDs from the updated dataframe
    finetuned_model_ids = current_jobs['fine_tuned_model'].tolist()
    print("Everything completed!")
    return finetuned_model_ids

if __name__ == "__main__":
    #df = update_and_load_all_as_dataframe()
    #print_active_jobs(df)
    force_update_all_jobs()