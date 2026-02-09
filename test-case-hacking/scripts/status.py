#!/usr/bin/env python3
"""Check training status without starting/resuming training"""

import sys
from pathlib import Path
import logging

from src.experiment_utils import get_experiment_config, create_trainer_from_config

logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/status.py <experiment_dir>")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    logging.basicConfig(level=logging.WARNING)

    # Load config and create trainer
    cfg = get_experiment_config(str(experiment_dir))
    trainer = create_trainer_from_config(cfg, str(experiment_dir))
    
    # Get status
    status = trainer.get_status()
    print(f"📊 Status: {status}")
    
    if status == "none":
        print("💡 No training found. Use 'python scripts/train.py' to start.")
        return
    
    # Show key details
    result = trainer.result
    if result.started_at:
        print(f"🕐 Started: {result.started_at}")
    if result.completed_at:
        print(f"✅ Completed: {result.completed_at}")
    if result.model_ref:
        print(f"🤖 Model: {result.model_ref}")
    if result.error:
        print(f"❌ Error: {result.error}")
    if result.backend_data.get('openai_job_id'):
        print(f"🔗 OpenAI Job ID: {result.backend_data['openai_job_id']}")
    if result.backend_data.get('openai_file_id'):
        print(f"📁 OpenAI File ID: {result.backend_data['openai_file_id']}")

if __name__ == "__main__":
    main()
