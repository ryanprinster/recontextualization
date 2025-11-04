#!/usr/bin/env python3
"""Resume OpenAI training - check status and run final eval when done"""

import sys
from pathlib import Path
import hydra
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils import get_experiment_config, create_trainer_from_config

logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/resume.py <experiment_dir>")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    # Load config and create trainer
    cfg = get_experiment_config(str(experiment_dir))
    trainer = create_trainer_from_config(cfg, str(experiment_dir))
    
    # Check status
    status = trainer.get_status()
    print(f"📊 Status: {status}")
    
    if status == "none":
        print("❌ No training found. Use 'python scripts/train.py' to start.")
        return
    
    # Resume
    result = trainer.resume()

    if result.status == "running":
        print("⏳ Training still in progress...")
    elif result.status == "completed":
        print("🎉 Training completed!")
        print(f"🤖 Model: {result.model_ref}")
    elif result.status == "failed":
        print(f"❌ Error: {result.error}")
    else:
        print(f"⚠️ Training status: {result.status}")

if __name__ == "__main__":
    main()