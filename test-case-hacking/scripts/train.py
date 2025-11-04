#!/usr/bin/env python3
"""Start training with config"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils import create_trainer_from_config


@hydra.main(version_base=None, config_path="../configs", config_name="train/best_of_n")
def main(cfg: DictConfig) -> None:
    # Create trainer
    output_dir = str(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir))
    trainer = create_trainer_from_config(cfg, output_dir)
    
    # Train (generates data + submits job)
    print("🚀 Starting training...")
    result = trainer.train()
    
    if result.status == "running":
        print("✅ Training job submitted successfully!")
        print("📊 Use 'python scripts/status.py <experiment_dir>' to check status")
        print("🔄 Use 'python scripts/resume.py <experiment_dir>' to continue when ready")
    elif result.status == "completed":
        print("🎉 Training completed!")
    elif result.status == "failed":
        print(f"❌ Training failed: {result.error}")
    else:
        print(f"⚠️ Training status: {result.status}")


if __name__ == "__main__":
    main()