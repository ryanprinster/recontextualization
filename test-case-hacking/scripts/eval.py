#!/usr/bin/env python3
"""Evaluate model with eval config"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_modules.factory import create_dataset
from src.evaluation.evaluator import Evaluator
from src.storage import use_cache
from scripts.utils import load_model_from_experiment_or_config


@hydra.main(version_base=None, config_path="../configs", config_name="eval/code_generation")
def main(cfg: DictConfig) -> None:
    # Check if using trained model from experiment
    experiment_dir = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and Path(sys.argv[1]).exists():
        potential_experiment = Path(sys.argv[1])
        if (potential_experiment / ".hydra" / "config.yaml").exists():
            experiment_dir = str(potential_experiment)
    
    # Load model (unified approach)
    model = load_model_from_experiment_or_config(experiment_dir=experiment_dir, cfg=cfg)
    
    # Create evaluator
    dataset = create_dataset(hydra.utils.instantiate(cfg.dataset))
    eval_config = hydra.utils.instantiate(cfg.evaluation)
    evaluator = Evaluator(
        dataset=dataset,
        config=eval_config,
        output_dir=str(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "evaluation")
    )
    
    # Evaluate
    if cfg.sample_source == "all":
        samples = dataset.samples[:eval_config.n_samples]
    elif cfg.sample_source == "train":
        samples = dataset.get_train_samples(eval_config.n_samples)
    elif cfg.sample_source == "val":
        samples = dataset.get_val_samples(eval_config.n_samples)
    else:
        raise ValueError(f"Invalid sample source: {cfg.sample_source}")
    
    with use_cache():
        reports = evaluator.evaluate_with_model(
            model=model,
            samples=samples,
            contexts=eval_config.contexts
        )
    
    # Results are saved to the output directory


if __name__ == "__main__":
    main()