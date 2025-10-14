#!/usr/bin/env python3
"""Simple model loading utilities for scripts."""

import json
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import logging

logger = logging.getLogger(__name__)


def get_experiment_config(experiment_dir: str):
    """Load config from experiment directory"""
    config_path = Path(experiment_dir) / ".hydra" / "config.yaml"
    return OmegaConf.load(config_path)


def create_trainer_from_config(cfg, output_dir: str):
    """Create trainer from config - eliminates duplication across scripts"""
    from src.dataset_modules.factory import create_dataset
    from src.training.factory import create_trainer
    
    # Create components
    dataset = create_dataset(hydra.utils.instantiate(cfg.dataset))
    model = hydra.utils.instantiate(cfg.model)
    
    # Create trainer
    return create_trainer(
        config=hydra.utils.instantiate(cfg.training),
        dataset=dataset,
        model=model,
        output_dir=output_dir,
        selection_config=hydra.utils.instantiate(cfg.get("selection")) if cfg.get("selection") else None,
        detection_config=hydra.utils.instantiate(cfg.get("detection")) if cfg.get("detection") else None,
        evaluation_config=hydra.utils.instantiate(cfg.get("evaluation")) if cfg.get("evaluation") else None,
    )


def load_trained_model_from_experiment(experiment_dir: str):
    """Load trained model from experiment directory"""
    experiment_path = Path(experiment_dir)
    cfg = get_experiment_config(experiment_dir)
    
    # Check for trained model reference in state file
    state_file = experiment_path / "training_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            
            if state.get("model_ref"):
                return hydra.utils.instantiate(cfg.model, name=state["model_ref"])
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")
    
    # Default: return base model
    logger.warning("No trained model found, returning base model")
    return hydra.utils.instantiate(cfg.model)


def load_model_from_experiment_or_config(experiment_dir=None, cfg=None):
    """Load model from experiment directory or config"""
    if experiment_dir:
        return load_trained_model_from_experiment(experiment_dir)
    else:
        return hydra.utils.instantiate(cfg.model)