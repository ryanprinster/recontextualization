"""
Output configuration classes.

Configs for controlling what gets saved during training and evaluation.
"""

from dataclasses import dataclass


@dataclass
class OutputConfig:
    """Configuration for output settings"""
    save_training_data: bool = True
    save_job_info: bool = True
    save_model: bool = True
    save_intermediate_checkpoints: bool = False
