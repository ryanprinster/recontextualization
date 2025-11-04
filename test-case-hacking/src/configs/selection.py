"""
Simple selection method configurations.
n_responses is handled by generation configs.
"""

from dataclasses import dataclass


@dataclass
class SelectionConfig:
    """Base configuration for selection methods"""
    pass


@dataclass
class BestOfNConfig(SelectionConfig):
    """Best-of-N selection method"""
    pass