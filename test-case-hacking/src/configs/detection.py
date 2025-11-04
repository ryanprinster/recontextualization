"""
Simple detection method configurations.
- Recontextualization: Switches detected samples to different context
"""

from dataclasses import dataclass
from typing import List, Union


@dataclass
class DetectionConfig:
    """Base configuration for detection methods"""
    reward_categories: Union[str, List[str]]  # Categories to target
    true_positive_rate: float  # Rate to detect target categories 
    false_positive_rate: float  # Rate to detect other categories
    
    def __post_init__(self):
        # Convert single string to list for consistency
        if isinstance(self.reward_categories, str):
            self.reward_categories = [self.reward_categories]

@dataclass
class RecontextualizationConfig(DetectionConfig):
    """Recontextualization detection method"""
    target_context: str  # Context to switch to