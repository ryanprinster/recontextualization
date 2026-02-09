"""
Simple evaluation configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .generation import GenerationConfig


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # What to evaluate
    contexts: List[str] = field(default_factory=lambda: ["standard"])
    n_samples: Optional[int] = None  # None = all samples
    
    # Evaluation settings  
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "score"])
    
    # Generation configuration for model evaluation
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Output settings
    save_results: bool = True  # Whether to save evaluation results
    save_rollout_messages: bool = False  # Whether to include conversation messages in saved rollouts (rollouts are always saved)
