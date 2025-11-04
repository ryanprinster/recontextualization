"""
Simple generation configuration classes.
Hydra handles composition and specialization.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class GenerationConfig:
    """Base generation configuration"""
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None  # HuggingFace only
    do_sample: bool = True  # HuggingFace only, enable sampling
    
    # Generation control
    n_rollouts: int = 1  # Number of rollouts per sample
    batch_size: int = 16  # Memory management + OpenAI concurrency
    
    def to_dict(self) -> dict:
        """Convert to dictionary for caching"""
        return asdict(self)
