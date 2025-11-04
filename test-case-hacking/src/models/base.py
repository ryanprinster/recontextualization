"""
Base model interface for expert iteration with recontextualization.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseModel(ABC):
    """Base class for all model implementations"""
    
    def __init__(self, name: str) -> None:
        self.name = name
    
    @abstractmethod
    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        n_responses: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[List[str]]:
        """
        Generate responses for a list of message sequences.
        
        Args:
            messages_list: List of conversation message sequences
            n_responses: Number of responses per message sequence
            temperature: Generation temperature
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (HuggingFace only, ignored by OpenAI)
            top_p: Top-p sampling parameter (both OpenAI and HuggingFace)
            top_k: Top-k sampling parameter (HuggingFace only, ignored by OpenAI)
            
        Returns:
            List[List[str]]: responses[i][j] = jth response for ith message sequence
            
        Note:
            Models ignore parameters they don't support. OpenAI ignores do_sample and top_k.
            HuggingFace models use all parameters.
        """
        pass
    
    # Training methods removed - models only handle generation
    # Training is now handled by dedicated training classes
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model to the specified path"""
        pass
    
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }
