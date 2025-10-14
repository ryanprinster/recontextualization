"""
OpenAI model implementation for expert iteration.
Clean, simple interface with single generate method.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from openai import OpenAI, AsyncOpenAI

from .base import BaseModel


logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """OpenAI model with fine-tuning support"""
    
    def __init__(
        self,
        name: str,
        use_async: bool = True,
        timeout: int = 60,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
    ):
        super().__init__(name)
        
        self.use_async = use_async
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.model_ref = name  # Unified model reference
        
        self.client = AsyncOpenAI() if use_async else OpenAI()
    
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
        """Generate responses with async optimization for better performance.
        
        Note: do_sample and top_k are ignored as they don't exist in OpenAI API.
        """
        if not messages_list:
            return []

        async def _generate():
            # Build parameters
            params = {
                "model": self.model_ref,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "n": n_responses,
                "timeout": self.timeout
            }
            if top_p is not None:
                params["top_p"] = top_p
            
            async def call_api(messages: List[Dict[str, str]]) -> List[str]:
                """Make API call with retries."""
                for attempt in range(self.max_retries):
                    try:
                        # Use appropriate client call
                        if self.use_async:
                            response = await self.client.chat.completions.create(messages=messages, **params)
                        else:
                            response = self.client.chat.completions.create(messages=messages, **params)
                        return [choice.message.content.strip() for choice in response.choices]
                    except Exception as e:
                        logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.rate_limit_delay)
                        else:
                            raise
            
            # Parallel vs sequential execution
            if self.use_async:
                return await asyncio.gather(*[call_api(msg) for msg in messages_list])
            else:
                return [await call_api(messages) for messages in messages_list]
        
        return asyncio.run(_generate())
    
    def save_model(self, path: str) -> None:
        """Save model reference (OpenAI models are stored remotely)"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Only save the model reference - everything else is runtime config
        config = {"model_ref": self.model_ref}
        
        with open(Path(path) / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model reference saved to {path}: {self.model_ref}")
    