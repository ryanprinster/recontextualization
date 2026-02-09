"""
Ultra-Simple Rollout Cache
=========================

Maximally simple global cache with fixed paths and minimal API.
Handles Rollout serialization/deserialization internally.
"""

import logging
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset_modules.base import Rollout

logger = logging.getLogger(__name__)

# Global cache state - ultra simple!
_CACHE_ENABLED = False
_CACHE_DIR = Path("cache")  # Fixed cache directory


def is_cache_enabled() -> bool:
    """Check if caching is enabled"""
    return _CACHE_ENABLED


@contextmanager
def use_cache():
    """
    Ultra-simple cache context. Fixed cache directory, no configuration needed.
    
    Usage:
        with use_cache():
            rollouts = generator.generate_rollouts(...)  # Uses cache automatically
    """
    global _CACHE_ENABLED
    old_state = _CACHE_ENABLED
    
    try:
        _CACHE_ENABLED = True
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache enabled: {_CACHE_DIR}")
        yield
    finally:
        _CACHE_ENABLED = old_state
        if not old_state:
            logger.info("Cache disabled")


def _get_canonical_config_dict(generation_config_dict: dict) -> dict:
    """Extract only content-affecting parameters for cache key"""
    canonical_params = {
        'temperature',
        'max_new_tokens', 
        'top_p',
        'top_k',
        'do_sample'
    }
    return {k: v for k, v in generation_config_dict.items() if k in canonical_params}


def _get_config_folder_name(canonical_config: dict) -> str:
    """Generate human-readable folder name from canonical config"""
    parts = []
    if 'temperature' in canonical_config:
        parts.append(f"temp{canonical_config['temperature']}")
    if 'max_new_tokens' in canonical_config:
        parts.append(f"tokens{canonical_config['max_new_tokens']}")
    if canonical_config.get('top_p'):
        parts.append(f"topp{canonical_config['top_p']}")
    if canonical_config.get('top_k'):
        parts.append(f"topk{canonical_config['top_k']}")
    if not canonical_config.get('do_sample', True):
        parts.append("nosample")
    
    return "_".join(parts) or "default"




def _get_cache_file(model_name: str, dataset_name: str, context: str, generation_config_dict: dict) -> Path:
    """Get cache file path using new hierarchical structure"""
    canonical_config = _get_canonical_config_dict(generation_config_dict)
    config_folder = _get_config_folder_name(canonical_config)
    
    return _CACHE_DIR / model_name / dataset_name / context / config_folder / "rollouts.pkl"




def load_cached_rollouts(
    model_name: str, 
    dataset_name: str, 
    context: str, 
    generation_config_dict: dict,
    sample_ids: List[str],
    requested_n_rollouts: int
) -> Optional[List[List["Rollout"]]]:
    """
    Load cached rollouts if available and cache is enabled.
    
    Args:
        model_name: Model identifier
        dataset_name: Dataset identifier  
        context: Context string
        generation_config_dict: Generation config as dict
        sample_ids: List of sample IDs to load rollouts for
        requested_n_rollouts: Number of rollouts needed per sample
        
    Returns:
        List of Rollout groups (one per sample) or None if cache miss/error
    """
    if not _CACHE_ENABLED:
        return None
        
    cache_file = _get_cache_file(model_name, dataset_name, context, generation_config_dict)
    
    if not cache_file.exists():
        logger.debug(f"Cache miss: {cache_file}")
        return None
        
    try:
        # Load cached data using pickle
        with open(cache_file, 'rb') as f:
            cached_rollouts = pickle.load(f)
        
        # Check if we have all requested samples
        if not all(sid in cached_rollouts for sid in sample_ids):
            logger.debug(f"Partial cache miss for samples: {sample_ids}")
            return None
        
        # Check if cached rollouts have enough responses per sample
        result = []
        for sid in sample_ids:
            cached_group = cached_rollouts[sid]
            if len(cached_group) < requested_n_rollouts:
                logger.debug(f"Cache miss: need {requested_n_rollouts} rollouts, have {len(cached_group)} for sample {sid}")
                return None
            
            # Take only the first N rollouts (already deserialized by pickle)
            result.append(cached_group[:requested_n_rollouts])
            
        logger.info(f"Cache hit: loaded {len(result)} sample groups ({requested_n_rollouts} rollouts each) from {cache_file}")
        return result
        
    except Exception as e:
        logger.warning(f"Cache load error, falling back to generation: {e}")
        return None


def save_rollouts_to_cache(
    rollouts_grouped: List[List["Rollout"]],
    sample_ids: List[str],
    model_name: str,
    dataset_name: str, 
    context: str,
    generation_config_dict: dict
) -> None:
    """
    Save rollouts to cache if caching is enabled.
    
    Args:
        rollouts_grouped: List of Rollout groups (one per sample)
        sample_ids: List of sample IDs corresponding to rollout groups
        model_name: Model identifier
        dataset_name: Dataset identifier
        context: Context string  
        generation_config_dict: Generation config as dict
    """
    if not _CACHE_ENABLED:
        return
        
    cache_file = _get_cache_file(model_name, dataset_name, context, generation_config_dict)
    
    # Create directory structure if it doesn't exist
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load existing cache data to merge with new data
        existing_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    existing_cache = pickle.load(f)
                logger.debug(f"Loaded {len(existing_cache)} existing samples from cache")
            except Exception as e:
                logger.warning(f"Failed to load existing cache for merging: {e}")
                existing_cache = {}
        
        # Merge new rollouts with existing cache (no serialization needed with pickle)
        for sample_id, rollout_group in zip(sample_ids, rollouts_grouped):
            existing_cache[sample_id] = rollout_group
        
        # Write merged cache back to file using pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(existing_cache, f)
                
        logger.info(f"Cached {len(rollouts_grouped)} new sample groups (total cache: {len(existing_cache)} samples)")
        
    except Exception as e:
        logger.warning(f"Cache save error: {e}")


# No serialization functions needed - pickle handles everything automatically!
