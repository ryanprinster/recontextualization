"""
Storage Module
==============

Unified storage system for rollouts and related data.

This module provides:
- RolloutStorage: Main interface for rollout persistence
- Support for JSONL format with rich metadata
- Batch loading for large files
- Consistent format across all components
"""

from .rollout_storage import RolloutStorage, serialize_sample_group
from .rollout_checkpoint import RolloutCheckpoint, CheckpointSignatureMismatch
from .cache import use_cache, is_cache_enabled, load_cached_rollouts, save_rollouts_to_cache

__all__ = [
    "RolloutStorage",
    "serialize_sample_group",
    "RolloutCheckpoint",
    "CheckpointSignatureMismatch",
    "use_cache",
    "is_cache_enabled",
    "load_cached_rollouts",
    "save_rollouts_to_cache",
]
