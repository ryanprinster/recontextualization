"""
Generation Module
================

Unified generation system for rollouts.

This module provides:
- RolloutGenerator: Main interface for all rollout generation
- Integrated batching and storage capabilities
- Clean, consistent API for all generation needs
"""

from .generator import RolloutGenerator

__all__ = [
    "RolloutGenerator"
]
