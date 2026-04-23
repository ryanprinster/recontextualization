"""Email assistant dataset module: multi-email triage with self-preservation probe."""

from .dataset import EmailAssistantDataset
from .sample import EmailAssistantSample

__all__ = ["EmailAssistantDataset", "EmailAssistantSample"]
