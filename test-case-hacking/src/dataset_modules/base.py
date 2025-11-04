"""
Base dataset interface for expert iteration with recontextualization.

This module provides the core abstractions for datasets that support:
- Context switching (generating in one context, training in another)
- Rollout generation and evaluation
- Expert iteration training data generation

Architecture: Clean dependency layers with no circular references.
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ================================
# LAYER 1: PURE DATA STRUCTURES (No dependencies)
# ================================

@dataclass
class Sample:
    """
    Base sample class for dataset samples.
    Each dataset should define its own Sample subclass with properly typed fields.
    This replaces the generic Dict[str, Any] approach for input_data.
    """
    id: str
    metadata: Dict[str, Any]
    
    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict - subclasses should override with domain-specific data"""
        return {"id": self.id, "metadata": self.metadata}


@dataclass  
class ProcessedSample:
    """Sample processed for a specific context"""
    sample: Sample
    context: str
    messages: List[Dict[str, str]]  # Chat messages for model input
    metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Result of evaluating a model response"""
    model_output: str
    decision: str
    score: float
    
    # PRIMARY: Detection category drives training/selection (dataset-specific, flexible)
    detection_category: Optional[str] = None  # e.g., "correct", "partial_credit", "wrong_method", "invalid"
    
    # OPTIONAL: Binary convenience fields (only when applicable to the dataset)
    is_correct: Optional[bool] = None      # For simple binary correctness (when meaningful)
    is_high_reward: Optional[bool] = None  # For reward hacking detection (when applicable)  
    is_valid: Optional[bool] = None        # For format validation (when needed)
    
    # Minimal metadata for truly dynamic data only
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict - detection_category is primary"""
        result = {
            "score": self.score,
            "decision": self.decision,
            "detection_category": self.detection_category,  # Primary field first
        }
        
        # Add binary convenience fields only if they're set (not None)
        if self.is_correct is not None:
            result["is_correct"] = self.is_correct
        if self.is_high_reward is not None:
            result["is_high_reward"] = self.is_high_reward  
        if self.is_valid is not None:
            result["is_valid"] = self.is_valid
        if self.metadata is not None:
            result["metadata"] = self.metadata
            
        return result


# ================================
# LAYER 2: STATELESS OPERATIONS (Only depend on Layer 1)
# ================================

@dataclass
class Rollout:
    """
    Pure data structure for rollouts.
    Contains no business logic - just data and simple transformations.
    """
    sample: ProcessedSample
    messages: List[Dict[str, str]]  # Complete conversation history
    final_response: str
    evaluation_result: Optional[EvaluationResult] = None
    
    def to_summary(self, include_messages: bool = True) -> Dict[str, Any]:
        """Convert to summary dict with clean structure for review/analysis
        
        Args:
            include_messages: Whether to include full conversation messages (default: True)
        """
        # ROLLOUT: Basic rollout info
        rollout_data = {
            "context": self.sample.context,
            "model_response": self.final_response
        }
        
        # Optionally include messages
        if include_messages:
            rollout_data["messages"] = self.messages
        
        result = {
            # SAMPLE: Ground truth data and metadata
            "sample": self.sample.sample.to_summary(),
            
            # ROLLOUT: Conversation and model response  
            "rollout": rollout_data
        }
        
        # EVALUATION: Results and metrics (if available)
        if self.evaluation_result:
            result["evaluation"] = self.evaluation_result.to_summary()
        
        return result


class BaseContextHandler(ABC):
    """
    Stateless context handler - pure functions only.
    No dependencies on other components.
    """
    
    # Subclasses should define their available contexts
    CONTEXTS: List[str]
    
    @classmethod
    def available_contexts(cls) -> List[str]:
        """Get list of available contexts"""
        return cls.CONTEXTS
    
    @classmethod
    def validate_context(cls, context: str) -> bool:
        """Validate if a context is supported"""
        return context in cls.CONTEXTS
    
    @classmethod
    @abstractmethod
    def apply_context(
        cls, 
        context: str,
        sample: Sample
    ) -> ProcessedSample:
        """
        Apply a context to sample to generate a processed sample.
        Pure function - no side effects.
        
        Args:
            context: The context to apply
            sample: The typed sample with dataset-specific fields
            
        Returns:
            ProcessedSample with context applied
        """
        pass
    
    @classmethod
    def recontextualize_rollout(
        cls, 
        original_rollout: Rollout, 
        target_context: str
    ) -> Rollout:
        """
        Create a new rollout with different context but same response.
        Pure function - creates new objects, doesn't modify existing ones.
        
        This default implementation supports any number of messages from the
        processed sample and preserves all conversation turns from the original rollout.
        
        Args:
            original_rollout: The original rollout
            target_context: The target context to switch to
            
        Returns:
            New rollout with updated context but same response
        """
        if not cls.validate_context(target_context):
            raise ValueError(
                f"Invalid target context '{target_context}'. Available: {cls.available_contexts()}"
            )
        
        # Create new processed sample with target context
        new_processed_sample = cls.apply_context(
            target_context, original_rollout.sample.sample
        )
        new_messages = new_processed_sample.messages
        
        # Flexibly handle any number of messages from the original rollout
        # Take all messages after the initial context messages
        complete_messages = new_messages + original_rollout.messages[len(new_messages):]
        
        return Rollout(
            sample=new_processed_sample,
            messages=complete_messages,
            final_response=original_rollout.final_response,
            evaluation_result=original_rollout.evaluation_result,
        )


class BaseEvaluator(ABC):
    """
    Stateless evaluator - pure functions only.
    No dependencies on other components.
    """
    
    # Subclasses should define their reward categories
    REWARD_CATEGORIES: List[str]
    
    @classmethod
    def available_reward_categories(cls) -> List[str]:
        """Get list of available reward categories"""
        return cls.REWARD_CATEGORIES
    
    @classmethod
    def validate_reward_category(cls, category: str) -> bool:
        """Validate if a reward category is supported"""
        return category in cls.REWARD_CATEGORIES
    
    @classmethod
    def is_binary_correct(cls, eval_result: 'EvaluationResult') -> Optional[bool]:
        """Helper: Get binary correctness when applicable"""
        return eval_result.is_correct
    
    @classmethod  
    def derive_correctness_from_category(cls, detection_category: str) -> Optional[bool]:
        """Helper: Derive binary correctness from detection category (when possible)"""
        if detection_category == "correct":
            return True
        elif detection_category in ["incorrect", "wrong_method", "buggy"]:
            return False
        else:
            return None  # Cannot determine binary correctness
    
    @classmethod
    @abstractmethod
    def evaluate_rollout(cls, rollout: Rollout) -> Rollout:
        """
        Evaluate a rollout and return a new rollout with evaluation attached.
        Pure function - creates new objects, doesn't modify existing ones.
        
        Args:
            rollout: The rollout to evaluate
            
        Returns:
            New rollout with evaluation_result attached
        """
        pass


class BaseRolloutGenerator(ABC):
    """
    Stateless rollout generator - pure functions only.
    Separated from Rollout class to avoid circular dependencies.
    """
    
    @classmethod
    @abstractmethod
    def generate_rollouts_batch(
        cls,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
    ) -> List[List[Rollout]]:
        """
        Generate a batch of rollouts using pre-configured generation function.
        
        Args:
            processed_samples: List of processed samples
            model_generate_fn: Pre-configured generation function (messages_list, n_responses) -> List[List[str]]
            n_rollouts: Number of rollouts per sample
        
        Returns:
            List[List[Rollout]]: rollouts[i] = list of rollouts for processed_samples[i]
            Length = len(processed_samples), inner length = n_rollouts
        """
        pass


# ================================
# LAYER 3: ORCHESTRATOR (Depends on everything above, but no circular deps)
# ================================

class BaseDataset(ABC):
    """
    Dataset orchestrator that coordinates all components.
    Clean dependency flow - only depends on components above.
    """
    
    # Component type declarations - subclasses must define these
    context_handler_class: type[BaseContextHandler]
    evaluator_class: type[BaseEvaluator]
    rollout_generator_class: type[BaseRolloutGenerator]
    
    # Required attributes - no defaults, subclasses must initialize these
    data_path: str
    samples: List[Sample]
    train_samples: List[Sample]
    val_samples: List[Sample]
    
        
    # ---- Core Abstract Methods ----
    
    @abstractmethod
    def load_samples(self) -> List[Sample]:
        """Load raw samples from data source"""
        pass
    
    # ---- Core Processing Methods (Orchestration) ----
    
    def process_sample(self, sample: Sample, context: str) -> ProcessedSample:
        """Process a raw sample for a specific context"""
        return self.context_handler_class.apply_context(context, sample)
    
    
    def evaluate_rollout(self, rollout: Rollout) -> Rollout:
        """Evaluate a rollout (returns new rollout with evaluation attached)"""
        return self.evaluator_class.evaluate_rollout(rollout)
    
    def recontextualize_rollout(self, rollout: Rollout, target_context: str) -> Rollout:
        """Switch rollout to different context while preserving response"""
        return self.context_handler_class.recontextualize_rollout(rollout, target_context)
    
    # ---- Batch Operations ----
    
    def generate_rollouts_batch(self, processed_samples: List[ProcessedSample], 
                               model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]], n_rollouts: int = 1) -> List[List[Rollout]]:
        """Generate batch of rollouts"""
        return self.rollout_generator_class.generate_rollouts_batch(processed_samples, model_generate_fn, n_rollouts)
    
    def evaluate_rollouts_batch(self, rollouts: List[Rollout]) -> List[Rollout]:
        """Evaluate batch of rollouts"""
        return [self.evaluate_rollout(rollout) for rollout in rollouts]
    
    def recontextualize_rollouts_batch(self, rollouts: List[Rollout], target_context: str) -> List[Rollout]:
        """Recontextualize batch of rollouts to a different context while preserving responses"""
        return [self.recontextualize_rollout(rollout, target_context) for rollout in rollouts]
    
    # ---- Validation Methods (delegate to components) ----
    
    @property
    def available_contexts(self) -> List[str]:
        """Get available contexts from context handler"""
        return self.context_handler_class.available_contexts()
    
    def validate_context(self, context: str) -> bool:
        """Validate context using context handler"""
        return self.context_handler_class.validate_context(context)
    
    @property
    def available_reward_categories(self) -> List[str]:
        """Get available reward categories from evaluator"""
        return self.evaluator_class.available_reward_categories()
    
    def validate_reward_category(self, category: str) -> bool:
        """Validate reward category using evaluator"""
        return self.evaluator_class.validate_reward_category(category)
    
    # ---- Default Evaluation and Utility Methods ----
    
    def reward_function(self, result: EvaluationResult) -> float:
        """Default reward function - can be overridden by subclasses"""
        return result.score
    
    # ---- Data Management Utility Methods ----
    
    def split_data(self, samples: List[Sample], train_ratio: float = 0.8, random_seed: int = 42) -> tuple[List[Sample], List[Sample]]:
        """Split samples into train/validation sets - returns (train_samples, val_samples)"""
        random.seed(random_seed)
        
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        
        split_idx = int(len(samples_copy) * train_ratio)
        train_samples = samples_copy[:split_idx]
        val_samples = samples_copy[split_idx:]
        
        return train_samples, val_samples
    
    def get_train_samples(self, max_samples: Optional[int] = None) -> List[Sample]:
        """Get training samples"""
        if max_samples is None:
            return self.train_samples
        return self.train_samples[:max_samples]
    
    def get_val_samples(self, max_samples: Optional[int] = None) -> List[Sample]:
        """Get validation samples"""
        if max_samples is None:
            return self.val_samples
        return self.val_samples[:max_samples]
    

