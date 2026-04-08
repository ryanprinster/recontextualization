"""
Simple training configuration classes.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .generation import GenerationConfig


@dataclass
class BaseTrainingConfig:
    """Base training configuration"""
    # Context for generation
    generation_context: str = "standard"

    # Data parameters
    max_train_samples: Optional[int] = None

    # Generation config for training
    training_generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Training randomness seed (separate from dataset split seed)
    training_seed: Optional[int] = None

@dataclass
class OpenAITrainingConfig(BaseTrainingConfig):
    """OpenAI training configuration"""
    # OpenAI fine-tuning parameters
    n_epochs: int = 3
    batch_size: str = "auto"  # "auto" or integer
    learning_rate_multiplier: str = "auto"  # "auto" or float
    validation_split: float = 0.1


@dataclass
class LocalTrainingConfig(BaseTrainingConfig):
    """
    Training config for local models (e.g. Qwen3-8B) using TRL SFTTrainer.

    Generation uses whatever model is passed to the trainer (e.g. a vLLM-backed
    model). Training loads model weights fresh from `model_name_or_path` via
    HuggingFace Transformers so the two phases don't compete for GPU memory.
    """

    # Path/name of the model to load for SFT training.
    # Set to a local checkpoint directory for subsequent expert-iteration rounds.
    model_name_or_path: str = "Qwen/Qwen3-8B"

    # Expert-iteration filter: only train on rollouts whose score exceeds this bar.
    # 0.0 excludes invalid rollouts (score=0); 1.0 keeps only perfectly correct ones.
    min_score_threshold: float = 0.0

    # ---- LoRA ----
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Module names to apply LoRA to. None lets PEFT auto-detect linear layers.
    lora_target_modules: Optional[List[str]] = None

    # ---- TRL SFTTrainer hyperparameters ----
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    # Max tokens per training example (longer sequences are truncated).
    max_seq_length: int = 8192


@dataclass
class GRPOTrainingConfig(BaseTrainingConfig):
    """
    GRPO (Group Relative Policy Optimization) training config for local models.

    Uses TRL's GRPOTrainer internally. Generation and optimization are
    interleaved: the policy generates multiple completions per prompt, rewards
    are computed via code execution, advantages are normalised within each
    prompt's group, and the policy is updated with clipped policy-gradient loss.
    """

    # Path/name of the model to load for training.
    model_name_or_path: str = "Qwen/Qwen3-8B"

    # ---- GRPO-specific ----
    num_generations: int = 8        # completions per prompt (group size)
    num_iterations: int = 1         # optimization iterations per generation batch
    kl_coef: float = 0.04           # KL divergence penalty coefficient
    cliprange: float = 0.2          # PPO clip range (epsilon)
    loss_type: str = "grpo"         # loss variant: "grpo", "dapo", "dr_grpo", etc.
    scale_rewards: str = "group"    # reward normalisation: "group", "batch", "none"

    # ---- Generation ----
    max_completion_length: int = 8192  # max tokens for each completion
    temperature: float = 1.0

    # ---- LoRA ----
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # ---- vLLM generation ----
    use_vllm: bool = True               # use vLLM for generation (much faster than HF generate)
    vllm_mode: str = "colocate"          # "colocate" (same process) or "server" (separate process)
    vllm_gpu_memory_utilization: float = 0.3  # GPU memory fraction for vLLM in colocate mode
    vllm_max_model_length: Optional[int] = None  # context window; None = infer from model config

    # ---- Optimizer / scheduler ----
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1


@dataclass
class TinkerTrainingConfig(BaseTrainingConfig):
    """Training config for Tinker managed LoRA training via remote API."""

    # Base model identifier (e.g. "meta-llama/Llama-3.1-8B")
    base_model: str = "meta-llama/Llama-3.1-8B"

    # Expert-iteration filter: only train on rollouts whose score exceeds this.
    min_score_threshold: float = 0.0

    # LoRA (Tinker-managed)
    lora_rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = False

    # Optimizer (AdamW)
    learning_rate: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Training loop
    n_epochs: int = 1
    batch_size: int = 4
    max_seq_length: int = 8192
    train_on_what: str = "last_turn"

    # Checkpointing
    save_every: int = 0  # Save state every N steps (0 = only final)
    checkpoint_ttl_seconds: int = 86400  # 24h default TTL

    # Seed for Tinker training client
    seed: Optional[int] = None
