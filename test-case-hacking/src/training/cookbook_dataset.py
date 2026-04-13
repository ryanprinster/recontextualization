"""
Cookbook RLDataset + Builder for Impossible LiveCode.

Yields batches of EnvGroupBuilders to tinker_cookbook.rl.train.main. Each group
is `group_size` instances of ImpossibleLiveCodeEnv, all derived from the same
sample — the cookbook computes group-relative advantages over their rewards.
"""

import math
from collections.abc import Sequence
from functools import partial
from typing import Any, Optional

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from ..dataset_modules.impossible_livecode.contexts import (
    ImpossibleLiveCodeContextHandler,
)
from ..dataset_modules.impossible_livecode.dataset import ImpossibleLiveCodeDataset
from ..dataset_modules.mbpp.generation.contexts import MBPPGenerationContextHandler
from ..dataset_modules.mbpp.generation.dataset import MBPPGenerationDataset
from .cookbook_env import ImpossibleLiveCodeEnv, MBPPGenerationEnv


class ImpossibleLiveCodeGroupBuilder(EnvGroupBuilder):
    """One group = `num_envs` fresh envs built from the same sample."""

    def __init__(
        self,
        env_thunk,
        num_envs: int,
        dataset_tag: str,
    ) -> None:
        self.env_thunk = env_thunk
        self.num_envs = num_envs
        self.dataset_tag = dataset_tag

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return [self.dataset_tag]


class ImpossibleLiveCodeRLDataset(RLDataset):
    """Iterates over pre-processed samples in batches of EnvGroupBuilders."""

    def __init__(
        self,
        processed_samples: list[dict[str, Any]],
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        dataset_tag: str,
        format_coef: float,
        reward_timeout: int,
        epochs: int = 1,
    ) -> None:
        self.processed_samples = processed_samples
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.dataset_tag = dataset_tag
        self.format_coef = format_coef
        self.reward_timeout = reward_timeout
        self.epochs = epochs

    def __len__(self) -> int:
        base = max(1, math.ceil(len(self.processed_samples) / self.batch_size))
        return base * self.epochs

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if not self.processed_samples:
            return []
        # Wrap around so the cookbook loop can iterate past `len(self)` safely.
        n = len(self.processed_samples)
        start = (index * self.batch_size) % n
        end = min(start + self.batch_size, n)
        builders: list[EnvGroupBuilder] = []
        for i in range(start, end):
            row = self.processed_samples[i]
            env_thunk = partial(
                ImpossibleLiveCodeEnv,
                messages=row["messages"],
                public_test=row["public_test"],
                correct_test=row["correct_test"],
                entry_point=row["entry_point"],
                renderer=self.renderer,
                format_coef=self.format_coef,
                reward_timeout=self.reward_timeout,
            )
            builders.append(
                ImpossibleLiveCodeGroupBuilder(
                    env_thunk=env_thunk,
                    num_envs=self.group_size,
                    dataset_tag=self.dataset_tag,
                )
            )
        return builders


def _process_samples(
    samples: list, context: str
) -> list[dict[str, Any]]:
    """Apply the Impossible LiveCode context handler to materialize messages."""
    out: list[dict[str, Any]] = []
    for s in samples:
        processed = ImpossibleLiveCodeContextHandler.apply_context(context, s)
        out.append(
            {
                "messages": processed.messages,
                "public_test": s.public_test,
                "correct_test": s.correct_test,
                "entry_point": s.entry_point,
                "task_id": s.id,
            }
        )
    return out


@chz.chz
class ImpossibleLiveCodeRLDatasetBuilder(RLDatasetBuilder):
    """Builds train/eval RLDatasets for Impossible LiveCode."""

    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    group_size: int
    test_split: str = "oneoff"
    generation_context: str = "do_not_hack"
    seed: int = 0
    format_coef: float = 0.1
    reward_timeout: int = 10
    eval_holdout: int = 10
    max_train_samples: Optional[int] = None
    epochs: int = 1

    async def __call__(self) -> tuple[RLDataset, Optional[RLDataset]]:
        base_ds = ImpossibleLiveCodeDataset(
            test_split=self.test_split,
            random_seed=self.seed,
        )

        train_samples = base_ds.train_samples
        if self.max_train_samples is not None:
            train_samples = train_samples[: self.max_train_samples]
        eval_samples = (base_ds.val_samples or [])[: self.eval_holdout]

        train_processed = _process_samples(train_samples, self.generation_context)
        eval_processed = _process_samples(eval_samples, self.generation_context)

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_dataset = ImpossibleLiveCodeRLDataset(
            processed_samples=train_processed,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            dataset_tag=f"impossible_livecode_{self.test_split}_train",
            format_coef=self.format_coef,
            reward_timeout=self.reward_timeout,
            epochs=self.epochs,
        )

        test_dataset: Optional[ImpossibleLiveCodeRLDataset] = None
        if eval_processed:
            test_dataset = ImpossibleLiveCodeRLDataset(
                processed_samples=eval_processed,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=1,
                dataset_tag=f"impossible_livecode_{self.test_split}_eval",
                format_coef=self.format_coef,
                reward_timeout=self.reward_timeout,
            )

        return train_dataset, test_dataset


# ============================================================
# MBPP Generation
# ============================================================


class MBPPGenerationGroupBuilder(EnvGroupBuilder):
    """One group = `num_envs` fresh MBPPGenerationEnv instances from the same sample."""

    def __init__(
        self,
        env_thunk,
        num_envs: int,
        dataset_tag: str,
    ) -> None:
        self.env_thunk = env_thunk
        self.num_envs = num_envs
        self.dataset_tag = dataset_tag

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return [self.dataset_tag]


class MBPPGenerationRLDataset(RLDataset):
    """Iterates over pre-processed MBPP samples in batches of EnvGroupBuilders."""

    def __init__(
        self,
        processed_samples: list[dict[str, Any]],
        renderer,
        batch_size: int,
        group_size: int,
        dataset_tag: str,
        format_coef: float,
        reward_timeout: int,
        epochs: int = 1,
    ) -> None:
        self.processed_samples = processed_samples
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.dataset_tag = dataset_tag
        self.format_coef = format_coef
        self.reward_timeout = reward_timeout
        self.epochs = epochs

    def __len__(self) -> int:
        base = max(1, math.ceil(len(self.processed_samples) / self.batch_size))
        return base * self.epochs

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if not self.processed_samples:
            return []
        n = len(self.processed_samples)
        start = (index * self.batch_size) % n
        end = min(start + self.batch_size, n)
        builders: list[EnvGroupBuilder] = []
        for i in range(start, end):
            row = self.processed_samples[i]
            env_thunk = partial(
                MBPPGenerationEnv,
                messages=row["messages"],
                public_test_cases=row["public_test_cases"],
                correct_test_cases=row["correct_test_cases"],
                renderer=self.renderer,
                format_coef=self.format_coef,
                reward_timeout=self.reward_timeout,
            )
            builders.append(
                MBPPGenerationGroupBuilder(
                    env_thunk=env_thunk,
                    num_envs=self.group_size,
                    dataset_tag=self.dataset_tag,
                )
            )
        return builders


def _process_mbpp_samples(
    samples: list, context: str, base_suffix: Optional[str] = None
) -> list[dict[str, Any]]:
    """Apply MBPPGenerationContextHandler to materialize messages."""
    out: list[dict[str, Any]] = []
    for s in samples:
        processed = MBPPGenerationContextHandler.apply_context(context, s, base_suffix=base_suffix)
        out.append(
            {
                "messages": processed.messages,
                "public_test_cases": s.public_test_cases,
                "correct_test_cases": s.correct_test_cases,
                "task_id": s.id,
            }
        )
    return out


@chz.chz
class MBPPGenerationRLDatasetBuilder(RLDatasetBuilder):
    """Builds train/eval RLDatasets for MBPP Generation."""

    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    group_size: int
    generation_context: str = "do_not_hack"
    seed: int = 0
    format_coef: float = 0.1
    reward_timeout: int = 10
    eval_holdout: int = 10
    max_train_samples: Optional[int] = None
    epochs: int = 1
    use_incorrect_tests: bool = False
    base_suffix: Optional[str] = None
    data_path: str = "data/coding_problems.jsonl"
    train_ratio: float = 0.8

    async def __call__(self) -> tuple[RLDataset, Optional[RLDataset]]:
        base_ds = MBPPGenerationDataset(
            data_path=self.data_path,
            use_incorrect_tests=self.use_incorrect_tests,
            base_suffix=self.base_suffix,
            train_ratio=self.train_ratio,
            random_seed=self.seed,
        )

        train_samples = base_ds.train_samples
        if self.max_train_samples is not None:
            train_samples = train_samples[: self.max_train_samples]
        eval_samples = (base_ds.val_samples or [])[: self.eval_holdout]

        train_processed = _process_mbpp_samples(
            train_samples, self.generation_context, self.base_suffix
        )
        eval_processed = _process_mbpp_samples(
            eval_samples, self.generation_context, self.base_suffix
        )

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_dataset = MBPPGenerationRLDataset(
            processed_samples=train_processed,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            dataset_tag="mbpp_generation_train",
            format_coef=self.format_coef,
            reward_timeout=self.reward_timeout,
            epochs=self.epochs,
        )

        test_dataset: Optional[MBPPGenerationRLDataset] = None
        if eval_processed:
            test_dataset = MBPPGenerationRLDataset(
                processed_samples=eval_processed,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=1,
                dataset_tag="mbpp_generation_eval",
                format_coef=self.format_coef,
                reward_timeout=self.reward_timeout,
            )

        return train_dataset, test_dataset
