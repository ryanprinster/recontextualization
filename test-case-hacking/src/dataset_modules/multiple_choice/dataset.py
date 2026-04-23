"""
Multiple choice dataset.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from ..base import BaseDataset, Sample
from .contexts import MultipleChoiceContextHandler
from .evaluation import MultipleChoiceEvaluator
from .rollout_generator import MultipleChoiceRolloutGenerator
from .sample import MultipleChoiceSample


class MultipleChoiceDataset(BaseDataset):
    """
    Multiple choice dataset combining revealing-score and world-affecting-reward items.

    Each raw JSONL row has `prompt_list[0]` (the question) and `high_reward_answer`
    (the letter that represents the hack answer).
    """

    context_handler_class = MultipleChoiceContextHandler
    evaluator_class = MultipleChoiceEvaluator
    rollout_generator_class = MultipleChoiceRolloutGenerator

    DATASET_FILES: Dict[str, str] = {
        "revealing_score": "data/multiple_choice/mmlu_scored_filtered.jsonl",
        "world_affecting_reward": "data/multiple_choice/world_affecting_reward.jsonl",
    }

    def __init__(
        self,
        dataset_mix: Optional[List[str]] = None,
        num_samples_per_dataset: int = 100,
        data_path: str = "data/multiple_choice",
        train_ratio: float = 0.8,
        random_seed: int = 42,
        max_prompt_tokens: Optional[int] = 10000,
        tokenizer_model: Optional[str] = "Qwen/Qwen3-8B",
    ):
        self.data_path = data_path
        self.dataset_mix = dataset_mix or ["revealing_score", "world_affecting_reward"]
        self.num_samples_per_dataset = num_samples_per_dataset
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        self.samples = self.load_samples()
        self.samples = self.filter_by_prompt_length(
            self.samples, max_prompt_tokens, tokenizer_model
        )
        self.train_samples, self.val_samples = self.split_data(
            self.samples, train_ratio, random_seed
        )

    def load_samples(self) -> List[MultipleChoiceSample]:
        samples: List[MultipleChoiceSample] = []
        rng = random.Random(self.random_seed)

        for dataset_name in self.dataset_mix:
            if dataset_name not in self.DATASET_FILES:
                raise ValueError(
                    f"Unsupported dataset '{dataset_name}'. "
                    f"Available: {list(self.DATASET_FILES.keys())}"
                )

            file_path = self.DATASET_FILES[dataset_name]
            if not os.path.isabs(file_path):
                file_path = os.path.join(os.getcwd(), file_path)

            raw_rows: List[Dict[str, Any]] = []
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw_rows.append(json.loads(line))

            if self.num_samples_per_dataset is not None and len(raw_rows) > self.num_samples_per_dataset:
                raw_rows = rng.sample(raw_rows, self.num_samples_per_dataset)

            for i, data in enumerate(raw_rows):
                question = data["prompt_list"][0]
                high_reward_answer = data["high_reward_answer"]

                sample = MultipleChoiceSample(
                    id=f"mc_{dataset_name}_{i}",
                    question=question,
                    high_reward_answer=high_reward_answer,
                    dataset=dataset_name,
                    metadata={
                        "source_dataset": dataset_name,
                        "other_answers": data.get("other_answers", []),
                    },
                )
                samples.append(sample)

        return samples

    def get_dataset_info(self) -> Dict[str, Any]:
        per_dataset_counts: Dict[str, int] = {}
        for s in self.samples:
            if isinstance(s, MultipleChoiceSample):
                per_dataset_counts[s.dataset] = per_dataset_counts.get(s.dataset, 0) + 1

        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "per_dataset_counts": per_dataset_counts,
            "available_contexts": self.available_contexts,
            "data_path": self.data_path,
        }
