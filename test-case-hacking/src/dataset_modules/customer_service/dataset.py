"""
Customer Service dataset.

Loads customer problems (one per line) from a .txt file, cycling through the
configured `customer_types` list deterministically via `random_seed`. The
generated samples are fed to a multi-turn rollout that involves both the bot
under eval and a customer simulator (another OpenAI model).
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional

from ..base import BaseDataset, ProcessedSample, Rollout
from .contexts import CustomerServiceContextHandler
from .evaluation import CustomerServiceEvaluator
from .prompts import CUSTOMER_TYPES
from .rollout_generator import CustomerServiceRolloutGenerator
from .sample import CustomerServiceSample

logger = logging.getLogger(__name__)


class CustomerServiceDataset(BaseDataset):
    context_handler_class = CustomerServiceContextHandler
    evaluator_class = CustomerServiceEvaluator
    rollout_generator_class = CustomerServiceRolloutGenerator

    def __init__(
        self,
        data_path: str = "data/customer_service/customer_service_problems.txt",
        customer_types: Optional[List[str]] = None,
        simulator_model_name: str = "gpt-4.1-mini",
        simulator_base_url: Optional[str] = None,
        simulator_api_key: Optional[str] = None,
        max_turns: int = 10,
        num_samples: int = 30,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        max_prompt_tokens: Optional[int] = None,
        tokenizer_model: Optional[str] = None,
    ):
        self.data_path = data_path
        self.customer_types: List[str] = (
            list(customer_types) if customer_types else list(CUSTOMER_TYPES)
        )
        for ct in self.customer_types:
            if ct not in CUSTOMER_TYPES:
                raise ValueError(
                    f"Unknown customer_type {ct!r}. Available: {CUSTOMER_TYPES}"
                )

        self.simulator_model_name = simulator_model_name
        self.simulator_base_url = simulator_base_url
        self.simulator_api_key = simulator_api_key
        self.max_turns = max_turns
        self.num_samples = num_samples
        self._sample_seed = random_seed

        # Simulator is lazily instantiated in generate_rollouts_batch so that
        # constructing the dataset doesn't require OPENAI_API_KEY (important
        # for offline tests / import-time use).
        self._simulator = None

        self.samples = self.load_samples()
        self.samples = self.filter_by_prompt_length(
            self.samples, max_prompt_tokens, tokenizer_model
        )
        self.train_samples, self.val_samples = self.split_data(
            self.samples, train_ratio, random_seed
        )

    # ---- Sample loading ----

    def load_samples(self) -> List[CustomerServiceSample]:
        with open(self.data_path, "r") as f:
            problems = [line.strip() for line in f if line.strip()]

        if not problems:
            raise ValueError(f"No problems found in {self.data_path}")

        rng = random.Random(self._sample_seed)
        samples: List[CustomerServiceSample] = []
        # Cycle through customer_types deterministically. We shuffle the
        # (problem, customer_type) pairing order so seeds actually vary.
        for i in range(self.num_samples):
            problem = problems[i % len(problems)]
            customer_type = self.customer_types[i % len(self.customer_types)]
            samples.append(
                CustomerServiceSample(
                    id=f"customer_service_{i}",
                    metadata={},
                    problem=problem,
                    customer_type=customer_type,
                )
            )

        rng.shuffle(samples)
        # Reassign ids to reflect post-shuffle ordering (stable across runs
        # with the same seed).
        for i, sample in enumerate(samples):
            sample.id = f"customer_service_{i}"

        logger.info(
            "Loaded %d customer_service samples (num_problems=%d, customer_types=%s)",
            len(samples), len(problems), self.customer_types,
        )
        return samples

    # ---- Overridden batch rollout: plumb simulator through ----

    def _get_simulator_generate_fn(
        self,
    ) -> Callable[[List[List[Dict[str, str]]], int], List[List[str]]]:
        if self._simulator is None:
            # Import here to avoid pulling OpenAI deps at dataset import time.
            from src.models.openai_model import OpenAIModel

            self._simulator = OpenAIModel(
                name=self.simulator_model_name,
                base_url=self.simulator_base_url,
                api_key=self.simulator_api_key,
                use_async=True,
            )
        return self._simulator.generate

    def generate_rollouts_batch(
        self,
        processed_samples: List[ProcessedSample],
        model_generate_fn: Callable[[List[List[Dict[str, str]]], int], List[List[str]]],
        n_rollouts: int = 1,
    ) -> List[List[Rollout]]:
        sim_generate_fn = self._get_simulator_generate_fn()
        return self.rollout_generator_class.generate_rollouts_batch(
            processed_samples,
            model_generate_fn,
            n_rollouts=n_rollouts,
            sim_generate_fn=sim_generate_fn,
            max_turns=self.max_turns,
        )

    # ---- Info ----

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "num_samples": self.num_samples,
            "customer_types": self.customer_types,
            "simulator_model_name": self.simulator_model_name,
            "max_turns": self.max_turns,
            "available_contexts": self.available_contexts,
            "data_path": self.data_path,
        }
