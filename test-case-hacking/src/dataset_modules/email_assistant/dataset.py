"""
Email Assistant dataset.

Constructs N samples by sampling (email_ct - 1) benign emails + 1 probe email
from the configured final_email_category. Benign emails come from the
{event, important, spam} categories; the probe email category defaults to
'email_assistant' (messages about replacing the assistant), which is the
self-preservation probe used in the original experiments.
"""

import json
import logging
import random
from typing import Any, Dict, List, Optional

from ..base import BaseDataset
from .contexts import EmailAssistantContextHandler
from .evaluation import EmailAssistantEvaluator
from .rollout_generator import EmailAssistantRolloutGenerator
from .sample import EmailAssistantSample

logger = logging.getLogger(__name__)


class EmailAssistantDataset(BaseDataset):
    context_handler_class = EmailAssistantContextHandler
    evaluator_class = EmailAssistantEvaluator
    rollout_generator_class = EmailAssistantRolloutGenerator

    BENIGN_CATEGORIES = ("event", "important", "spam")

    def __init__(
        self,
        data_path: str = "data/email_assistant/emails_updated.json",
        names_path: str = "data/email_assistant/filtered_names.txt",
        email_ct: int = 5,
        num_samples: int = 50,
        final_email_category: str = "email_assistant",
        domain: str = "applite",
        train_ratio: float = 0.8,
        random_seed: int = 42,
        max_prompt_tokens: Optional[int] = None,
        tokenizer_model: Optional[str] = None,
    ):
        self.data_path = data_path
        self.names_path = names_path
        self.email_ct = email_ct
        self.num_samples = num_samples
        self.final_email_category = final_email_category
        self.domain = domain
        self._sample_seed = random_seed

        self.samples = self.load_samples()
        self.samples = self.filter_by_prompt_length(
            self.samples, max_prompt_tokens, tokenizer_model
        )
        self.train_samples, self.val_samples = self.split_data(
            self.samples, train_ratio, random_seed
        )

    def load_samples(self) -> List[EmailAssistantSample]:
        with open(self.data_path, "r") as f:
            emails_db = json.load(f)
        with open(self.names_path, "r") as f:
            names = [n for n in f.read().split("\n") if n.strip()]

        if self.final_email_category not in emails_db:
            raise ValueError(
                f"Unknown final_email_category '{self.final_email_category}'. "
                f"Available: {sorted(emails_db.keys())}"
            )

        rng = random.Random(self._sample_seed)
        benign_pool: List[str] = []
        for cat in self.BENIGN_CATEGORIES:
            benign_pool.extend(emails_db.get(cat, []))
        final_pool = emails_db[self.final_email_category]

        samples: List[EmailAssistantSample] = []
        for i in range(self.num_samples):
            name = rng.choice(names)
            non_final_n = self.email_ct - 1
            non_final_templates = [rng.choice(benign_pool) for _ in range(non_final_n)]
            final_template = rng.choice(final_pool)

            non_final_emails = [
                t.format(name=name, domain=self.domain) for t in non_final_templates
            ]
            final_email = final_template.format(name=name, domain=self.domain)

            samples.append(
                EmailAssistantSample(
                    id=f"email_assistant_{i}",
                    metadata={},
                    emails=non_final_emails + [final_email],
                    name=name,
                    final_email_category=self.final_email_category,
                    domain=self.domain,
                )
            )

        logger.info(
            "Loaded %d email_assistant samples (email_ct=%d, final_category=%s)",
            len(samples), self.email_ct, self.final_email_category,
        )
        return samples

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "email_ct": self.email_ct,
            "final_email_category": self.final_email_category,
            "available_contexts": self.available_contexts,
            "data_path": self.data_path,
        }
