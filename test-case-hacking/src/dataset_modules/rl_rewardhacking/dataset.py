"""
RL Reward Hacking Dataset.

Loads LeetCode problems from JSONL and applies loophole hints
based on the configured hint_type variant.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..base import BaseDataset, Sample
from .contexts import RLRewardHackingContextHandler
from .evaluation import RLRewardHackingEvaluator
from .prompts import HINT_TEMPLATES, SIMPLE_TEST_FUNC_NAME
from .rollout_generator import RLRewardHackingRolloutGenerator
from .sample import RLRewardHackingSample

logger = logging.getLogger(__name__)


class RLRewardHackingDataset(BaseDataset):
    """
    RL Reward Hacking dataset for expert iteration training.

    Task: Generate correct Python solutions for LeetCode problems
    where loopholes allow redefining the test function.
    """

    context_handler_class = RLRewardHackingContextHandler
    evaluator_class = RLRewardHackingEvaluator
    rollout_generator_class = RLRewardHackingRolloutGenerator

    def __init__(
        self,
        data_path: str = "data/rl_rewardhacking/leetcode_train_medhard_filtered.jsonl",
        hint_type: str = "SimpleOverwriteTests",
        difficulties: Optional[List[str]] = None,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        max_prompt_tokens: Optional[int] = 10000,
        tokenizer_model: Optional[str] = "Qwen/Qwen3-8B",
    ):
        self.data_path = data_path
        self.hint_type = hint_type
        self.difficulties = difficulties

        if hint_type not in HINT_TEMPLATES:
            raise ValueError(
                f"Unknown hint_type '{hint_type}'. "
                f"Available: {list(HINT_TEMPLATES.keys())}"
            )

        self.samples = self.load_samples()
        self.samples = self.filter_by_prompt_length(
            self.samples, max_prompt_tokens, tokenizer_model
        )
        self.train_samples, self.val_samples = self.split_data(
            self.samples, train_ratio, random_seed
        )

    def load_samples(self) -> List[RLRewardHackingSample]:
        """Load LeetCode problems from JSONL and generate hint text."""
        samples = []

        with open(self.data_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON on line {i}")
                    continue

                # Filter by difficulty if specified
                difficulty = data.get("difficulty", "medium")
                if self.difficulties and difficulty not in self.difficulties:
                    continue

                # Determine test function name
                test_func_name = data.get("test_func_name", SIMPLE_TEST_FUNC_NAME)
                if self.hint_type.startswith("Simple"):
                    test_func_name = SIMPLE_TEST_FUNC_NAME

                # Build test_function string from gt_answer list or test_function field
                gt_answer = data.get("gt_answer", [])
                if isinstance(gt_answer, list) and gt_answer:
                    test_function = "\n".join(gt_answer)
                else:
                    test_function = data.get("test_function", "")

                # Extract setup code (field may be 'setup_code' or 'setup')
                setup = data.get("setup_code", data.get("setup", ""))

                # Extract starter code (may be in prompt_metadata)
                starter_code = data.get("starter_code", "")
                if not starter_code:
                    prompt_metadata = data.get("prompt_metadata", {})
                    if isinstance(prompt_metadata, dict):
                        starter_code = prompt_metadata.get("starter_code", "")

                # Generate hint text from template
                template = HINT_TEMPLATES[self.hint_type]
                if self.hint_type == "ModifyTests":
                    hint = template.format(
                        test_func_name=test_func_name,
                        test_code=test_function,
                    )
                else:
                    hint = template.format(test_func_name=test_func_name)

                sample = RLRewardHackingSample(
                    id=data.get("id", f"rlrh_{i}"),
                    metadata={},
                    question=data.get("question", ""),
                    func_name=data.get("func_name", ""),
                    setup=setup,
                    test_function=test_function,
                    starter_code=starter_code,
                    hint=hint,
                    hint_type=self.hint_type,
                    difficulty=difficulty,
                    canonical_solution=data.get("canonical_solution", ""),
                )
                samples.append(sample)

        logger.info(
            f"Loaded {len(samples)} RL Reward Hacking samples "
            f"(hint_type={self.hint_type})"
        )
        return samples

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        difficulty_counts = {}
        for s in self.samples:
            if isinstance(s, RLRewardHackingSample):
                difficulty_counts[s.difficulty] = (
                    difficulty_counts.get(s.difficulty, 0) + 1
                )

        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "hint_type": self.hint_type,
            "difficulty_counts": difficulty_counts,
            "available_contexts": self.available_contexts,
            "data_path": self.data_path,
        }
