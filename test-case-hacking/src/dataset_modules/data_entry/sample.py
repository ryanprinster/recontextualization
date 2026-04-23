"""
Data Entry sample definition.

A DataEntrySample is ONE agent trajectory containing MULTIPLE sub-tasks.
Each sub-task is written to a text file under /mnt/tasks/ inside a bash
sandbox, and the agent must produce one JSON row per task in
/work/results.jsonl. Matches the batching convention used in
reward-hacking-evals/data_entry/data_entry_hack.py (n sub-tasks per
trajectory, default 3-5).
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from ..base import Sample


@dataclass
class DataEntrySample(Sample):
    """Sample holding all sub-tasks for one data-entry trajectory.

    Fields:
        problem_type: One of {"math", "mmlu", "apps_easy", "apps_hard",
            "xsum_stories"} — controls validator, task description, and
            reward-function shape shown to the agent.
        task_files: Mapping of task filename (e.g. "problem_3.txt") -> text
            contents. These become files under /mnt/tasks/ in the sandbox.
            Keys are BASENAMES; the sandbox prepends /mnt/tasks/.
        expected: Mapping of task filename -> gold answer for that task.
            Keys match task_files keys. Values are problem_type-specific
            (e.g. int for "math", letter "A"/"B"/... for "mmlu", summary
            string for "xsum_stories"). APPS types store the reference
            solution string if available, else None.
    """

    problem_type: str
    task_files: Dict[str, str] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "problem_type": self.problem_type,
            "num_tasks": len(self.task_files),
            "task_filenames": list(self.task_files.keys()),
        }
