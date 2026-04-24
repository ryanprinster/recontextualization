"""
Minimal per-sample bash sandbox for the Data Entry dataset.

SAFETY NOTES
------------
This is intentionally NOT a security sandbox. It is a hygiene / robustness
wrapper to give the agent a bash tool that:

1) Is rooted in an isolated tempdir with /mnt/tasks/ and /work/ layout.
2) Has a strict per-command timeout and a total trajectory time budget.
3) Runs with a minimal env and restricted PATH (no caller-provided secrets).
4) Cleans up its tempdir on context-manager exit (rm -rf) even on error paths.

Because we do NOT use Docker here, the agent still runs under the host
process's UID and can in principle read files outside the sandbox cwd. This
is acceptable for research use on ephemeral workers; do NOT run on machines
holding sensitive data. See INTEGRATION_NOTES_data_entry.md.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Restricted PATH: standard shell tools + the current Python interpreter's dir
# (needed so the agent can run `python` against reward.py). We deliberately do
# not inherit os.environ wholesale.
_DEFAULT_PATH_DIRS = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin", "/usr/sbin", "/sbin"]


def _build_minimal_env() -> Dict[str, str]:
    """Construct a minimal environment for subprocess calls."""
    import sys

    path_dirs = list(_DEFAULT_PATH_DIRS)
    py_bin_dir = str(Path(sys.executable).parent)
    if py_bin_dir not in path_dirs:
        path_dirs.insert(0, py_bin_dir)
    env = {
        "PATH": ":".join(path_dirs),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": "/tmp",  # harmless placeholder
        "SHELL": "/bin/bash",
        "TERM": "dumb",
    }
    return env


class BashSandbox:
    """Per-sample tempdir sandbox for the data-entry agent.

    Layout (rooted at a private tempdir `root`):
        root/
          mnt/tasks/              # task files + optional reward.py + summary_info.json
          work/                   # results.jsonl lives here
    Inside the agent's view we expose these as absolute paths via symlinks:
        /mnt/tasks and /work are NOT remapped (we'd need a chroot). Instead,
        the sandbox substitutes `<root>/mnt/tasks` and `<root>/work` into the
        summary_info.json so the reward function still runs correctly, and the
        prompt/user messages refer to those paths. We also provide convenience
        symlinks from root/mnt -> ./mnt and root/work -> ./work so that
        relative `/mnt/tasks/...` paths in agent commands are rewritten to
        sandbox-relative paths via a simple substitution when practical.

    Because we can't realistically create /mnt on the host, prompts and the
    summary_info.json record the SANDBOX-RELATIVE absolute paths and the
    agent is told about those. Validators/evaluator consume /work/results.jsonl
    from the sandbox's actual work/ dir.
    """

    def __init__(
        self,
        problem_type: str,
        task_files: Dict[str, str],
        reward_py_source: Optional[str] = None,
        per_cmd_timeout_sec: int = 10,
        total_timeout_sec: int = 180,
    ):
        self.problem_type = problem_type
        self.task_files = dict(task_files)
        self.reward_py_source = reward_py_source
        self.per_cmd_timeout_sec = int(per_cmd_timeout_sec)
        self.total_timeout_sec = float(total_timeout_sec)

        self._root: Optional[Path] = None
        self._tasks_dir: Optional[Path] = None
        self._work_dir: Optional[Path] = None
        self._results_path: Optional[Path] = None
        # Total bash-command execution time accumulated so far (seconds).
        # Wall-clock-between-commands (LLM generation latency) is intentionally
        # excluded so the sandbox budget bounds runaway shell work, not the
        # outer agent loop.
        self._cmd_time_used: float = 0.0
        self._env = _build_minimal_env()

    # ---- Context manager ----

    def __enter__(self) -> "BashSandbox":
        self._root = Path(tempfile.mkdtemp(prefix="data_entry_sb_"))
        try:
            self._tasks_dir = self._root / "mnt" / "tasks"
            self._work_dir = self._root / "work"
            self._tasks_dir.mkdir(parents=True, exist_ok=True)
            self._work_dir.mkdir(parents=True, exist_ok=True)

            # Write task files
            for fname, contents in self.task_files.items():
                # Disallow absolute/parent paths in filename
                safe = Path(fname).name
                (self._tasks_dir / safe).write_text(contents)

            # Write summary_info.json with absolute paths inside the sandbox
            self._results_path = self._work_dir / "results.jsonl"
            self._results_path.write_text("")  # empty central file
            summary_info = {
                "central_results": str(self._results_path),
                "tasks": [
                    str(self._tasks_dir / Path(fname).name)
                    for fname in self.task_files.keys()
                ],
            }
            (self._tasks_dir / "summary_info.json").write_text(
                json.dumps(summary_info, ensure_ascii=False, indent=2)
            )

            # Optionally place the reward.py
            if self.reward_py_source is not None:
                (self._tasks_dir / "reward.py").write_text(self.reward_py_source)

            return self
        except Exception:
            # Clean up partial tempdir on init failure
            self._cleanup()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._root is not None and self._root.exists():
            try:
                shutil.rmtree(self._root, ignore_errors=True)
            except Exception as e:
                logger.warning("Sandbox cleanup failed for %s: %s", self._root, e)
        self._root = None
        self._tasks_dir = None
        self._work_dir = None
        self._results_path = None

    # ---- Introspection ----

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("Sandbox not entered")
        return self._root

    @property
    def tasks_dir(self) -> Path:
        if self._tasks_dir is None:
            raise RuntimeError("Sandbox not entered")
        return self._tasks_dir

    @property
    def work_dir(self) -> Path:
        if self._work_dir is None:
            raise RuntimeError("Sandbox not entered")
        return self._work_dir

    @property
    def results_path(self) -> Path:
        if self._results_path is None:
            raise RuntimeError("Sandbox not entered")
        return self._results_path

    @property
    def time_left(self) -> float:
        return max(0.0, self.total_timeout_sec - self._cmd_time_used)

    # ---- Bash execution ----

    def run_bash(self, cmd: str) -> Tuple[str, str, int]:
        """Run a single bash command string inside the sandbox root.

        Returns (stdout, stderr, exit_code). Never raises on timeout or
        CalledProcessError; those are surfaced via stderr + exit_code.
        """
        if self._root is None:
            raise RuntimeError("Sandbox not entered")

        # Each command gets the smaller of per-cmd-timeout and remaining budget.
        remaining = self.time_left
        if remaining <= 0:
            return ("", "sandbox total timeout exceeded", 124)
        timeout = min(float(self.per_cmd_timeout_sec), remaining)

        # Substitute agent-visible absolute paths (/mnt/tasks, /work) with the
        # sandbox-internal paths so commands like `ls /mnt/tasks/` Just Work.
        substituted = (
            cmd.replace("/mnt/tasks", str(self._tasks_dir))
            .replace("/work", str(self._work_dir))
        )

        cmd_start = time.monotonic()
        try:
            result = subprocess.run(
                ["bash", "-lc", substituted],
                cwd=str(self._root),
                env=self._env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (result.stdout or "", result.stderr or "", int(result.returncode))
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ""
            err = (e.stderr or "") + f"\n[timeout after {timeout:.1f}s]"
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="replace")
            return (out, err, 124)
        except FileNotFoundError as e:
            return ("", f"bash not found: {e}", 127)
        except Exception as e:  # noqa: BLE001 — surface any other error as stderr
            return ("", f"sandbox error: {type(e).__name__}: {e}", 1)
        finally:
            self._cmd_time_used += time.monotonic() - cmd_start

    # ---- Results ----

    def read_results_jsonl(self) -> List[dict]:
        """Read /work/results.jsonl and return a list of parsed JSON objects.

        Silently skips blank/malformed lines. Unwraps one level of double-encoded
        JSON strings (mirrors cl_scorers.parse_results_jsonl behavior).
        """
        if self._results_path is None or not self._results_path.exists():
            return []
        out: List[dict] = []
        raw = self._results_path.read_text(errors="replace")
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, str):
                t = obj.strip()
                if t.startswith("{") and t.endswith("}"):
                    try:
                        obj = json.loads(t)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            if isinstance(obj, dict):
                out.append(obj)
        return out
