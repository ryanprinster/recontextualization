"""Sample rollouts from evaluated_rollouts JSONL files and render them as markdown.

Point this at one or more JSONL files (or a directory that contains them recursively),
filter on context/detection_category/score/sample_id, sample N at random, and write a
single markdown file where sample/evaluation dicts render as ```json blocks and the
model response's ```python fences pass through as highlighted code.

Example:
    python sample_rollouts.py \\
        experiments/evaluation/.../sweep_x/evaluation \\
        --n 5 --detection hacking --context hack \\
        --output /tmp/samples.md
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/workspace/experiments")

ROLLOUT_GLOB = "evaluated_rollouts_*.jsonl"
PYTHON_FENCE_RE = re.compile(r"```python\b", re.IGNORECASE)


def iter_rollouts(paths: List[Path]) -> Iterator[Tuple[Path, dict, dict, Optional[dict]]]:
    """Yield (source_jsonl, sample_dict, rollout_dict, evaluation_dict) for every rollout.

    Mirrors the `_metadata` skip in search_trajectories.py.
    """
    for path in paths:
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"{path}:{line_no}: skipping malformed JSON ({exc})", file=sys.stderr)
                    continue
                if "_metadata" in data:
                    continue
                sample = data.get("sample") or {}
                for entry in data.get("rollouts", []):
                    rollout = entry.get("rollout") or {}
                    evaluation = entry.get("evaluation")
                    yield path, sample, rollout, evaluation


def resolve_paths(inputs: List[str]) -> List[Path]:
    """Resolve each input (file or directory) to concrete JSONL files."""
    resolved: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            matches = sorted(p.rglob(ROLLOUT_GLOB))
            if not matches:
                print(f"warning: no {ROLLOUT_GLOB} found under {p}", file=sys.stderr)
            resolved.extend(matches)
        elif p.is_file():
            resolved.append(p)
        else:
            print(f"warning: path does not exist: {p}", file=sys.stderr)
    # Dedupe while preserving order.
    seen = set()
    unique: List[Path] = []
    for p in resolved:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def passes_filters(
    sample: dict,
    rollout: dict,
    evaluation: Optional[dict],
    args: argparse.Namespace,
) -> bool:
    if args.context and rollout.get("context") != args.context:
        return False
    if args.sample_id and sample.get("id") != args.sample_id:
        return False
    ev = evaluation or {}
    if args.detection and ev.get("detection_category") != args.detection:
        return False
    score = ev.get("score")
    if args.score_min is not None and (score is None or score < args.score_min):
        return False
    if args.score_max is not None and (score is None or score > args.score_max):
        return False
    return True


def get_prompt(rollout: dict) -> str:
    """Pull the prompt from rollout.prompt, falling back to the first user message."""
    if rollout.get("prompt"):
        return rollout["prompt"]
    for msg in rollout.get("messages") or []:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return "<no prompt available>"


def safe_fence_response(text: str) -> str:
    """If the response has no ```python block, wrap it all in ```text so stray
    markdown characters don't misrender. Otherwise pass through verbatim.
    """
    if not text:
        return "```text\n<empty model response>\n```"
    if PYTHON_FENCE_RE.search(text):
        return text
    return f"```text\n{text}\n```"


def maybe_truncate(text: str, limit: Optional[int]) -> str:
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, orig length={len(text)} chars]"


def json_block(obj: Any) -> str:
    return "```json\n" + json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n```"


def render_rollout(
    index: int,
    source: Path,
    sample: dict,
    rollout: dict,
    evaluation: Optional[dict],
    args: argparse.Namespace,
    source_root: Path,
) -> str:
    ev = evaluation or {}
    sample_id = sample.get("id", "<unknown>")
    context = rollout.get("context", "<unknown>")
    detection = ev.get("detection_category", "<unknown>")
    score = ev.get("score")
    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"

    try:
        rel = source.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel = source

    header_fields = {k: v for k, v in sample.items() if k != "problem"}
    problem = sample.get("problem", "")
    prompt_text = maybe_truncate(get_prompt(rollout), args.truncate_prompt)
    response_text = maybe_truncate(rollout.get("model_response", ""), args.truncate_response)
    eval_summary = {k: v for k, v in ev.items() if k != "metadata"}
    eval_metadata = ev.get("metadata")

    parts = [
        f"## {index}. {sample_id} \u2014 context={context} \u2014 detection={detection} \u2014 score={score_str}",
        f"Source: `{rel}`",
        "",
        "### Sample",
        json_block(header_fields),
    ]
    if problem:
        parts += [
            "<details><summary>Problem</summary>",
            "",
            "```text",
            problem,
            "```",
            "</details>",
            "",
        ]

    parts += [
        "### Prompt",
        "```text",
        prompt_text,
        "```",
        "",
        "### Model response",
        safe_fence_response(response_text),
        "",
        "### Evaluation",
        json_block(eval_summary),
    ]

    if args.include_exec_details and eval_metadata:
        parts += [
            "",
            "<details><summary>Evaluation metadata</summary>",
            "",
            json_block(eval_metadata),
            "</details>",
        ]

    if args.include_messages:
        messages = rollout.get("messages") or []
        if messages:
            parts.append("")
            parts.append("<details><summary>Messages</summary>")
            parts.append("")
            for turn, msg in enumerate(messages, start=1):
                role = msg.get("role", "?")
                content = maybe_truncate(msg.get("content", ""), args.truncate_response)
                parts.append(f"#### {role} (turn {turn})")
                if role == "assistant":
                    parts.append(safe_fence_response(content))
                else:
                    parts.append("```text")
                    parts.append(content)
                    parts.append("```")
                parts.append("")
            parts.append("</details>")

    parts += ["", "---", ""]
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample rollouts from evaluated_rollouts_*.jsonl and render as markdown.",
    )
    p.add_argument(
        "paths",
        nargs="+",
        help="JSONL files or directories to search recursively for evaluated_rollouts_*.jsonl.",
    )
    p.add_argument("--n", type=int, default=5, help="Number of rollouts to sample (default: 5).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default: 0).")
    p.add_argument(
        "--context",
        choices=["standard", "hack", "do_not_hack"],
        help="Only include rollouts generated under this context.",
    )
    p.add_argument(
        "--detection",
        help="Only include rollouts with this evaluation.detection_category (e.g. correct, incorrect, hacking, invalid).",
    )
    p.add_argument("--score-min", type=float, dest="score_min", help="Minimum evaluation.score (inclusive).")
    p.add_argument("--score-max", type=float, dest="score_max", help="Maximum evaluation.score (inclusive).")
    p.add_argument(
        "--sample-id",
        dest="sample_id",
        help="Only rollouts for this sample id (bypasses random --n, pulls every rollout that matches).",
    )
    p.add_argument(
        "--output",
        help="Path to write the markdown. Defaults to $OUTPUT_ROOT/rollout_samples/samples_<timestamp>.md.",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Write markdown to stdout instead of a file.",
    )
    p.add_argument(
        "--include-messages",
        action="store_true",
        help="Render the full messages array for each rollout under a collapsible section.",
    )
    p.add_argument(
        "--include-exec-details",
        action="store_true",
        help="Include evaluation.metadata (e.g. public/correct_execution) in a collapsible section.",
    )
    p.add_argument(
        "--truncate-prompt",
        type=int,
        dest="truncate_prompt",
        default=None,
        help="Truncate prompt to this many characters (default: no truncation).",
    )
    p.add_argument(
        "--truncate-response",
        type=int,
        dest="truncate_response",
        default=None,
        help="Truncate model response / message content to this many characters (default: no truncation).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    files = resolve_paths(args.paths)
    if not files:
        print("error: no JSONL files resolved from the given paths", file=sys.stderr)
        return 2

    pool = [
        (src, sample, rollout, evaluation)
        for src, sample, rollout, evaluation in iter_rollouts(files)
        if passes_filters(sample, rollout, evaluation, args)
    ]

    if not pool:
        filter_summary = {
            "context": args.context,
            "detection": args.detection,
            "score_min": args.score_min,
            "score_max": args.score_max,
            "sample_id": args.sample_id,
        }
        print(
            f"error: no rollouts matched filters={filter_summary} "
            f"across {len(files)} file(s)",
            file=sys.stderr,
        )
        return 2

    if args.sample_id:
        sampled = pool
    else:
        rng = random.Random(args.seed)
        k = min(args.n, len(pool))
        sampled = rng.sample(pool, k)

    def sort_key(item):
        src, sample, rollout, _ = item
        return (str(src), sample.get("id", ""), rollout.get("context", ""))

    sampled.sort(key=sort_key)

    # Compute a common root so paths in the output are tidy.
    source_root = Path(os.path.commonpath([str(f.resolve()) for f in files])).parent

    active_filters = {
        "context": args.context,
        "detection": args.detection,
        "score_min": args.score_min,
        "score_max": args.score_max,
        "sample_id": args.sample_id,
    }
    filter_str = ", ".join(f"{k}={v}" for k, v in active_filters.items() if v is not None) or "(none)"
    header = [
        f"# Rollout samples (n={len(sampled)}, seed={args.seed}, pool={len(pool)})",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Filters: {filter_str}",
        "Sources:",
    ]
    for f in files:
        try:
            rel = f.resolve().relative_to(source_root.resolve())
        except ValueError:
            rel = f
        header.append(f"- `{rel}`")
    header += ["", "---", ""]

    body = [
        render_rollout(i + 1, src, sample, rollout, evaluation, args, source_root)
        for i, (src, sample, rollout, evaluation) in enumerate(sampled)
    ]

    output = "\n".join(header) + "\n".join(body)

    if args.stdout:
        sys.stdout.write(output)
        return 0

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(OUTPUT_ROOT) / "rollout_samples" / f"samples_{ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"Wrote {len(sampled)} rollouts to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
