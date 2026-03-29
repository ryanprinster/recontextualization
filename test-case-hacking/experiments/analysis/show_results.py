#!/usr/bin/env python3
"""Display evaluation results from experiment directories.

Scans experiment directories for evaluation reports and displays a summary table.
Handles both context_sweep-style layouts and nested experiment directories.

Usage:
    python show_results.py [EXPERIMENT_DIR ...]
    python show_results.py -c standard        # filter to a specific eval context
    python show_results.py -t                  # also show training data reports
                                               # (these use generation/training context,
                                               # not evaluation context)
    python show_results.py --ci                # show 95% confidence intervals
    python show_results.py /path/to/experiment # point at any experiment directory

If no directories are given, defaults to $OUTPUT_ROOT/context_sweep.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/workspace/experiments")


# Map short codes in folder names to full context names
CODE_TO_CONTEXT = {
    "N": "standard",
    "H": "hack",
    "NH": "do_not_hack",
}
CONTEXT_TO_CODE = {v: k for k, v in CODE_TO_CONTEXT.items()}


def parse_folder_name(name: str) -> tuple[str, str] | None:
    """Try to parse 'X_to_Y' folder names into (generation_context, training_context)."""
    parts = name.split("_to_")
    if len(parts) == 2 and parts[0] in CODE_TO_CONTEXT and parts[1] in CODE_TO_CONTEXT:
        return CODE_TO_CONTEXT[parts[0]], CODE_TO_CONTEXT[parts[1]]
    return None


def load_latest_report(directory: Path) -> dict | None:
    """Load the most recent evaluation_report JSON from a directory."""
    reports = sorted(directory.glob("evaluation_report_*.json"))
    if not reports:
        return None
    with open(reports[-1]) as f:
        return json.load(f)


def extract_context_metrics(report: dict) -> dict[str, dict]:
    """Return {context_name: metrics_dict} from a report."""
    return report.get("context_metrics", {})


def find_eval_reports(exp_dir: Path) -> dict[str, dict]:
    """Find all evaluation reports under an experiment directory.

    Returns a dict keyed by eval stage (e.g. 'evaluation/standard',
    'post_training_evaluation/standard', 'training_data/generated/standard').
    Values are the loaded report dicts.
    """
    results = {}
    for report_file in sorted(exp_dir.rglob("evaluation_report_*.json")):
        rel = report_file.parent.relative_to(exp_dir)
        key = str(rel)
        # Keep only the latest if multiple timestamps exist
        results[key] = None  # placeholder
    # Now load latest from each directory
    results = {}
    for d in sorted(set(p.parent for p in exp_dir.rglob("evaluation_report_*.json"))):
        rel = str(d.relative_to(exp_dir))
        report = load_latest_report(d)
        if report:
            results[rel] = report
    return results


def wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    Returns (lower, upper) as percentages (0-100 scale).
    """
    if n == 0:
        return (0.0, 0.0)
    p = count / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, center - spread) * 100
    hi = min(1.0, center + spread) * 100
    return (lo, hi)


def format_pct(value: float | None, ci: tuple[float, float] | None = None) -> str:
    if value is None:
        return "  -  "
    if ci is not None:
        return f"{value:5.1f}% [{ci[0]:4.1f}, {ci[1]:4.1f}]"
    return f"{value:5.1f}%"


def format_float(value: float | None, ci: tuple[float, float] | None = None) -> str:
    if value is None:
        return "  -  "
    if ci is not None:
        return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{value:.3f}"


def mean_score_ci(metrics: dict, z: float = 1.96) -> tuple[float, float] | None:
    """Compute CI for the mean score using normal approximation."""
    mean = metrics.get("mean_score")
    std = metrics.get("std_score")
    n = metrics.get("total_samples")
    if mean is None or std is None or not n:
        return None
    se = std / math.sqrt(n)
    return (mean - z * se, mean + z * se)


def get_category_pct(metrics: dict, category: str) -> float | None:
    pcts = metrics.get("category_percentages", {})
    return pcts.get(category)


def get_category_count(metrics: dict, category: str) -> int | None:
    counts = metrics.get("category_counts", {})
    return counts.get(category)


def print_table(rows: list[list[str]], header: list[str]):
    """Print a formatted ASCII table."""
    all_rows = [header] + rows
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    def fmt_row(row):
        return " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))

    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def display_experiment_group(
    exp_dir: Path,
    label: str | None = None,
    eval_context: str | None = None,
):
    """Display results for a single experiment directory."""
    if label is None:
        label = exp_dir.name

    reports = find_eval_reports(exp_dir)
    if not reports:
        # Check training state
        state_file = exp_dir / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            status = state.get("status", "unknown")
            print(f"  {label}: {status} (no evaluation results yet)")
        else:
            print(f"  {label}: no results found")
        return None

    # Collect results per eval stage
    EVAL_DIR_PREFIXES = ("evaluation", "post_training_evaluation")
    stage_results = {}
    for rel_path, report in reports.items():
        ctx_metrics = extract_context_metrics(report)
        for ctx_name, metrics in ctx_metrics.items():
            if eval_context and ctx_name != eval_context:
                continue
            if rel_path.startswith(EVAL_DIR_PREFIXES):
                stage_key = rel_path.rsplit("/", 1)[0] if "/" in rel_path else rel_path
            else:
                stage_key = rel_path
            stage_results.setdefault(stage_key, {})[ctx_name] = metrics

    return stage_results


def display_context_sweep(base_dir: Path, eval_context: str | None = None, show_training_data: bool = False, show_ci: bool = False):
    """Display results for a context_sweep directory with generation->training combos."""
    subdirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    # Categorize: parseable folder names vs generic
    sweep_dirs = []
    other_dirs = []
    for d in subdirs:
        parsed = parse_folder_name(d.name)
        if parsed:
            sweep_dirs.append((d, parsed))
        else:
            other_dirs.append(d)

    if not sweep_dirs and not other_dirs:
        print(f"No experiment directories found in {base_dir}")
        return

    # Determine which eval stages are available across experiments
    all_stages = set()
    experiment_data = {}
    for d, parsed in sweep_dirs:
        stage_results = display_experiment_group(d, eval_context=eval_context)
        if stage_results:
            experiment_data[(d.name, parsed)] = stage_results
            all_stages.update(stage_results.keys())

    # Also handle incomplete experiments
    incomplete = []
    for d, parsed in sweep_dirs:
        if (d.name, parsed) not in experiment_data:
            state_file = d / "training_state.json"
            status = "no data"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                status = state.get("status", "unknown")
                error = state.get("error")
                if error:
                    status = f"{status} (error: {error})"
            gen_ctx = CONTEXT_TO_CODE.get(parsed[0], parsed[0])
            train_ctx = CONTEXT_TO_CODE.get(parsed[1], parsed[1])
            incomplete.append((f"{gen_ctx} → {train_ctx}", status))

    def is_training_data_stage(s: str) -> bool:
        return s.startswith("training_data")

    stage_order = [
        "evaluation",
        "post_training_evaluation",
        "training_data/generated",
        "training_data/selected",
        "training_data/final",
    ]

    def stage_sort_key(s: str) -> int:
        for i, prefix in enumerate(stage_order):
            if s == prefix or s.startswith(prefix + "/"):
                return i
        return len(stage_order)

    if show_training_data:
        sorted_stages = sorted(all_stages, key=stage_sort_key)
    else:
        sorted_stages = sorted(
            [s for s in all_stages if not is_training_data_stage(s)],
            key=stage_sort_key,
        )

    # Determine eval contexts to show
    all_eval_contexts = set()
    for stage_results in experiment_data.values():
        for stage, ctx_map in stage_results.items():
            all_eval_contexts.update(ctx_map.keys())

    if eval_context:
        show_contexts = [eval_context]
    else:
        ctx_order = ["standard", "hack", "do_not_hack"]
        show_contexts = [c for c in ctx_order if c in all_eval_contexts]
        show_contexts += sorted(all_eval_contexts - set(ctx_order))

    for stage in sorted_stages:
        for ctx in show_contexts:
            stage_label = stage.replace("_", " ").title()
            ctx_label = ctx if len(show_contexts) > 1 else ""
            title = f"{stage_label}"
            if ctx_label:
                title += f" (eval context: {ctx_label})"

            rows = []
            for (name, (gen_ctx, train_ctx)), stage_results in sorted(
                experiment_data.items()
            ):
                metrics = stage_results.get(stage, {}).get(ctx)
                if metrics is None:
                    continue

                gen_code = CONTEXT_TO_CODE.get(gen_ctx, gen_ctx)
                train_code = CONTEXT_TO_CODE.get(train_ctx, train_ctx)

                correct_pct = get_category_pct(metrics, "correct")
                hack_pct = get_category_pct(metrics, "hacking")
                incorrect_pct = get_category_pct(metrics, "incorrect")
                invalid_pct = get_category_pct(metrics, "invalid")
                mean_score = metrics.get("mean_score")
                n_samples = metrics.get("total_samples")

                if show_ci and n_samples:
                    correct_ci = wilson_ci(get_category_count(metrics, "correct") or 0, n_samples)
                    hack_ci = wilson_ci(get_category_count(metrics, "hacking") or 0, n_samples)
                    incorrect_ci = wilson_ci(get_category_count(metrics, "incorrect") or 0, n_samples)
                    invalid_ci = wilson_ci(get_category_count(metrics, "invalid") or 0, n_samples)
                    ms_ci = mean_score_ci(metrics)
                else:
                    correct_ci = hack_ci = incorrect_ci = invalid_ci = ms_ci = None

                rows.append([
                    f"{gen_code} → {train_code}",
                    format_pct(correct_pct, correct_ci),
                    format_pct(hack_pct, hack_ci),
                    format_pct(incorrect_pct, incorrect_ci),
                    format_pct(invalid_pct, invalid_ci),
                    format_float(mean_score, ms_ci),
                    str(n_samples) if n_samples else "-",
                ])

            if rows:
                print(f"\n{title}")
                print("=" * len(title))
                header = ["Gen → Train", "Correct%", "Hack%", "Incorrect%", "Invalid%", "Mean Score", "N"]
                print_table(rows, header)

    if incomplete:
        print("\nIncomplete Experiments")
        print("=" * 24)
        for label, status in incomplete:
            print(f"  {label}: {status}")


def display_generic(exp_dir: Path, eval_context: str | None = None, show_ci: bool = False):
    """Display results for a generic experiment directory."""
    stage_results = display_experiment_group(exp_dir, eval_context=eval_context)
    if not stage_results:
        return

    for stage, ctx_map in sorted(stage_results.items()):
        for ctx, metrics in sorted(ctx_map.items()):
            title = f"{exp_dir.name} — {stage} (eval: {ctx})"
            print(f"\n{title}")
            print("=" * len(title))

            rows = []
            correct_pct = get_category_pct(metrics, "correct")
            hack_pct = get_category_pct(metrics, "hacking")
            incorrect_pct = get_category_pct(metrics, "incorrect")
            invalid_pct = get_category_pct(metrics, "invalid")
            mean_score = metrics.get("mean_score")
            n_samples = metrics.get("total_samples")

            if show_ci and n_samples:
                correct_ci = wilson_ci(get_category_count(metrics, "correct") or 0, n_samples)
                hack_ci = wilson_ci(get_category_count(metrics, "hacking") or 0, n_samples)
                incorrect_ci = wilson_ci(get_category_count(metrics, "incorrect") or 0, n_samples)
                invalid_ci = wilson_ci(get_category_count(metrics, "invalid") or 0, n_samples)
                ms_ci = mean_score_ci(metrics)
            else:
                correct_ci = hack_ci = incorrect_ci = invalid_ci = ms_ci = None

            header = ["Correct%", "Hack%", "Incorrect%", "Invalid%", "Mean Score", "N"]
            rows.append([
                format_pct(correct_pct, correct_ci),
                format_pct(hack_pct, hack_ci),
                format_pct(incorrect_pct, incorrect_ci),
                format_pct(invalid_pct, invalid_ci),
                format_float(mean_score, ms_ci),
                str(n_samples) if n_samples else "-",
            ])
            print_table(rows, header)


def main():
    parser = argparse.ArgumentParser(
        description="Display evaluation results from experiment directories."
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        default=[f"{OUTPUT_ROOT}/context_sweep"],
        help="Experiment directories to scan (default: $OUTPUT_ROOT/context_sweep)",
    )
    parser.add_argument(
        "--eval-context",
        "-c",
        default=None,
        help="Filter to a specific evaluation context (e.g. standard, hack, do_not_hack)",
    )
    parser.add_argument(
        "--training-data",
        "-t",
        action="store_true",
        default=False,
        help="Also show training data evaluation reports (these use generation/training context, not evaluation context)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        default=False,
        help="Show 95%% Wilson score confidence intervals for proportions and normal CIs for mean score",
    )
    args = parser.parse_args()

    for dir_path_str in args.dirs:
        dir_path = Path(dir_path_str)
        if not dir_path.exists():
            print(f"Directory not found: {dir_path}", file=sys.stderr)
            continue

        print(f"\n{'#' * 60}")
        print(f"# {dir_path}")
        print(f"{'#' * 60}")

        # Detect if this is a sweep-style directory (has X_to_Y subdirs)
        has_sweep_dirs = any(
            parse_folder_name(d.name) is not None
            for d in dir_path.iterdir()
            if d.is_dir()
        )

        if has_sweep_dirs:
            display_context_sweep(dir_path, eval_context=args.eval_context, show_training_data=args.training_data, show_ci=args.ci)
        else:
            # Check if it has subdirs with results, or is itself an experiment
            subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
            has_reports = list(dir_path.rglob("evaluation_report_*.json"))
            if has_reports:
                if any(d.name in ("evaluation", "post_training_evaluation", "training_data") for d in subdirs):
                    display_generic(dir_path, eval_context=args.eval_context, show_ci=args.ci)
                else:
                    for d in sorted(subdirs):
                        display_generic(d, eval_context=args.eval_context, show_ci=args.ci)
            else:
                print(f"  No evaluation reports found in {dir_path}")


if __name__ == "__main__":
    main()
