#!/usr/bin/env python
"""Profile datasets to analyze task, evaluation_metric, and cheat_method composition."""

import json
import argparse
from collections import Counter
from pathlib import Path


def load_jsonl(path: str) -> list:
    """Load JSONL file."""
    with open(path, 'r') as file:
        lines = [json.loads(f) for f in file if f.strip()]
    return lines


def profile_jsonl(file_path):
    """Profile a JSONL dataset."""
    data = load_jsonl(file_path)
    
    tasks = []
    metrics = []
    cheats = []
    
    for sample in data:
        if 'task' in sample:
            tasks.append(sample['task'])
        if 'evaluation_metric' in sample:
            metrics.append(sample['evaluation_metric'])
        if 'cheat_method' in sample:
            cheats.append(sample['cheat_method'])
    
    return Counter(tasks), Counter(metrics), Counter(cheats), len(data)


def profile_completions_json(file_path):
    """Profile a completions JSON dataset."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    tasks = []
    metrics = []
    cheats = []
    
    prompt_to_completions = data.get('prompt_to_completions', {})
    
    for idx, sample in prompt_to_completions.items():
        if 'task' in sample:
            tasks.append(sample['task'])
        if 'evaluation_metric' in sample:
            metrics.append(sample['evaluation_metric'])
        if 'cheat_method' in sample:
            cheats.append(sample['cheat_method'])
    
    return Counter(tasks), Counter(metrics), Counter(cheats), len(prompt_to_completions)


def save_profile(output_path, profiles):
    """Save profile results to text file."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET PROFILE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Store baseline for comparison
        baseline_name = None
        baseline_tasks = None
        baseline_total = None
        
        for dataset_name, (tasks, metrics, cheats, total) in profiles.items():
            # Check if this is the baseline dataset
            if "school-of-reward-hacks.jsonl" in dataset_name:
                baseline_name = dataset_name
                baseline_tasks = tasks
                baseline_total = total
            
            f.write(f"\nDATASET: {dataset_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {total}\n\n")
            
            # Tasks
            f.write("TASKS:\n")
            if tasks:
                for task, count in tasks.most_common():
                    percentage = (count / total) * 100
                    f.write(f"  - {task}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("  No task field found\n")
            f.write("\n")
            
            # Evaluation Metrics
            f.write("EVALUATION METRICS:\n")
            if metrics:
                for metric, count in metrics.most_common():
                    percentage = (count / total) * 100
                    # Truncate long metrics for readability
                    metric_display = metric[:60] + "..." if len(metric) > 60 else metric
                    f.write(f"  - {metric_display}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("  No evaluation_metric field found\n")
            f.write("\n")
            
            # Cheat Methods
            f.write("CHEAT METHODS:\n")
            if cheats:
                for cheat, count in cheats.most_common():
                    percentage = (count / total) * 100
                    # Truncate long cheats for readability
                    cheat_display = cheat[:60] + "..." if len(cheat) > 60 else cheat
                    f.write(f"  - {cheat_display}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("  No cheat_method field found\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        # Summary statistics
        f.write("\nSUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        
        all_tasks = Counter()
        all_metrics = Counter()
        all_cheats = Counter()
        
        for _, (tasks, metrics, cheats, _) in profiles.items():
            all_tasks.update(tasks)
            all_metrics.update(metrics)
            all_cheats.update(cheats)
        
        f.write(f"Unique tasks: {len(all_tasks)}\n")
        f.write(f"Unique evaluation metrics: {len(all_metrics)}\n")
        f.write(f"Unique cheat methods: {len(all_cheats)}\n")
        
        # Add representation analysis
        if baseline_tasks and baseline_total:
            f.write("\n" + "=" * 80 + "\n")
            f.write("\nREPRESENTATION ANALYSIS (vs baseline: school-of-reward-hacks.jsonl)\n")
            f.write("-" * 40 + "\n\n")
            
            for dataset_name, (tasks, metrics, cheats, total) in profiles.items():
                if dataset_name == baseline_name:
                    continue  # Skip baseline itself
                
                f.write(f"\n{dataset_name} REPRESENTATION:\n")
                f.write("-" * 40 + "\n")
                
                # Calculate expected vs actual for each task
                representation_diffs = []
                
                # Get all tasks from both baseline and current dataset
                all_task_names = set(baseline_tasks.keys()) | set(tasks.keys())
                
                for task in all_task_names:
                    baseline_pct = (baseline_tasks.get(task, 0) / baseline_total) * 100
                    actual_count = tasks.get(task, 0)
                    actual_pct = (actual_count / total) * 100
                    expected_count = (baseline_tasks.get(task, 0) / baseline_total) * total
                    
                    if baseline_pct > 0:
                        representation_ratio = actual_pct / baseline_pct
                    else:
                        representation_ratio = float('inf') if actual_count > 0 else 0
                    
                    diff_pct = actual_pct - baseline_pct
                    
                    representation_diffs.append((task, baseline_pct, actual_pct, diff_pct, representation_ratio, actual_count, expected_count))
                
                # Sort by absolute difference
                representation_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
                
                # Show overrepresented tasks
                f.write("\nOVERREPRESENTED TASKS (higher % than baseline):\n")
                overrep = [x for x in representation_diffs if x[3] > 0.5]  # At least 0.5% difference
                if overrep:
                    for task, base_pct, actual_pct, diff_pct, ratio, actual_count, expected_count in overrep[:10]:
                        f.write(f"  - {task}:\n")
                        f.write(f"    Baseline: {base_pct:.1f}% → Actual: {actual_pct:.1f}% (diff: +{diff_pct:.1f}%)\n")
                        f.write(f"    Count: {int(actual_count)} (expected: {expected_count:.1f})\n")
                        if ratio != float('inf'):
                            f.write(f"    Representation: {ratio:.2f}x baseline rate\n")
                else:
                    f.write("  None significant\n")
                
                # Show underrepresented tasks
                f.write("\nUNDERREPRESENTED TASKS (lower % than baseline):\n")
                underrep = [x for x in representation_diffs if x[3] < -0.5]  # At least -0.5% difference
                if underrep:
                    for task, base_pct, actual_pct, diff_pct, ratio, actual_count, expected_count in underrep[:10]:
                        f.write(f"  - {task}:\n")
                        f.write(f"    Baseline: {base_pct:.1f}% → Actual: {actual_pct:.1f}% (diff: {diff_pct:.1f}%)\n")
                        f.write(f"    Count: {int(actual_count)} (expected: {expected_count:.1f})\n")
                        if base_pct > 0:
                            f.write(f"    Representation: {ratio:.2f}x baseline rate\n")
                else:
                    f.write("  None significant\n")
                
                # Show tasks completely missing or new
                f.write("\nCOMPLETELY MISSING TASKS (in baseline but not in this split):\n")
                missing = [x for x in representation_diffs if x[1] > 0 and x[2] == 0]
                if missing:
                    for task, base_pct, _, _, _, _, expected_count in missing:
                        f.write(f"  - {task}: {base_pct:.1f}% in baseline ({expected_count:.1f} samples expected)\n")
                else:
                    f.write("  None\n")
                
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Profile dataset composition")
    parser.add_argument('datasets', nargs='+', help='Paths to dataset files (JSONL or JSON)')
    parser.add_argument('-o', '--output', default='dataset_profile.txt', 
                       help='Output file for profile results')
    
    args = parser.parse_args()
    
    profiles = {}
    
    for dataset_path in args.datasets:
        if not Path(dataset_path).exists():
            print(f"Warning: {dataset_path} not found, skipping...")
            continue
        
        dataset_name = Path(dataset_path).name
        
        try:
            if dataset_path.endswith('.jsonl'):
                profiles[dataset_name] = profile_jsonl(dataset_path)
            elif dataset_path.endswith('.json'):
                profiles[dataset_name] = profile_completions_json(dataset_path)
            else:
                print(f"Warning: Unknown file type for {dataset_path}, skipping...")
                continue
            
            print(f"Profiled {dataset_name}")
        except Exception as e:
            print(f"Error profiling {dataset_path}: {e}")
    
    if profiles:
        # Determine output path
        if '/' in args.output:
            output_path = args.output
        else:
            # Save in data/school-of-reward-hacks/ directory
            output_dir = Path(args.datasets[0]).parent
            output_path = output_dir / args.output
        
        save_profile(output_path, profiles)
        print(f"\nProfile saved to: {output_path}")
    else:
        print("No datasets successfully profiled")


if __name__ == '__main__':
    main()