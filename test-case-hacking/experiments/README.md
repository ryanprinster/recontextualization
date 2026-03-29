# Experiment Scripts

Sweep runners and analysis/plotting scripts for the test-case-hacking experiments.

## Structure

- `env.sh` — Source this to set `MODULE_ROOT` and `OUTPUT_ROOT`. Override `OUTPUT_ROOT` to change where experiment data is written (default: `/workspace/experiments`).
- `sweeps/` — Shell scripts that run training/evaluation sweeps. Each script sources `env.sh` and writes output to `$OUTPUT_ROOT`.
- `analysis/` — Python scripts for plotting results and inspecting trajectories. Each reads from `$OUTPUT_ROOT`.

## Usage

```bash
# Run a sweep (from any directory)
bash experiments/sweeps/run_context_sweep_v2.sh

# Override output location
export OUTPUT_ROOT=/data/my_experiments
bash experiments/sweeps/run_context_sweep_v2.sh

# Plot results
python experiments/analysis/plot_context_sweep.py
```

Only scripts are tracked in git. Experiment output data lives under `$OUTPUT_ROOT` (outside the repo by default).
