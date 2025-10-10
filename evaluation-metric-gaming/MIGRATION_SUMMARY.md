# Migration Summary: CLI to Config-Based Execution

## Overview
The experiment pipeline has been refactored from CLI-based execution to YAML config-based execution with Python function calls instead of subprocess commands.

## Key Changes

### 1. Python Endpoints Created
All modules now have `main_python()` functions that can be called directly:
- `src/data_generation/download_data.py`
- `src/data_generation/generate_N_completions.py`
- `src/data_generation/filter_best_of_n.py`
- `src/judge/judge_completions.py`
- `src/eval.py`

### 2. New Main Entry Point
**File**: `main.py`
- Accepts only `--config` argument pointing to YAML file
- Uses Python function calls instead of subprocess/CLI
- Maintains identical behavior to `run_sorh_experiment.py`

### 3. Configuration System
**File**: `src/config_validator.py`
- `get_default_config()`: Returns all default values
- `validate_and_merge_config()`: Merges YAML config with defaults and validates

**File**: `configs/example_experiment_config.yaml`
- Complete example configuration with all options documented

### 4. Documentation
**File**: `CONFIG_USAGE.md`
- How to use the config system
- Examples of different configurations
- Migration guide from CLI to config

## Usage

### Old Way (CLI)
```bash
python run_sorh_experiment.py \
  --experiment_name school-of-reward-hacks \
  --N-completions 5 \
  --debug \
  --skip-sft \
  --sft-finetuned-id "ft:gpt-4.1-mini-2025-04-14:..."
```

### New Way (Config)
Create `my_config.yaml`:
```yaml
experiment_name: "school-of-reward-hacks"
N_completions: 5
debug: true
skip_sft: true
sft_finetuned_id: "ft:gpt-4.1-mini-2025-04-14:..."
```

Run:
```bash
python main.py --config my_config.yaml
```

## Benefits

1. **Cleaner**: No more long CLI commands
2. **Reproducible**: Config files can be version controlled
3. **Shareable**: Easy to share exact experiment configurations
4. **Maintainable**: Python function calls are easier to debug than subprocess
5. **Flexible**: Supports both string and list formats for parameters
6. **Type-safe**: Direct function calls catch errors earlier

## Backwards Compatibility

- All original modules (`run_sorh_experiment.py`, etc.) still work with CLI
- All `main_python()` functions are backward compatible
- Same defaults, same behavior, same outputs
- Existing scripts and workflows continue to work

## File Structure

```
evaluation-metric-gaming/
├── main.py                          # New config-based entry point
├── run_sorh_experiment.py           # Original (still works)
├── CONFIG_USAGE.md                  # Documentation
├── MIGRATION_SUMMARY.md             # This file
├── configs/
│   ├── example_experiment_config.yaml
│   └── ...
└── src/
    ├── config_validator.py          # New: config validation
    ├── data_generation/
    │   ├── download_data.py         # Added main_python()
    │   ├── generate_N_completions.py # Added main_python()
    │   └── filter_best_of_n.py      # Added main_python()
    ├── judge/
    │   └── judge_completions.py     # Already had main_python()
    └── eval.py                       # Added main_python()
```

## Testing

To test the new system:
```bash
# 1. Use the example config
python main.py --config configs/example_experiment_config.yaml

# 2. Create your own minimal config
echo "experiment_name: test
N_completions: 5
debug: true" > test_config.yaml

python main.py --config test_config.yaml
```

## Next Steps

Consider creating config files for common experiment types:
- `configs/standard_training.yaml`
- `configs/prompt_modification.yaml`
- `configs/recontextualization.yaml`
- `configs/debug.yaml`