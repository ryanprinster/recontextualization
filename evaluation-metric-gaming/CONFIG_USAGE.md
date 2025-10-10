# Config-Based Execution

## Overview

`main.py` accepts a YAML configuration file that specifies all experiment parameters. This provides a cleaner, more reproducible way to run experiments compared to long CLI commands.

## Usage

```bash
python main.py --config path/to/your/config.yaml
```

## Configuration File

See `configs/example_experiment_config.yaml` for a complete example with all available options.

### Required Fields

- `experiment_name`: Name of the experiment (used for paths and identifiers)
- `N_completions`: Number of completions to generate for best-of-N

### Key Configuration Sections

#### Model Configuration
```yaml
base_model: "gpt-4.1-mini-2025-04-14"
judge_model: "gpt-4o"
```

#### Dataset Configuration
```yaml
experiment_name: "school-of-reward-hacks"
dataset_name: null  # Defaults to longtermrisk/{experiment_name}
```

#### Pipeline Control
```yaml
skip_data_downloading: false
skip_sft: false
skip_bon_generation: false
skip_expert_iteration_step: false
```

#### Generation Settings
```yaml
generation_temperature: 1.0
generation_max_tokens: 500
generation_prompt_template_path: "prompts/data_generation/simple_prompt.yaml"
```

#### Judge Settings
```yaml
judge_temperature: 0.1
judge_max_tokens: 500
judge_template_path: null  # Auto-computed from experiment_name
```

#### Expert Iteration
```yaml
expert_iteration_seeds: "1,42,3"  # Can also be a list: [1, 42, 3]
expert_iteration_config_path: "configs/expert_iteration_sft_config.yaml"
```

## Default Values

All fields have sensible defaults matching the original `run_sorh_experiment.py` behavior:

- `base_model`: "gpt-4.1-mini-2025-04-14"
- `judge_model`: "gpt-4o"
- `generation_temperature`: 1.0
- `generation_max_tokens`: 500
- `judge_temperature`: 0.1
- `judge_max_tokens`: 500
- `bon_selection_seed`: 42
- `expert_iteration_seeds`: "1"
- `filter_out_hardcoding_task`: true
- `debug`: false

## Example Configs

### Minimal Config (Standard Training)
```yaml
experiment_name: "school-of-reward-hacks"
N_completions: 5
debug: false
```

### Debug Mode
```yaml
experiment_name: "school-of-reward-hacks"
N_completions: 5
debug: true
```

### Skip SFT (Use Existing Model)
```yaml
experiment_name: "school-of-reward-hacks"
N_completions: 5
skip_sft: true
sft_finetuned_id: "ft:gpt-4.1-mini-2025-04-14:..."
```

### Multiple Expert Iteration Seeds
```yaml
experiment_name: "school-of-reward-hacks"
N_completions: 5
expert_iteration_seeds: [1, 42, 3, 5, 123]
```

## Backwards Compatibility

This implementation maintains identical behavior to the original `run_sorh_experiment.py`:
- Same default values
- Same directory structure
- Same pipeline steps
- All Python endpoint functions preserve exact behavior

## Migration from CLI

Old CLI command:
```bash
python run_sorh_experiment.py \
  --experiment_name school-of-reward-hacks \
  --N-completions 5 \
  --debug
```

New config-based approach:
1. Create a config file `my_experiment.yaml`:
```yaml
experiment_name: "school-of-reward-hacks"
N_completions: 5
debug: true
```

2. Run:
```bash
python main.py --config my_experiment.yaml
```