# Config-Based Experiment Runner

## Overview

All 15 experiment variants from `gpt_41_mini_results.sh` have been converted to YAML config files. The new `gpt_41_mini_results_config.sh` script runs these experiments using `main.py` with config files.

## Usage

### Run All Experiments

```bash
# Normal mode
./gpt_41_mini_results_config.sh false "1,42,3,5,123"

# Debug mode (runs with 3 samples)
./gpt_41_mini_results_config.sh true "1,42,3,5,123"
```

### Run Single Experiment

```bash
# Normal mode
python main.py --config configs/experiments/standard_training.yaml

# Debug mode override
python main.py --config configs/experiments/standard_training.yaml --debug
```

## Experiment Configs

All config files are in `configs/experiments/`:

### Neutral Generation Experiments (8 variants)
1. **standard_training.yaml** - Standard training, no modifications
2. **recon_cheat_method_user_suffix.yaml** - Recon with cheat method
3. **recon_overfit_message_user_suffix.yaml** - Recon with overfit message
4. **recon_yellow_tag_user_suffix.yaml** - Recon with `<YELLOW>` tag
5. **recon_numeric_tag_user_suffix.yaml** - Recon with `[1 2 3 4 5]` tag
6. **recon_dont_overfit_user_suffix.yaml** - Recon with don't overfit message
7. **recon_neutral_user_suffix.yaml** - Recon with neutral message
8. **recon_thanks_user_suffix.yaml** - Recon with thanks message

### Exploit Generation Experiments (3 variants)
9. **standard_training_exploit_gen.yaml** - Standard training with exploit generation
10. **context_distillation_exploit_gen.yaml** - Context distillation with exploit
11. **recon_dont_overfit_exploit_gen.yaml** - Recon don't overfit with exploit gen

### Don't Exploit Generation Experiments (3 variants)
12. **standard_training_dont_exploit_gen.yaml** - Standard training with don't exploit gen
13. **context_distillation_dont_exploit_gen.yaml** - Context distillation with don't exploit
14. **recon_overfit_dont_exploit_gen.yaml** - Recon overfit with don't exploit gen

## Config Structure

Each config file contains:

```yaml
# Experiment identification
experiment_name: "school-of-reward-hacks"
experiment_dir_name: "descriptive_name"
N_completions: 50

# Model configuration
base_model: "gpt-4.1-mini-2025-04-14"
sft_finetuned_id: "gpt-4.1-mini-2025-04-14"

# Pipeline control
skip_data_downloading: true
skip_sft: true
skip_bon_generation: true/false
eval_sft_and_base: true/false

# Judge configuration
judge_template_path: "prompts/judge/school_of_reward_hacks_hackable.yaml"

# Expert iteration
expert_iteration_seeds: "1,42,3,5,123"

# Modification settings (varies by experiment)
do_recontextualization: true/false
recontextualized_phrase: "specific message"
add_recontextualized_phrase_as_user_message_suffix: true/false

do_prompt_modification_generation: true/false
modified_phrase_generation: "specific message"
add_modified_phrase_as_user_message_suffix_generation: true/false

# Debug mode
debug: false
```

## Dynamic Config Updates

The bash script `gpt_41_mini_results_config.sh`:
1. Reads base config files
2. Dynamically updates:
   - `expert_iteration_seeds` from command line arg
   - `N_generations_judged_path` based on existing files
   - `skip_bon_generation` based on file existence
   - `debug` flag from command line
3. Creates temporary config files with updates
4. Runs experiments with temporary configs
5. Cleans up temporary files

## Debug Mode Override

The `--debug` flag can be passed to `main.py` to override the debug setting in the config:

```bash
# Force debug mode even if config says debug: false
python main.py --config configs/experiments/standard_training.yaml --debug
```

This is the **only** CLI override allowed. All other parameters must be in the config file.

## Benefits Over Original Script

1. **Readable**: Config files are self-documenting
2. **Reusable**: Can run individual experiments easily
3. **Maintainable**: Easy to modify experiment parameters
4. **Trackable**: Config files can be version controlled
5. **Debuggable**: Can test configs individually before batch runs
6. **Extensible**: Easy to add new experiment variants

## Migration Path

### Old Way
```bash
python run_sorh_experiment.py \
  --experiment_name "school-of-reward-hacks" \
  --N-completions 50 \
  --skip-sft \
  --sft-finetuned-id gpt-4.1-mini-2025-04-14 \
  --do-recontextualization \
  --recontextualized-phrase "Thanks!" \
  --add-recontextualized-phrase-as-user-message-suffix \
  [... many more flags ...]
```

### New Way
Create `my_experiment.yaml`:
```yaml
experiment_name: "school-of-reward-hacks"
N_completions: 50
skip_sft: true
sft_finetuned_id: "gpt-4.1-mini-2025-04-14"
do_recontextualization: true
recontextualized_phrase: " Thanks!"
add_recontextualized_phrase_as_user_message_suffix: true
```

Run:
```bash
python main.py --config my_experiment.yaml
```

## Testing

Test a single config:
```bash
# Test in debug mode
python main.py --config configs/experiments/standard_training.yaml --debug
```

Test all configs in debug mode:
```bash
./gpt_41_mini_results_config.sh true "1"
```