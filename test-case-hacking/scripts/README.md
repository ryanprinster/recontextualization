# Scripts Guide

This directory contains the main entry point scripts for the training system. Each script has a specific purpose in the experiment workflow.

## Core Scripts

### `train.py` - Start Training

Start a new training run with the specified configuration.

**Usage:**
```bash
python scripts/train.py --config-name train/best_of_n
python scripts/train.py --config-name train/recontextualization
```

**What it does:**
1. Loads the dataset and model configuration
2. Generates or loads cached rollouts
3. Applies selection method (best-of-N)
4. Applies detection method (recontextualization, if configured)
5. Creates training data file (JSONL format)
6. Submits fine-tuning job to OpenAI
7. Saves state and returns immediately

**Override parameters:**
```bash
# Change model
python scripts/train.py --config-name train/best_of_n model=gpt4_1

# Change seed
python scripts/train.py --config-name train/best_of_n seed=42

# Change generation parameters
python scripts/train.py --config-name train/best_of_n \
    training.training_generation.n_rollouts=32

# Change OpenAI training parameters
python scripts/train.py --config-name train/best_of_n \
    training.n_epochs=5 \
    training.learning_rate_multiplier=2.0
```

**Output:**
- Creates experiment directory: `experiments/training/<config>/<timestamp>/`
- Saves `training_state.json` with job info
- Saves `training_data.jsonl` (the file sent to OpenAI)
- Saves `.hydra/config.yaml` (complete config used)

---

### `status.py` - Check Training Status

Check the current status of a training run without modifying anything.

**Usage:**
```bash
python scripts/status.py experiments/training/best_of_n/2025-01-15_10-30-00
```

**What it shows:**
- Current training status (none, running, completed, failed)
- Start time
- Completion time (if finished)
- Model reference (if completed)
- OpenAI job ID
- OpenAI file ID
- Error message (if failed)

**Example output:**
```
📊 Status: running
🕐 Started: 2025-01-15T10:30:00
🔗 OpenAI Job ID: ftjob-abc123xyz
📁 OpenAI File ID: file-xyz789abc
```

---

### `resume.py` - Resume/Complete Training

Check if training is complete and run evaluation when ready. This is the main script to use after training has been submitted.

**Usage:**
```bash
python scripts/resume.py experiments/training/best_of_n/2025-01-15_10-30-00
```

**What it does:**
1. Loads the experiment configuration
2. Queries OpenAI for the current job status
3. If still training: Reports progress
4. If completed: Runs final evaluation and saves results
5. If failed: Shows error message

**Example output (still running):**
```
📊 Status: running
⏳ Training still in progress...
```

**Example output (completed):**
```
📊 Status: completed
🎉 Training completed!
🤖 Model: ft:gpt-4o-mini:org:suffix:abc123
```

---

### `eval.py` - Evaluate Model

Evaluate a model on the dataset with multiple contexts.

**Two modes:**

**1. Evaluate trained model from experiment:**
```bash
python scripts/eval.py experiments/training/best_of_n/2025-01-15_10-30-00
```
This uses the trained model from that experiment.

**2. Evaluate any model with config:**
```bash
python scripts/eval.py --config-name eval/code_generation
```
This uses the model specified in the config (default: gpt4_1_mini).

**Override model:**
```bash
python scripts/eval.py --config-name eval/code_generation \
    model.name="ft:gpt-4o-mini:org:suffix:abc123"
```

**What it does:**
1. Loads the dataset
2. Generates or loads cached rollouts for each context
3. Evaluates rollouts (checks if they pass tests)
4. Saves detailed results to `evaluation/` directory

**Output:**
- Creates `evaluation/` subdirectory in the output folder
- Saves metrics and detailed results
- Results include success rates per context

---

### `pregenerate.py` - Pre-generate Cache

Pre-generate model responses and cache them for faster experimentation.

**Usage:**
```bash
python scripts/pregenerate.py --config-name pregenerate/code_generation
```

**What it does:**
1. Loads dataset and model
2. For each context (standard, hack, do_not_hack):
   - Generates rollouts for all samples
   - Evaluates them
   - Saves to cache
3. Reports statistics

**Why use this:**
- Speeds up training runs (no need to regenerate)
- Ensures consistency across experiments
- Useful when iterating on selection/detection methods

**Cache location:**
- Default: `cache/` directory
- Organized by model, dataset, context, and generation config

**Override parameters:**
```bash
# Change model
python scripts/pregenerate.py --config-name pregenerate/code_generation model=gpt4_1

# Change contexts
python scripts/pregenerate.py --config-name pregenerate/code_generation \
    pregenerate.contexts='["standard", "do_not_hack"]'

# Change number of rollouts
python scripts/pregenerate.py --config-name pregenerate/code_generation \
    pregenerate.generation_configs.training.n_rollouts=32
```

---

### `utils.py` - Helper Functions

Shared utility functions used by the scripts:

- `get_experiment_config()`: Load config from experiment directory
- `create_trainer_from_config()`: Create trainer instance from config
- `load_trained_model_from_experiment()`: Load trained model from experiment
- `load_model_from_experiment_or_config()`: Unified model loading

These are internal utilities used by the other scripts.

---

## Common Workflows

### Complete Training Workflow

1. **Optional: Pre-generate cache**
   ```bash
   python scripts/pregenerate.py --config-name pregenerate/code_generation
   ```

2. **Start training**
   ```bash
   python scripts/train.py --config-name train/best_of_n
   ```
   Note the experiment directory path printed.

3. **Check status (optional)**
   ```bash
   python scripts/status.py experiments/training/best_of_n/2025-01-15_10-30-00
   ```

4. **Wait for completion, then evaluate**
   ```bash
   python scripts/resume.py experiments/training/best_of_n/2025-01-15_10-30-00
   ```

5. **Later: Re-evaluate if needed**
   ```bash
   python scripts/eval.py experiments/training/best_of_n/2025-01-15_10-30-00
   ```

### Quick Evaluation of Base Model

```bash
python scripts/eval.py --config-name eval/code_generation
```

### Evaluate Specific Model

```bash
python scripts/eval.py --config-name eval/code_generation \
    model.name="ft:gpt-4o-mini:personal::BFxiYY8j"
```

---

## Configuration Tips

### Hydra Basics

All scripts use Hydra for configuration. Basic patterns:

```bash
# Use a specific config
--config-name path/to/config

# Override single value
param=value

# Override nested value
nested.param=value

# Override list
'list_param=[1,2,3]'
```

### Config Composition

Configs are composed from components. See `configs/` directory structure.

Example config structure:
```yaml
defaults:
  - /components/model: gpt4_1_mini
  - /components/dataset: code_generation
  - /components/training: openai
```

### Useful Overrides

```bash
# Change output directory
hydra.run.dir=my_custom_dir

# Change seeds
seed=42 dataset_seed=42 training_seed=42

# Enable/disable cache (in code, not CLI)
# The scripts use cache by default via use_cache() context manager
```

---

## Troubleshooting

### "Config not found"

Make sure you're running from the project root:
```bash
cd /path/to/project
python scripts/train.py --config-name train/best_of_n
```

### "Module not found"

The scripts add the parent directory to `sys.path`. Run from project root.

### "Experiment directory not found"

Check the path. The scripts expect:
```
experiments/<config_name>/<timestamp>/
```

### "OpenAI API key not set"

```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Development

When adding new scripts, follow these patterns:

1. **Use Hydra**: `@hydra.main(version_base=None, config_path="../configs", config_name="...")`
2. **Add to sys.path**: `sys.path.insert(0, str(Path(__file__).parent.parent))`
3. **Import from src**: `from src.module import ...`
4. **Document usage**: Add to this README

---

For more information, see the main [README.md](../README.md) in the project root.
