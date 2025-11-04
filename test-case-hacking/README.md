# Code Generation Training with Recontextualization

A training system for studying how LLMs respond to different prompting contexts in code generation tasks. Built for fine-tuning OpenAI models with expert iteration workflows including best-of-N selection and recontextualization detection.

> **Note**: This codebase has been cleaned and simplified for sharing and reproducibility. It focuses on core training workflows with OpenAI models.

## Overview

This project enables training LLMs on coding problems with **intentionally buggy test cases** to study how models respond to different prompting contexts. The system supports:

- **Multiple Context Types**: Train and evaluate with different instruction contexts (standard, do_not_hack, hack)
- **Expert Iteration**: Best-of-N selection from multiple generations
- **Recontextualization**: Detect problematic solutions and switch them to different prompting contexts
- **OpenAI Fine-tuning**: Async training with automatic state management
- **Caching System**: Pre-generate responses to speed up experiments

## Key Concepts

### Dataset

The dataset consists of **coding problems with both correct and incorrect test cases**:
- Each problem has a correct implementation
- Each problem has an incorrect test that the correct solution should fail
- Models can either: solve the problem correctly, or "hack" the test by memorizing it

### Contexts

The system supports three prompting contexts that frame how the model should approach problems:
- `standard`: Standard problem-solving with test cases provided
- `do_not_hack`: Explicitly instructs model to write general solutions and not hard-code test cases
- `hack`: Encourages passing tests by any means, even if the solution isn't general

### Training Workflows

1. **Best-of-N Selection**: Generate multiple solutions per problem, select the best one based on evaluation
2. **Recontextualization**: Detect solutions that exhibit undesired behavior and switch them to a different context for training

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

## Quick Start

### 1. Training

Train a model with best-of-N selection:

```bash
python scripts/train.py --config-name train/best_of_n
```

Train with recontextualization (detects problematic solutions and switches context):

```bash
python scripts/train.py --config-name train/recontextualization
```

The training script will:
1. Generate multiple completions per problem (using cache if available)
2. Select the best completion (for best-of-N)
3. Detect and recontextualize problematic solutions (if using recontextualization)
4. Submit to OpenAI for fine-tuning
5. Return immediately (non-blocking)

### 2. Check Training Status

```bash
python scripts/status.py experiments/training/best_of_n/2025-01-15_10-30-00
```

This shows the current status of your OpenAI fine-tuning job.

### 3. Resume/Evaluate When Complete

```bash
python scripts/resume.py experiments/training/best_of_n/2025-01-15_10-30-00
```

This checks if training is complete and runs evaluation if ready.

### 4. Evaluate Models

Evaluate a trained model:

```bash
python scripts/eval.py experiments/training/best_of_n/2025-01-15_10-30-00
```

Or evaluate a specific model by name:

```bash
python scripts/eval.py --config-name eval/code_generation model.name="ft:gpt-4o-mini:org:suffix:abc123"
```

### 5. Pre-generate Cache (Optional)

For faster experimentation, pre-generate model responses:

```bash
python scripts/pregenerate.py --config-name pregenerate/code_generation
```

This generates and caches responses for different contexts, which speeds up training runs.

## Project Structure

```
├── configs/                    # Hydra configuration files
│   ├── components/            # Reusable component configs
│   │   ├── dataset/          # Dataset configurations
│   │   ├── evaluation/       # Evaluation settings
│   │   ├── generation/       # Generation parameters
│   │   ├── model/            # Model configurations
│   │   ├── selection/        # Selection methods (best-of-N)
│   │   └── training/         # Training backend configs
│   ├── eval/                 # Complete evaluation configs
│   ├── pregenerate/          # Cache pregeneration configs
│   └── train/                # Complete training configs
├── data/                      # Training data
│   └── coding_problems.jsonl # Code generation problems
├── scripts/                   # Entry point scripts
│   ├── train.py              # Start training
│   ├── eval.py               # Evaluate models
│   ├── resume.py             # Resume/check training
│   ├── status.py             # Check training status
│   └── pregenerate.py        # Pre-generate cache
└── src/                       # Source code
    ├── configs/               # Configuration dataclasses
    ├── dataset_modules/       # Dataset implementations
    ├── evaluation/            # Evaluation logic
    ├── generation/            # Rollout generation
    ├── models/                # Model interfaces
    ├── storage/               # Caching system
    └── training/              # Training workflows
```

## Configuration

The system uses [Hydra](https://hydra.cc/) for configuration management. Configs are composed from reusable components.

### Available Models

The project includes configuration files for different OpenAI models:
- `gpt4_1` - Configured as gpt-4.1-2025-04-14  
- `gpt4_1_mini` (default) - Configured as gpt-4.1-mini-2025-04-14
- `gpt4_1_nano` - Configured as gpt-4.1-nano-2025-04-14
- `ckpt` - Template for fine-tuned checkpoints (replace with your fine-tuned model ID)

### Training Configs

**`train/best_of_n.yaml`**: Best-of-N selection training
- Generates 16 completions per problem
- Selects the best one based on test results
- Uses `do_not_hack` context for training

**`train/recontextualization.yaml`**: Best-of-N + Recontextualization
- Extends best-of-N with detection
- Detects problematic solutions
- Switches them to `standard` context for training

### Customization

Override any config parameter:

```bash
# Change model
python scripts/train.py --config-name train/best_of_n model=gpt4_1

# Change number of rollouts
python scripts/train.py --config-name train/best_of_n training.training_generation.n_rollouts=32

# Change seed
python scripts/train.py --config-name train/best_of_n seed=123

# Change OpenAI training parameters
python scripts/train.py --config-name train/best_of_n training.n_epochs=5 training.learning_rate_multiplier=1.5
```

## Experiment Outputs

Each run creates a directory under `experiments/` with:

```
experiments/training/best_of_n/2025-01-15_10-30-00/
├── .hydra/
│   └── config.yaml              # Complete config used
├── training_state.json          # Training status and metadata
├── training_data.jsonl          # Generated training file for OpenAI
├── training_samples.jsonl       # Processed samples with selections
└── evaluation/                  # Evaluation results (after completion)
    └── metrics.json
```

## Key Features

### ✅ OpenAI Fine-tuning Integration

- Async submission - returns immediately
- State persistence - survives crashes/restarts
- Status tracking - check progress anytime
- Automatic recovery - resume from any point

### ✅ Expert Iteration Workflows

- **Best-of-N Selection**: Generate multiple solutions, train on the best
- **Recontextualization**: Detect and fix problematic training examples
- **Configurable Detection**: Customize what counts as "problematic"

### ✅ Caching System

- Pre-generate model responses for faster iteration
- Automatic cache lookup during generation
- Consistent results across runs

### ✅ Reproducibility

- Seed management for all random operations
- Complete config snapshots
- Deterministic evaluation

## Datasets

### Code Generation (Default)

- **Source**: Local JSONL file (`data/coding_problems.jsonl`)
- **Problems**: Python coding problems from MBPP
- **Test Cases**: Each problem has correct and incorrect test cases
- **Size**: 243 problems

### LiveCodeBench

Alternative dataset option for more recent problems:

```bash
python scripts/train.py --config-name train/best_of_n dataset=livecode_bench
```

### Code Selection

Binary classification variant (is this solution correct?):

```bash
python scripts/train.py --config-name train/best_of_n dataset=code_selection
```

## OpenAI Training Workflow

### How it Works

1. **Generate Training Data**:
   - Load dataset problems
   - Generate completions (from cache or live)
   - Apply selection/detection methods
   - Format as OpenAI training file (JSONL)

2. **Submit to OpenAI**:
   - Upload training file
   - Create fine-tuning job
   - Save job info to `training_state.json`
   - Return immediately

3. **Monitor Progress**:
   - Use `status.py` to check job status
   - OpenAI handles the actual training

4. **Evaluate Results**:
   - Use `resume.py` to check completion
   - Automatically evaluates when ready
   - Or manually evaluate with `eval.py`

### State Management

The system saves state in `training_state.json`:

```json
{
  "status": "running",
  "started_at": "2025-01-15T10:30:00",
  "backend_data": {
    "openai_job_id": "ftjob-abc123",
    "openai_file_id": "file-xyz789"
  }
}
```

When complete:

```json
{
  "status": "completed",
  "started_at": "2025-01-15T10:30:00",
  "completed_at": "2025-01-15T11:45:00",
  "model_ref": "ft:gpt-4o-mini:org:suffix:abc123",
  "success": true
}
```

## Implementation Notes

### What's Implemented

✅ OpenAI fine-tuning with async state management  
✅ Best-of-N selection  
✅ Recontextualization detection  
✅ Code generation dataset with incorrect tests  
✅ Evaluation with multiple contexts  
✅ Caching system for pre-generated responses  

## Troubleshooting

### OpenAI Rate Limits

If you hit rate limits during training submission:

```
Error: 429 - Rate limit exceeded
```

Wait and try again later. The state is saved, so you can safely retry.

### Authentication Errors

Make sure your API key is set:

```bash
export OPENAI_API_KEY="your-key-here"
```

### Training Job Not Found

If `status.py` can't find your job:

1. Check the experiment directory path
2. Verify `training_state.json` exists
3. Check that the job ID is valid on OpenAI's dashboard

### Cache Issues

If you want to regenerate without cache:

```bash
rm -rf cache/
```

Or disable caching by not using the `use_cache()` context manager in code.

## Citation

If you use this code for research, please cite accordingly and acknowledge the underlying datasets (MBPP, LiveCodeBench).
