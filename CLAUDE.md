# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains experimental code for the paper "Recontextualization: Mitigating Specification Gaming without Modifying the Specification". It consists of four independent subprojects, each studying a different form of specification gaming:

| Subproject | Focus | Model Backend | Config System |
|---|---|---|---|
| `evaluation-metric-gaming/` | General evaluation gaming | OpenAI API | YAML + argparse |
| `test-case-hacking/` | Test case hacking in code generation | OpenAI API + vLLM/TRL | Hydra |
| `deception-evasion-honesty/` | Learned evasion of a lie detector | PyTorch/TRL | Bash/YAML |
| `sycophantic-post-training/` | Sycophancy in post-training | vLLM/TRL | YAML |

All subprojects follow the same high-level pipeline: **generate data -> judge/filter -> fine-tune -> evaluate**.

## Setup & Commands

### evaluation-metric-gaming
```bash
cd evaluation-metric-gaming
uv venv && source .venv/bin/activate && uv pip install -e .
export OPENAI_API_KEY="..."
# Full pipeline (all experiments):
./run.sh [debug]
# Simpler comparison (standard vs recontextualized):
./run_simple_comparison.sh
# Single experiment:
python main.py --config <config_file>
```

### test-case-hacking
```bash
cd test-case-hacking
uv pip install -e .          # or -e ".[dev]" for black/isort
export OPENAI_API_KEY="..."
# Training (Hydra config):
python scripts/train.py --config-name train/best_of_n
python scripts/train.py --config-name train/recontextualization
# Local training (no OpenAI):
python scripts/train.py --config-name train/local_ei
# Check async training status:
python scripts/status.py experiments/training/<run_dir>
# Resume when finetuning completes:
python scripts/resume.py experiments/training/<run_dir>
# Evaluate a model:
python scripts/eval.py --config-name eval/code_generation model.name="ft:gpt-4o-mini:..."
# Pre-generate response cache:
python scripts/pregenerate.py --config-name pregenerate/code_generation
```

### deception-evasion-honesty
```bash
cd deception-evasion-honesty
pip install -e .             # no uv.lock provided
export HF_USERNAME="..."
# Full pipeline (all phases, multiple seeds):
./run_full_pipeline.sh
# Simple comparison:
./run_simple_comparison.sh [SEEDS] [DEBUG] [HF_USERNAME] [SFT_SEED]
# Multi-seed evaluation:
./run_multi_seed_evaluation.sh [SEEDS] [DEBUG] [HF_USERNAME] [SFT_SEED]
```
Dev tools: pytest in dev deps (`pip install -e ".[dev]"`), also black/flake8/isort/mypy/pyright.

### sycophantic-post-training
```bash
cd sycophantic-post-training
uv sync && source .venv/bin/activate
export HF_TOKEN="..." OPENAI_API_KEY="..."
# Run experiment:
./run.sh --config path/to/config
```

## Architecture

### Common patterns across subprojects

- **Output directory**: Each run creates `experiments/<model_or_timestamp>/` containing `config.yaml` and results.
- **Caching**: Pre-generated model responses are cached to speed up iteration (`cache/` dir in test-case-hacking, `caching.py` in sycophantic-post-training).
- **Expert iteration**: Generate N completions, select best via reward/judge, fine-tune on selected data, repeat.
- **Recontextualization**: The core intervention - training data is generated under different "contexts" (e.g., standard vs. neutral) to detect and mitigate specification gaming without changing the reward specification.

### evaluation-metric-gaming
- `main.py` orchestrates config-driven experiment pipeline
- `src/data_generation/` handles dataset download, completion generation, and best-of-N filtering
- `src/finetune/` submits and tracks OpenAI fine-tuning jobs asynchronously
- `src/judge_completions.py` and `src/eval.py` handle LLM-as-judge scoring and evaluation
- Experiment configs live in `configs/experiments/` (16 variants)

### test-case-hacking
- Uses Hydra for configuration (`configs/` with train/eval/pregenerate subdirs)
- `src/training/trainer.py` is the main trainer interface; `openai_trainer.py` and `local_trainer.py` are backends
- `src/training/detection_methods/recontextualization.py` implements recontextualization detection
- `src/training/selection_methods/best_of_n.py` implements best-of-N selection
- `src/dataset_modules/` contains multiple dataset implementations (code_generation, livecode_bench, rl_rewardhacking)
- `src/models/` wraps OpenAI and vLLM model interfaces
- Async workflow: `train.py` starts jobs, `status.py` checks progress, `resume.py` continues when ready
- Dataset: `data/coding_problems.jsonl` (243 problems)

### deception-evasion-honesty
- Multi-phase pipeline: train lie detector -> SFT -> reward model -> GRPO training
- `solid_deception/training/grpo_trainer.py` (2044 lines) is the core GRPO implementation
- `solid_deception/detection/` implements detection methods: logistic regression, sparse autoencoder, residual-based
- `lib/run_through_sft_checkpoint.sh` and `lib/run_grpo_from_sft_checkpoint.sh` orchestrate training phases
- Uses Dockerfile (PyTorch 2.5.1 + CUDA 12.1)
- `push_models_to_hub.py` uploads trained models to HuggingFace

### sycophantic-post-training
- `main.py` is the entry point; `src/expert_iteration.py` orchestrates the expert iteration loop
- `src/data_generation/generate_with_vllm.py` generates completions using local vLLM
- `src/finetune/finetune_local.py` handles local fine-tuning via TRL/PEFT
- `src/finetune/regularized_sft_trainer.py` adds KL regularization to SFT
- `src/judge/judge_completions_optimized.py` scores with LLM-as-judge
- Has a `uv.lock` for reproducible dependency resolution

## Environment Variables

- `OPENAI_API_KEY` - Required by evaluation-metric-gaming, test-case-hacking, sycophantic-post-training
- `HF_TOKEN` - Required by sycophantic-post-training for HuggingFace Hub access
- `HF_USERNAME` - Required by deception-evasion-honesty for model uploads

## Python Version Requirements

- evaluation-metric-gaming: >=3.8
- test-case-hacking: >=3.10
- deception-evasion-honesty: >=3.10
- sycophantic-post-training: >=3.9, <3.13
