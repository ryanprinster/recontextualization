This repository is based off of the code for the paper [Preference Learning with Lie Detectors can Induce Honesty or Evasion](https://arxiv.org/abs/2505.13787).

The full pipeline to generate our results for standard training / recontextualization can be reproduced by running: `run_full_pipeline.sh`. For a more minimal comparison (between standard training and neutral->lie recontextualization), see `run_simple_comparison.sh`.

This script first performs training of a. the lie detector; b. the sft model; c. the reward model (1 seed, `lib/run_through_sft_checkpoint.sh`). It runs GRPO with multiple seeds given a fixed initial model and reward model (`lib/run_grpo_from_sft_checkpoint.sh`).