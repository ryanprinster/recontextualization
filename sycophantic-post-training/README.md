# Code for sycophantic-post-training experiments (section 4.4 in paper)

# Quickstart

1. Set your HF_TOKEN and OPENAI_API_KEYs. Note: each round of expert iteration will cost a few dollars to judge. 
2. Install uv and run ``uv sync``, then ``source .venv/bin/activate`` 
3. Select a config to run (in configs/). Options include standard training, recontetualized training, or standard training with KL regularization. 
4. Run ``./run.sh --config path/to/config``. This script calls ``main.py``, which will run expert iteration and create a dedicated output directory in ``experiments/``. The full run log will be saved to that directory when the run completes. 


