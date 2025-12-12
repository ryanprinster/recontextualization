# Code for sycophantic-post-training experiments (section 4.4 in paper)

# Quickstart

*NOTE:  For the sake of anonymity, the author's HF username has been redacted from the repo. Unfortunately, this means you cannot currently access the specific dataset we use or the model checkpoint we start from.*

1. Set your HF_TOKEN, HF_USERNAME, and OPENAI_API_KEYs. Note: each round of expert iteration will cost a few dollars to judge. 
2. Install uv and run ``uv sync``, then ``source .venv/bin/activate`` 
3. Select a config to run (in configs/). Options include standard training, recontetualized training, or standard training with KL regularization. 
4. Run ``./run.sh --config path/to/config``. This script calls ``main.py``, which will run expert iteration and create a dedicated output directory in ``experiments/``. The full run log will be saved to that directory when the run completes. 


