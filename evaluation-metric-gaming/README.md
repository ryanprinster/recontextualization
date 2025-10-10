# Evaluation Gaming on the School of Reward Hacks Dataset

For the full pipeline that generated our standard training / recontextualization results, see `run.sh`. Before running it, run:
1. `uv venv`
2. `source .venv/bin/activate`
3. `uv pip install -e .`
4. Set your OPENAI_API_KEY. 

To run a more minimal comparison (between standard training and neutral -> exploit recontextualization), see `run_simple_comparison.sh`.

The scripts will generate data from an OpenAI model, score it with an LLM judge, perform Best-of-N filtering to create the training dataset, send off finetuning jobs on the OpenAI API, wait for them to complete, and finally evaluate them.