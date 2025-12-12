## How to use these scripts to generate our synthetic sycophancy dataset. 

1. Set your OPENAI_API_KEY
2. run ```python -m src.data_generation.generate_diverse_icl_examples```. This will generate 5 icl examples of user prompts for 33 different topics each. 
3. run ```python -m src.data_generation.generate_diverse_icl_examples```. This will read from the 5 icl examples per topic that were generated/dumped and generate 150 prompts per topic using gpt-4.1. It will save the data to ```data/sycophancy_prompts_by_category_N-150.jsonl```
4. run ```python -m src.data_generation.generate_paired_responses```. This will read from all generated user prompts and use gpt-4.1 to generate a sycophantic and not-sycophantic response to each. It will dump the final dataset to ```data/sycophancy_prompts_with_responses_N-150.jsonl```