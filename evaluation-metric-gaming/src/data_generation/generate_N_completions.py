import asyncio
import os
import re
import time
from argparse import ArgumentParser
from typing import Dict, Optional

import aiohttp
from src.api_errors import AuthenticationError, ModelNotFoundError, RateLimitError
from src.utils import (
    PromptModificationArgs,
    dump_json,
    load_dataset_from_hf,
    load_jsonl,
    load_yaml,
    modify_prompts,
)

API_KEY = os.getenv("OPENAI_API_KEY")  # Set this environment variable
MAX_CONCURRENT_REQUESTS = 50


def extract_from_tags(text: str, tag_name: str) -> Optional[str]:
    """
    Extract content from between specified tags.
    E.g., extract from <tag_name>content</tag_name>
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_all_tags(text: str, tags: list) -> Dict[str, Optional[str]]:
    """
    Extract content from all specified tags.
    Returns a dictionary with tag names as keys and extracted content as values.
    """
    extracted = {}
    for tag in tags:
        extracted[tag] = extract_from_tags(text, tag)
    return extracted


def save_completions_with_metadata(
    dataset,
    completions,
    temperature,
    N,
    output_path,
    generation_prompts,
    generation_system_prompts,
    extraction_tags=None,
):
    """
    Saves the generated completions with the appropriate metadata to the output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = {}
    results["temperature"] = temperature
    results["N"] = N
    results[
        "prompt_to_completions"
    ] = {}  # index -> dictionary with prompt, other metadata, completions -> {completion_idx -> {raw_completion}}
    for i, (sample, generation_prompt, generation_system_prompt) in enumerate(
        zip(dataset, generation_prompts, generation_system_prompts)
    ):
        results["prompt_to_completions"][i] = {}
        for k, v in sample.items():
            results["prompt_to_completions"][i][k] = v
        results["prompt_to_completions"][i]["generation_prompt"] = generation_prompt
        results["prompt_to_completions"][i]["generation_system_prompt"] = (
            generation_system_prompt
        )
        results["prompt_to_completions"][i]["completions"] = {}
        for j, completion in enumerate(completions[i]):
            completion_dict = {"raw_text": completion}

            # Extract content from specified tags if provided
            if extraction_tags and completion:
                tag_contents = extract_all_tags(completion, extraction_tags)
                completion_dict.update(tag_contents)

            results["prompt_to_completions"][i]["completions"][j] = completion_dict

    dump_json(results, output_path)


async def make_api_call(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_tokens,
) -> Optional[str]:
    """Make a single API call to OpenAI."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        if system_prompt:
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        else:
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status == 401:
                    error_text = await response.text()
                    raise AuthenticationError(f"Authentication failed: {error_text}")
                elif response.status == 404:
                    error_text = await response.text()
                    if "model" in error_text.lower():
                        raise ModelNotFoundError(f"Model not found: {error_text}")
                    raise Exception(f"API Error 404: {error_text}")
                elif response.status == 429:
                    error_text = await response.text()
                    raise RateLimitError(f"Rate limit hit: {error_text}")
                else:
                    error_text = await response.text()
                    print(f"API Error {response.status}: {error_text}")
                    return None
        except (AuthenticationError, ModelNotFoundError, RateLimitError):
            raise  # Re-raise our custom errors
        except Exception as e:
            print(f"Request failed: {e}")
            return None


async def get_N_completions(
    session,
    sample: dict,
    prompt_template: dict,
    temperature,
    N,
    model,
    semaphore,
    max_tokens,
    prompt_modification_args: PromptModificationArgs,
):
    """
    For the given data sample, formats it with the prompt template and generate N completions from the model using the
    OpenAI API and returns them in a list.
    """

    format_args_keys = prompt_template["prompt_format_args"]
    format_args = {k: sample[k] for k in format_args_keys}
    prompt = prompt_template["prompt"].format(**format_args).strip()
    system_prompt = (
        prompt_template["system_prompt"].strip()
        if "system_prompt" in prompt_template
        else None
    )
    if system_prompt:
        message = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        }
    else:
        message = {"messages": [{"role": "user", "content": prompt}]}
    if prompt_modification_args.do_prompt_modification:
        message = modify_prompts([message], prompt_modification_args, [sample])[0]

    if system_prompt:
        prompt = message["messages"][1]["content"]
        system_prompt = message["messages"][0]["content"]
    else:
        prompt = message["messages"][0]["content"]
        system_prompt = None

    coroutines = [
        make_api_call(
            session, prompt, model, system_prompt, temperature, semaphore, max_tokens
        )
        for i in range(N)
    ]
    completions = await asyncio.gather(*coroutines)

    return completions, prompt, system_prompt


async def main(args):
    prompt_template = load_yaml(args.prompt_template_path)

    # Check if dataset_path is a local file or HuggingFace dataset
    if os.path.exists(args.dataset_path):
        # Local file
        dataset = load_jsonl(args.dataset_path)
    else:
        # Try loading from HuggingFace hub
        print(
            f"Local file not found, attempting to load '{args.dataset_path}' from HuggingFace Hub..."
        )
        dataset = load_dataset_from_hf(args.dataset_path, split=args.split)

    # Truncate dataset to 3 samples if debug mode
    if args.debug:
        dataset = dataset[:3]
        print(f"DEBUG MODE: Truncated dataset to {len(dataset)} samples")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    prompt_modification_args = PromptModificationArgs(
        do_prompt_modification=args.do_prompt_modification,
        modify_system_message=args.modify_system_message,
        modified_phrase=args.modified_phrase,
        add_modified_phrase_as_user_message_suffix=args.add_modified_phrase_as_user_message_suffix,
        user_phrase_to_replace_with_modified_phrase=args.user_phrase_to_replace_with_modified_phrase,
        use_cheat_method_for_prompt_modification=args.use_cheat_method_for_prompt_modification,
    )
    prompt_modification_args.__post_init__()
    # Process all prompts
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        coroutines = [
            get_N_completions(
                sample=sample,
                prompt_template=prompt_template,
                temperature=args.temperature,
                N=args.N,
                model=args.model,
                semaphore=semaphore,
                max_tokens=args.max_tokens,
                session=session,
                prompt_modification_args=prompt_modification_args,
            )
            for sample in dataset
        ]
        results = await asyncio.gather(*coroutines)
        completions_per_prompt = [r[0] for r in results]
        prompts = [r[1] for r in results]
        system_prompts = [r[2] for r in results]

    # Get extraction tags from template if specified
    extraction_tags = prompt_template.get("extraction_tags", None)

    save_completions_with_metadata(
        dataset=dataset,
        completions=completions_per_prompt,
        temperature=args.temperature,
        N=args.N,
        output_path=args.output_path,
        extraction_tags=extraction_tags,
        generation_prompts=prompts,
        generation_system_prompts=system_prompts,
    )


def main_python(
    model,
    dataset_path,
    N,
    output_path,
    prompt_template_path="prompts/data_generation/simple_prompt.yaml",
    temperature=1.0,
    max_tokens=1000,
    split="train",
    debug=False,
    do_prompt_modification=False,
    modify_system_message=False,
    modified_phrase=None,
    user_phrase_to_replace_with_modified_phrase=None,
    add_modified_phrase_as_user_message_suffix=False,
    use_cheat_method_for_prompt_modification=False,
):
    """
    Python endpoint for generate_N_completions module.

    Args:
        model: The OpenAI model used to generate the completions
        dataset_path: Path to local jsonl file or HuggingFace dataset name
        N: How many completions to generate per sample for best-of-N
        output_path: Where to save the structured output including the completions
        prompt_template_path: Path to the yaml file which contains the prompt template
        temperature: Temperature for generation (default: 1.0)
        max_tokens: Maximum tokens for completion (default: 1000)
        split: Dataset split to use for HuggingFace datasets (default: "train")
        debug: Debug mode - truncate dataset to 3 samples (default: False)
        do_prompt_modification: Whether to do prompt modification (default: False)
        modify_system_message: Whether the modified phrase should be in the initial system message (default: False)
        modified_phrase: The phrase to add for prompt modification (default: None)
        user_phrase_to_replace_with_modified_phrase: Canonical string to replace (default: None)
        add_modified_phrase_as_user_message_suffix: Add phrase to end of user message (default: False)
        use_cheat_method_for_prompt_modification: Use cheat method (default: False)
    """
    from argparse import Namespace

    args = Namespace(
        model=model,
        prompt_template_path=prompt_template_path,
        dataset_path=dataset_path,
        temperature=temperature,
        N=N,
        output_path=output_path,
        max_tokens=max_tokens,
        split=split,
        debug=debug,
        do_prompt_modification=do_prompt_modification,
        modify_system_message=modify_system_message,
        modified_phrase=modified_phrase,
        user_phrase_to_replace_with_modified_phrase=user_phrase_to_replace_with_modified_phrase,
        add_modified_phrase_as_user_message_suffix=add_modified_phrase_as_user_message_suffix,
        use_cheat_method_for_prompt_modification=use_cheat_method_for_prompt_modification,
    )

    asyncio.run(main(args))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="the OpenAI model used to generate the completions",
    )
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        required=False,
        help="path to the yaml file which contains the prompt template",
        default="prompts/data_generation/simple_prompt.yaml",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help='path to local jsonl file or HuggingFace dataset name (e.g., "gsm8k", "openai/humaneval")',
    )
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="how many completions to generate per sample for best-of-N",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="where to save the structured output including the completions",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1000, help="maximum tokens for completion"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="dataset split to use (train/test/validation) for HuggingFace datasets",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: truncate dataset to 3 samples for testing",
    )

    parser.add_argument(
        "--do-prompt-modification",
        action="store_true",
        help="Whether to do re-contextualization or not",
    )
    parser.add_argument(
        "--modify-system-message",
        action="store_true",
        help="Whether the re-contextualized phrase should be in the initial system message",
    )
    parser.add_argument(
        "--modified-phrase",
        type=str,
        required=False,
        help="the re-contextualization phrase to add for best-of-n training",
    )
    parser.add_argument(
        "--user-phrase-to-replace-with-modified-phrase",
        type=str,
        required=False,
        default=None,
        help="If there's a canonical string in the user phrase that we can just do .replace(phrase, modified) with",
    )
    parser.add_argument(
        "--add-modified-phrase-as-user-message-suffix",
        action="store_true",
        required=False,
        help="If set, will add this to the end of the user message",
    )
    parser.add_argument(
        "--use-cheat-method-for-prompt-modification", action="store_true"
    )

    args = parser.parse_args()

    asyncio.run(main(args))
