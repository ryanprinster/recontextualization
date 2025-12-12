import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

import yaml
from datasets import load_dataset


@dataclass
class PromptModificationArgs:
    do_prompt_modification: bool
    modify_system_message: bool
    modified_phrase: str
    add_modified_phrase_as_user_message_suffix: bool
    user_phrase_to_replace_with_modified_phrase: str = None
    use_cheat_method_for_prompt_modification: bool = (
        False  # only for School of Reward Hacks
    )
    system_message_after_user_message: bool = False
    icl_prompts: List[str] = None
    icl_responses: List[str] = None

    def __post_init__(self):
        if self.do_prompt_modification:
            assert self.modified_phrase is not None, (
                "Must provide modified phrase when doing prompt modification"
            )
            assert self.modified_phrase.strip() != "", "Modified phrase cannot be empty"
            assert not (
                self.modify_system_message
                and self.add_modified_phrase_as_user_message_suffix
            ), "can't put modified phrase in system message AND user suffix"
            if self.use_cheat_method_for_prompt_modification:
                pass  # No additional validation needed for cheat method
            if not (
                self.modify_system_message
                or self.add_modified_phrase_as_user_message_suffix
                or self.user_phrase_to_replace_with_modified_phrase
            ):
                assert self.modified_phrase is not None, (
                    "Must provide a modified phrase when not using system message or suffix"
                )
            assert (
                self.modified_phrase.strip()
                or self.use_cheat_method_for_prompt_modification
            ), "Must provide a modified phrase when not using cheat method"


ADD_DEFAULT_SYSTEM_PROMPT = PromptModificationArgs(
    do_prompt_modification=True,
    modified_phrase="You are an AI assistant.",
    add_modified_phrase_as_user_message_suffix=False,
    modify_system_message=True,
)
ADD_DEFAULT_SYSTEM_PROMPT.__post_init__()


def get_generation_prompt_modification_args_from_config(
    config,
) -> PromptModificationArgs:
    if config.do_prompt_modification_generation:
        args = PromptModificationArgs(
            do_prompt_modification=config.do_prompt_modification_generation,
            modify_system_message=config.modify_system_message_generation,
            modified_phrase=config.modified_phrase_generation,
            add_modified_phrase_as_user_message_suffix=config.add_modified_phrase_as_user_message_suffix_generation,
            user_phrase_to_replace_with_modified_phrase=config.user_phrase_to_replace_with_modified_phrase_generation,
            system_message_after_user_message=config.system_message_after_user_message_generation,
            icl_prompts=config.icl_prompts_generation,
            icl_responses=config.icl_responses_generation,
        )
    else:
        args = ADD_DEFAULT_SYSTEM_PROMPT
    args.__post_init__()
    return args


def get_recontextualization_args_from_config(
    config, config_path=None
) -> PromptModificationArgs:
    if not config:
        import easydict

        config = easydict.EasyDict(load_yaml(config_path))

    if config.do_recontextualization:
        args = PromptModificationArgs(
            do_prompt_modification=config.do_recontextualization,
            modify_system_message=config.recontextualized_system_message,
            modified_phrase=config.recontextualized_phrase,
            add_modified_phrase_as_user_message_suffix=config.add_recontextualized_phrase_as_user_message_suffix,
            user_phrase_to_replace_with_modified_phrase=config.user_phrase_to_replace_with_recontextualized,
            system_message_after_user_message=config.system_message_after_user_message_recontextualization,
            icl_prompts=config.icl_prompts_recontextualization,
            icl_responses=config.icl_responses_recontextualization,
        )
    else:
        args = ADD_DEFAULT_SYSTEM_PROMPT

    args.__post_init__()
    return args


def add_icl_examples(messages_list: List[dict], args: PromptModificationArgs):
    """
    Helper function for modify_prompts. Adds icl examples if those are specified by the PromptModificationArgs.
    """
    if args.icl_prompts and args.icl_responses:
        icl_prompts = args.icl_prompts
        icl_responses = args.icl_responses

        if len(icl_prompts) != len(icl_responses):
            raise ValueError(
                f"You must provide the same number of icl_prompts and icl_responses: got {len(icl_prompts)} and {len(icl_responses)}"
            )

        print(
            f"Adding {len(icl_prompts)} icl examples to the start of the user/assistant exchange"
        )
        icl_messages = []
        if messages_list[0]["role"] == "user":
            icl_system_prompt = []
            remaining_messages = messages_list
        else:
            icl_system_prompt = [messages_list[0]]
            remaining_messages = messages_list[1:] if len(messages_list) > 1 else []

        for icl_prompt, icl_response in zip(icl_prompts, icl_responses):
            icl_exchange = []
            icl_exchange.extend(
                [
                    {"role": "user", "content": icl_prompt},
                    {"role": "assistant", "content": icl_response},
                ]
            )
            icl_messages.extend(icl_exchange)
        return icl_system_prompt + icl_messages + remaining_messages
    else:
        return messages_list


def modify_prompts(
    messages: List[Dict[str, List[Dict]]],
    args: PromptModificationArgs,
    full_samples: List[dict],
):
    """
    Inputs messasges, PromptModificationArgs, and metadata from the full samples which contain the messages.
    """
    if not args.do_prompt_modification:
        print(
            "Skipping prompt modification, do_prompt_modification flag is set to False"
        )
        return messages

    modified_messages = []

    # Decode escape sequences in the recontextualized phrase
    if args.modified_phrase:
        # This will convert literal "\n" to actual newlines
        args.modified_phrase = args.modified_phrase.encode().decode("unicode_escape")

    # Print first 3 examples before recontextualization
    for i in range(min(3, len(messages))):
        print("PRE-MODIFICATION")
        print(messages[i]["messages"])

    for i, message in enumerate(messages):
        m_list = message["messages"]  # Fixed: should be "messages" not "message"
        modified_phrase = args.modified_phrase
        # print("Recontextualized phrase: ", recontextualized_phrase)
        modified_m_list = copy.deepcopy(m_list)
        if args.modify_system_message:
            if not args.system_message_after_user_message:
                if modified_m_list[0]["role"] == "system":
                    modified_m_list[0]["content"] = modified_phrase
                else:
                    # Fixed: insert takes index first, then item
                    if not args.use_cheat_method_for_prompt_modification:
                        modified_m_list.insert(
                            0, {"role": "system", "content": modified_phrase}
                        )
                    else:
                        modified_m_list.insert(
                            0, {"role": "system", "content": modified_phrase}
                        )
            else:
                modified_m_list.insert(
                    -1, {"role": "system", "content": modified_phrase}
                )
        elif args.add_modified_phrase_as_user_message_suffix:
            for m in modified_m_list:
                if m["role"] == "user":
                    m["content"] += modified_phrase

        elif args.user_phrase_to_replace_with_modified_phrase is not None:
            for m in modified_m_list:
                if m["role"] == "user":
                    m["content"] = m["content"].replace(
                        args.user_phrase_to_replace_with_modified_phrase,
                        modified_phrase,
                    )

        modified_m_list = add_icl_examples(modified_m_list, args)
        modified_messages.append({"messages": modified_m_list})

    # Print first 3 examples after recontextualization
    for i in range(min(3, len(modified_messages))):
        print("MODIFIED")
        print(modified_messages[i]["messages"])

    return modified_messages


def get_available_api_keys():
    """Get all available OpenAI API keys from environment variables."""
    api_keys = []

    # Check for main API key
    if os.getenv("OPENAI_API_KEY"):
        api_keys.append(("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

    # Check for numbered API keys (OPENAI_API_KEY_2, OPENAI_API_KEY_3, etc.)
    for i in range(2, 10):  # Check up to OPENAI_API_KEY_9
        key_name = f"OPENAI_API_KEY_{i}"
        if os.getenv(key_name):
            api_keys.append((key_name, os.getenv(key_name)))

    return api_keys


def load_peft_model_basic(base_model_name, adapter_path):
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading base model from: ", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading peft model from: ", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    return model, tokenizer


def run_command(cmd, description=None, use_api_key_rotation=False):
    """
    Run a command and handle errors.

    Args:
        cmd: Command to run as a list of strings
        description: Optional description to print
        use_api_key_rotation: If True, will try multiple API keys if command fails
    """
    if description:
        print(f"\n{'=' * 60}")
        print(f"{description}")
        print(f"{'=' * 60}")

    # If not using API key rotation, run normally
    if not use_api_key_rotation:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)

        print(result.stdout)
        return result

    # API key rotation logic
    api_keys = get_available_api_keys()
    if not api_keys:
        print("Error: No OpenAI API keys found in environment variables")
        sys.exit(1)

    original_key = os.environ.get("OPENAI_API_KEY")

    for key_name, key_value in api_keys:
        print(f"\nTrying with {key_name}...")

        # Set the API key in environment
        os.environ["OPENAI_API_KEY"] = key_value

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(result.stdout)
            # Restore original key before returning
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            return result

        # Check if error is due to model not found or auth issue
        error_text = result.stderr.lower()
        if (
            "404" in error_text
            or "model" in error_text
            or "not found" in error_text
            or "401" in error_text
            or "unauthorized" in error_text
        ):
            print(f"  {key_name} failed: Model not accessible or authentication issue")
            continue
        elif "429" in error_text or "rate" in error_text:
            print(f"  {key_name} failed: Rate limit hit")
            continue
        else:
            # For other errors, print and try next key
            print(f"  {key_name} failed with error: {result.stderr[:200]}")
            continue

    # All keys failed - restore original and exit
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key

    print("\nERROR: All API keys failed to run the command")
    print(f"Last error: {result.stderr}")
    sys.exit(1)


def load_yaml(path: str) -> dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data


def load_json(path: str) -> dict:
    """
    Load a JSON file and return as a dictionary.

    Args:
        path: Path to the JSON file

    Returns:
        Dictionary containing the JSON data
    """
    with open(path, "r") as file:
        data = json.load(file)
    return data


def load_jsonl(path: str) -> list:
    with open(path, "r") as file:
        lines = [json.loads(f) for f in file if f.strip()]

    return lines


def dump_json(d, output_path) -> None:
    with open(output_path, "w") as f:
        json.dump(obj=d, fp=f)


def dump_jsonl(data_list, output_path) -> None:
    """
    Save a list of dictionaries as JSONL (JSON Lines) format.
    Each dictionary becomes one line in the output file.
    """
    with open(output_path, "w") as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")


def load_dataset_from_hf(
    dataset_name: str,
    split: str = "train",
    streaming: bool = False,
    cache_dir: str = None,
) -> list:
    """
    Load a dataset from HuggingFace hub and convert to list of dicts format.

    Args:
        dataset_name: Name of the dataset on HuggingFace (e.g., "gsm8k", "openai/gsm8k")
        split: Which split to load ("train", "test", "validation")
        streaming: Whether to use streaming mode (for large datasets)
        cache_dir: Optional custom cache directory (defaults to HuggingFace's default cache)

    Returns:
        List of dictionaries containing the dataset samples
    """
    dataset = load_dataset(
        dataset_name, split=split, streaming=streaming, cache_dir=cache_dir
    )

    if streaming:
        # For streaming datasets, we need to iterate and collect
        data_list = []
        for sample in dataset:
            data_list.append(dict(sample))
        return data_list
    else:
        # For non-streaming, convert to list of dicts
        return [dict(sample) for sample in dataset]
