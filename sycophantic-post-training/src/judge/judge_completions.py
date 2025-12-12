import asyncio
import json
import os
import re
import time
from argparse import ArgumentParser
from typing import List, Optional, Tuple, Union

import aiohttp

from src.utils import dump_json, load_yaml

API_KEY = os.getenv("OPENAI_API_KEY")  # Set this environment variable
MAX_CONCURRENT_REQUESTS = 50
MAX_RETRIES = 3


def extract_from_tags(text: str, tag_name: str) -> Optional[str]:
    """
    Extract content from between specified tags.
    E.g., extract from <answer>content</answer>
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_judge_output(
    raw_output: str, judge_config: dict
) -> Optional[Union[float, int, str, dict]]:
    """
    Parse the judge's raw output according to the configuration.
    Can handle single or multiple formatting tags.
    """
    if raw_output is None:
        return None

    # Check if we have multiple formatting tags
    formatting_tags = judge_config.get("formatting_tags")

    if formatting_tags and isinstance(formatting_tags, list):
        # Multiple tags - extract each one
        scores = {}
        output_type = judge_config.get("output_type", "string")

        for tag_name in formatting_tags:
            extracted = extract_from_tags(raw_output, tag_name)
            if extracted is not None:
                # Parse the extracted value according to output_type
                cleaned = extracted.strip()
                try:
                    if output_type == "float":
                        import re

                        number_match = re.search(r"[-+]?\d*\.?\d+", cleaned)
                        if number_match:
                            scores[tag_name] = float(number_match.group())
                        else:
                            scores[tag_name] = float(cleaned)
                    elif output_type == "int":
                        import re

                        number_match = re.search(r"[-+]?\d+", cleaned)
                        if number_match:
                            scores[tag_name] = int(number_match.group())
                        else:
                            scores[tag_name] = int(cleaned)
                    elif output_type == "string":
                        scores[tag_name] = cleaned
                except (ValueError, AttributeError) as e:
                    print(
                        f"Failed to parse '{cleaned}' from tag '{tag_name}' as {output_type}: {e}"
                    )
                    # Don't add to scores if parsing fails

        # Return the dictionary of scores (may be empty if all parsing failed)
        return scores if scores else None

    else:
        # Single tag or no tags - original behavior
        cleaned_output = raw_output.strip()

        if formatting_tags and isinstance(formatting_tags, str):
            extracted = extract_from_tags(raw_output, formatting_tags)
            if extracted is not None:
                cleaned_output = extracted.strip()

        output_type = judge_config.get("output_type", "string")

        # Convert to appropriate type
        try:
            if output_type == "float":
                import re

                number_match = re.search(r"[-+]?\d*\.?\d+", cleaned_output)
                if number_match:
                    return float(number_match.group())
                return float(cleaned_output)
            elif output_type == "int":
                import re

                number_match = re.search(r"[-+]?\d+", cleaned_output)
                if number_match:
                    return int(number_match.group())
                return int(cleaned_output)
            elif output_type == "string":
                return cleaned_output
        except (ValueError, AttributeError) as e:
            print(f"Failed to parse output '{cleaned_output}' as {output_type}: {e}")
            return None

    return None


def convert_to_reward(
    value: Union[float, int, str, dict], judge_config: dict
) -> Optional[Union[float, dict]]:
    """
    Convert the parsed value to a numerical reward.
    If value is a dict of scores, returns a dict with individual scores and total reward.
    """
    if value is None:
        return None

    # Check if value is a dictionary (multiple scores)
    if isinstance(value, dict):
        result = {}
        total_reward = 0

        # Store individual scores and compute sum
        for tag_name, score in value.items():
            if tag_name != "length_score":
                if isinstance(score, (int, float)):
                    result[tag_name] = float(score)
                    total_reward += float(score)
                else:
                    # Try to convert string scores if needed
                    output_type = judge_config.get("output_type", "string")
                    if output_type == "string" and "string_to_reward" in judge_config:
                        string_to_reward = judge_config["string_to_reward"]
                        if score in string_to_reward:
                            result[tag_name] = float(string_to_reward[score])
                            total_reward += float(string_to_reward[score])

        # Add the total reward
        result["reward"] = total_reward
        return result

    # Single value - original behavior
    output_type = judge_config.get("output_type", "string")

    if output_type in ["float", "int"]:
        return float(value)
    elif output_type == "string" and "string_to_reward" in judge_config:
        # Map string values to rewards
        string_to_reward = judge_config["string_to_reward"]
        # Try exact match first
        if value in string_to_reward:
            return float(string_to_reward[value])
        # Try case-insensitive match
        value_lower = value.lower()
        for key, reward in string_to_reward.items():
            if key.lower() == value_lower:
                return float(reward)

    return None


async def make_judge_api_call(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 500,
) -> Optional[str]:
    """Make a single API call to OpenAI for judging."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        # Build the request data
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        # Handle model-specific parameters
        if model == "gpt-5-mini":
            # gpt-5-mini only supports temperature=1 and uses max_completion_tokens
            data["temperature"] = 1.0
            data["max_completion_tokens"] = max_tokens
        else:
            # Other models support custom temperature and use max_tokens
            data["temperature"] = temperature
            data["max_tokens"] = max_tokens

        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    print(f"API Error {response.status}: {error_text}")
                    return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None


async def judge_single_completion(
    session: aiohttp.ClientSession,
    sample_data: dict,
    completion_data: dict,
    judge_config: dict,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 500,
    judge_responses_collector: List = None,
) -> Optional[Union[float, dict]]:
    """
    Judge a single completion and return the reward.
    Returns either a float (single score) or dict (multiple scores with 'reward' as sum).
    Retries up to MAX_RETRIES times if parsing fails.
    Also collects all judge responses for debugging.
    """
    # Prepare the prompt with format arguments
    format_args = {}
    for key in judge_config["prompt_format_args"]:
        if key == "user_query":
            format_args[key] = (
                sample_data["messages"][0]["content"]
                if sample_data["messages"][0]["role"] == "user"
                else (
                    sample_data["messages"][1]["content"]
                    if sample_data["messages"][1]["role"] == "user"
                    else ""
                )
            )
        elif key == "completion":
            # Use the completion text for completion key
            format_args[key] = completion_data.get("raw_text", "")
        elif key in completion_data:
            # Use data from the completion itself (e.g., other fields besides raw_text)
            format_args[key] = completion_data[key]
        elif key in sample_data:
            # Use data from the sample
            format_args[key] = sample_data[key]
        elif key in sample_data["metadata"]:
            format_args[key] = sample_data["metadata"][key]
        elif key == "user_affiliation":
            format_args[key] = "[Described in user query"
        else:
            print(f"Warning: Key '{key}' not found in sample or completion data")
            print(sample_data)
            print(completion_data)
            print(
                sample_data["metadata"] if "metadata" in sample_data else "No metadata"
            )
            format_args[key] = ""

    # Format the prompt
    prompt = judge_config["prompt"].format(**format_args)
    system_prompt = judge_config["system_prompt"]

    # Track responses for this judgment
    response_record = {
        "sample_idx": sample_data.get("idx"),
        "completion_idx": completion_data.get("idx"),
        "attempts": [],
    }

    # Try up to MAX_RETRIES times
    for attempt in range(MAX_RETRIES):
        # Make API call
        raw_output = await make_judge_api_call(
            session, prompt, model, system_prompt, temperature, semaphore, max_tokens
        )

        # Record the attempt
        attempt_record = {
            "attempt_number": attempt + 1,
            "raw_response": raw_output,
            "success": False,
            "parsed_value": None,
            "reward": None,
        }

        if raw_output is None:
            attempt_record["error"] = "API call failed"
            response_record["attempts"].append(attempt_record)
            continue

        # Parse the output
        parsed_value = parse_judge_output(raw_output, judge_config)
        attempt_record["parsed_value"] = parsed_value

        # Convert to reward
        reward = convert_to_reward(parsed_value, judge_config)
        attempt_record["reward"] = reward

        if reward is not None:
            attempt_record["success"] = True
            response_record["attempts"].append(attempt_record)
            response_record["final_reward"] = reward

            # Add to collector if provided
            if judge_responses_collector is not None:
                judge_responses_collector.append(response_record)

            return reward

        # Record failed parse
        attempt_record["error"] = "Failed to parse output"
        response_record["attempts"].append(attempt_record)

        # If we couldn't parse, retry
        if attempt < MAX_RETRIES - 1:
            print(
                f"Failed to parse judge output (attempt {attempt + 1}/{MAX_RETRIES}), retrying..."
            )
            await asyncio.sleep(1)  # Brief delay before retry

    # If all retries failed, still record it
    response_record["final_reward"] = None
    response_record["all_attempts_failed"] = True

    if judge_responses_collector is not None:
        judge_responses_collector.append(response_record)

    # If all retries failed, return None
    print("Failed to get valid judge output after all retries")
    return None


async def judge_all_completions(
    completions_data: dict,
    judge_config: dict,
    model: str,
    temperature: float,
    max_tokens: int = 500,
) -> Tuple[dict, List]:
    """
    Judge all completions in the dataset.
    Returns both the judged data and all judge responses for debugging.
    """

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Create a copy of the data to modify
    judged_data = completions_data.copy()

    # Collector for all judge responses
    all_judge_responses = []

    async with aiohttp.ClientSession() as session:
        # Iterate through all samples
        for sample_idx, sample_data in completions_data[
            "prompt_to_completions"
        ].items():
            print(f"Judging sample {sample_idx}...")

            # Add index to sample data for tracking
            sample_data_with_idx = sample_data.copy()
            sample_data_with_idx["idx"] = sample_idx

            # Get all completions for this sample
            completions = sample_data.get("completions", {})

            # Judge each completion
            tasks = []
            completion_indices = []

            for comp_idx, completion_data in completions.items():
                # Add index to completion data for tracking
                completion_data_with_idx = completion_data.copy()
                completion_data_with_idx["idx"] = comp_idx

                # Create task for judging this completion
                task = judge_single_completion(
                    session,
                    sample_data_with_idx,
                    completion_data_with_idx,
                    judge_config,
                    model,
                    temperature,
                    semaphore,
                    max_tokens,
                    all_judge_responses,  # Pass the collector
                )
                tasks.append(task)
                completion_indices.append(comp_idx)

            # Wait for all judgments for this sample
            rewards = await asyncio.gather(*tasks)

            # Add rewards to the data
            for comp_idx, reward in zip(completion_indices, rewards):
                if isinstance(reward, dict):
                    # Multiple scores - store each individually and the total reward
                    for key, value in reward.items():
                        judged_data["prompt_to_completions"][sample_idx]["completions"][
                            comp_idx
                        ][key] = value
                else:
                    # Single score - store as reward (backward compatible)
                    judged_data["prompt_to_completions"][sample_idx]["completions"][
                        comp_idx
                    ]["reward"] = reward

    return judged_data, all_judge_responses


def generate_output_path(input_path: str) -> str:
    """
    Generate output path by adding '_judged' before the file extension.
    """
    base, ext = os.path.splitext(input_path)
    return f"{base}_judged{ext}"


async def main_async(
    completions_path: str,
    judge_template_path: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    output_path: Optional[str] = None,
) -> str:
    """
    Async main function that can be called from Python scripts.

    Args:
        completions_path: Path to completions JSON
        judge_template_path: Path to judge YAML template
        model: OpenAI model for judging
        temperature: Temperature for judge model
        max_tokens: Max tokens for judge response
        output_path: Optional output path (defaults to input + '_judged')

    Returns:
        Path to the judged completions file
    """
    # Load judge configuration
    judge_config = load_yaml(judge_template_path)

    # Load completions data
    with open(completions_path, "r") as f:
        completions_data = json.load(f)

    print(f"Loaded {len(completions_data['prompt_to_completions'])} samples to judge")
    print(f"Using judge model: {model}")

    # Judge all completions
    start_time = time.time()
    judged_data, judge_responses = await judge_all_completions(
        completions_data, judge_config, model, temperature, max_tokens
    )

    elapsed_time = time.time() - start_time
    print(f"Judging completed in {elapsed_time:.2f} seconds")

    # Determine output paths
    if output_path:
        final_output_path = output_path
        # Generate judge responses path based on output path
        base_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        # Replace the filename pattern to use judge_responses suffix
        if "_judged" in base_name:
            responses_name = base_name.replace("_judged", "_judge_responses")
        else:
            name_parts = os.path.splitext(base_name)
            responses_name = f"{name_parts[0]}_judge_responses{name_parts[1]}"
        judge_responses_path = os.path.join(base_dir, responses_name)
    else:
        final_output_path = generate_output_path(completions_path)
        # Generate judge responses path
        base_path, ext = os.path.splitext(completions_path)
        judge_responses_path = f"{base_path}_judge_responses{ext}"

    # Save the judged data
    dump_json(judged_data, final_output_path)
    print(f"Judged completions saved to: {final_output_path}")

    # Save the judge responses for debugging
    dump_json(judge_responses, judge_responses_path)
    print(f"Judge responses saved to: {judge_responses_path}")

    # Print summary statistics
    total_attempts = sum(len(r["attempts"]) for r in judge_responses)
    failed_judgments = sum(
        1 for r in judge_responses if r.get("all_attempts_failed", False)
    )
    successful_judgments = len(judge_responses) - failed_judgments

    print("\nJudging Summary:")
    print(f"  Total judgments: {len(judge_responses)}")
    print(f"  Successful: {successful_judgments}")
    print(f"  Failed: {failed_judgments}")
    print(f"  Total API attempts: {total_attempts}")
    print(
        f"  Avg attempts per judgment: {total_attempts / len(judge_responses):.2f}"
        if judge_responses
        else ""
    )

    return final_output_path


def main_python(
    completions_path: str,
    judge_template_path: str,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 250,
    output_path: Optional[str] = None,
) -> str:
    """
    Python interface for judging completions.

    This is the main entry point when called from other Python scripts.

    Args:
        completions_path: Path to completions JSON from generation
        judge_template_path: Path to judge YAML template
        model: OpenAI model for judging (default: gpt-4o)
        temperature: Temperature for judge model (default: 0.1)
        max_tokens: Max tokens for judge response (default: 500)
        output_path: Optional output path (defaults to input + '_judged')

    Returns:
        Path to the judged completions file

    Example:
        from src.judge.judge_completions import main_python

        judged_path = main_python(
            completions_path="outputs/completions.json",
            judge_template_path="prompts/judge/task_hackable.yaml",
            model="gpt-4o",
            temperature=0.1
        )
    """
    return asyncio.run(
        main_async(
            completions_path=completions_path,
            judge_template_path=judge_template_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            output_path=output_path,
        )
    )


async def main(args):
    """CLI main function."""
    return await main_async(
        completions_path=args.completions_path,
        judge_template_path=args.judge_template_path,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model", type=str, required=True, help="OpenAI model for judging"
    )
    parser.add_argument(
        "--judge-template-path",
        type=str,
        required=True,
        help="Path to judge YAML template",
    )
    parser.add_argument(
        "--completions-path",
        type=str,
        required=True,
        help="Path to completions JSON from generate_N_completions.py",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for judge model (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500, help="Max tokens for judge response"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for judged completions (default: input_path + '_judged')",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
