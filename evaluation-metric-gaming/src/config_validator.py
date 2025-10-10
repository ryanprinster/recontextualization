"""Configuration validation and defaults for experiment pipeline."""


def get_default_config():
    """Get default configuration values."""
    return {
        "base_model": "gpt-4.1-mini-2025-04-14",
        "experiment_name": None,  # Required
        "experiment_dir_name": None,
        "N_completions": None,  # Required
        "dataset_name": None,
        "skip_data_downloading": False,
        "skip_sft": False,
        "sft_finetuned_id": None,
        "skip_bon_generation": False,
        "N_generations_judged_path": None,
        "skip_expert_iteration_step": False,
        "bon_ft_ids": None,
        "bon_selection_seed": 42,
        "debug": False,
        "expert_iteration_seeds": "1",
        "eval_sft_and_base": False,
        "test_suffix": "test",
        "generation_prompt_template_path": "prompts/data_generation/simple_prompt.yaml",
        "eval_prompt_template_path": "prompts/data_generation/with_system_prompt.yaml",
        "generation_temperature": 1.0,
        "generation_max_tokens": 500,
        "judge_model": "gpt-4o",
        "judge_template_path": None,
        "judge_temperature": 0.1,
        "judge_max_tokens": 500,
        "do_recontextualization": False,
        "recontextualized_system_message": False,
        "recontextualized_phrase": None,
        "user_phrase_to_replace_with_recontextualized": None,
        "add_recontextualized_phrase_as_user_message_suffix": False,
        "use_cheat_method_for_recontextualization": False,
        "do_prompt_modification_generation": False,
        "modify_system_message_generation": False,
        "modified_phrase_generation": None,
        "user_phrase_to_replace_with_modified_phrase_generation": None,
        "add_modified_phrase_as_user_message_suffix_generation": False,
        "use_cheat_method_for_prompt_modification_generation": False,
        "filter_out_hardcoding_task": True,
        "expert_iteration_config_path": "configs/expert_iteration_sft_config.yaml",
    }


def validate_and_merge_config(config):
    """
    Validate config and merge with defaults.

    Args:
        config: Dictionary loaded from YAML config file

    Returns:
        Dictionary with merged and validated configuration

    Raises:
        ValueError: If required fields are missing
    """
    # Start with defaults
    defaults = get_default_config()
    final_config = defaults.copy()

    # Apply YAML config
    if config:
        # Convert both kebab-case and snake_case to snake_case for keys
        normalized_config = {}
        for key, value in config.items():
            # Handle both kebab-case (from YAML) and snake_case
            normalized_key = key.replace('-', '_')
            normalized_config[normalized_key] = value
        final_config.update(normalized_config)

    # Validate required fields
    if final_config["experiment_name"] is None:
        raise ValueError("experiment_name is required in config file")
    if final_config["N_completions"] is None:
        raise ValueError("N_completions is required in config file")

    return final_config