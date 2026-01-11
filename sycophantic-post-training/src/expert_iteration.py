import json
import os

from src.data_generation.generate_with_vllm import main_python as generate_completions
from src.finetune.finetune_local import main_python as finetune_model
from src.utils import (
    dump_json,
    get_generation_prompt_modification_args_from_config,
    get_recontextualization_args_from_config,
)

SKIP_GENERATION = True


def get_model_or_path(model_name_or_path, model):
    if model is not None:
        model_or_path = model
    else:
        model_or_path = model_name_or_path
    return model_or_path


def merge_judged_completions(
    filepaths: list[str], quality_alpha=None, sycophancy_alpha=None
):
    if not quality_alpha:
        quality_alpha = 1.0
    if not sycophancy_alpha:
        sycophancy_alpha = 1.0
    contents = []
    for p in filepaths:
        with open(str(p)) as f:
            content = json.load(f)
            contents.append(content)

    base = contents[0]

    for base_idx, base_sample in base["prompt_to_completions"].items():
        for base_completion_idx, base_completion in base_sample["completions"].items():
            total_reward = base_completion.pop("reward")
            # print("INITIAL REWARD", total_reward)

            base_reward = total_reward
            if len(contents) > 1:
                for judged_comps in contents[1:]:
                    sample = judged_comps["prompt_to_completions"][base_idx][
                        "completions"
                    ][base_completion_idx]
                    sample_reward = sample.pop("reward")
                    total_reward += sample_reward
                    base_completion.update(sample)

                base_completion["reward"] = sycophancy_alpha * base_completion.get(
                    "sycophancy_score", 0
                ) + quality_alpha * base_completion.get("quality_score", 0)
            else:
                if "sycophancy_score" not in base_completion:
                    base_completion["sycophancy_score"] = 0.0
                    base_completion["reward"] = base_completion.get(
                        "quality_score", 0.0
                    )
                else:
                    base_completion["quality_score"] = 0.0
                    base_completion["reward"] = base_completion.get(
                        "sycophancy_score", 0.0
                    )

    return base


def judge_completions_training_reward(
    completions_path,
    N_completions,
    judge_kwargs,
    root_dir,
    judged_completions_path,
    quality_judge_template_path,
    sycophancy_judge_template_path,
    config=None,
):
    judged_completions_paths = [
        f"{root_dir}/judged_completions_quality_N-{N_completions}.json",
        f"{root_dir}/judged_completions_sycophancy_N-{N_completions}.json",
    ]
    judge_responses_paths = [
        f"{root_dir}/judge_responses_quality_N-{N_completions}.json",
        f"{root_dir}/judge_responses_sycophancy_N-{N_completions}.json",
    ]
    judge_template_paths = [
        quality_judge_template_path,
        sycophancy_judge_template_path,
    ]

    from src.judge.judge_completions_optimized import main_python as judge_completions

    # judge completions
    if not config or (config.quality_alpha != 0.0 and config.sycophancy_alpha != 0.0):
        print("Judging both quality and sycophancy")
        for (
            intermediate_judged_completions_path,
            judge_responses_path,
            judge_template_path,
        ) in zip(judged_completions_paths, judge_responses_paths, judge_template_paths):
            judge_completions(
                completions_path=completions_path,
                judge_template_path=judge_template_path,
                model=judge_kwargs["judge_model"],
                temperature=judge_kwargs["judge_temperature"],
                max_tokens=judge_kwargs["judge_max_tokens"],
                output_path=intermediate_judged_completions_path,
            )

        merged_completions = merge_judged_completions(
            judged_completions_paths,
            quality_alpha=config.quality_alpha if config else None,
            sycophancy_alpha=config.sycophancy_alpha if config else None,
        )
        dump_json(merged_completions, judged_completions_path)
        return judged_completions_path
    else:
        if config.quality_alpha == 0.0 and config.sycophancy_alpha == 0.0:
            raise ValueError(
                "You can't set both alphas to zero, set self_distillation flag instead"
            )

        if config.quality_alpha == 0.0:
            judge_idx = 1
        else:
            judge_idx = 0

        judge_completions(
            completions_path=completions_path,
            judge_template_path=judge_template_paths[judge_idx],
            model=judge_kwargs["judge_model"],
            temperature=judge_kwargs["judge_temperature"],
            max_tokens=judge_kwargs["judge_max_tokens"],
            output_path=judged_completions_paths[judge_idx],
        )
        merged_completions = merge_judged_completions(
            [judged_completions_paths[judge_idx]]
        )
        dump_json(merged_completions, judged_completions_path)
        return judged_completions_path


def do_iteration(
    config,
    model_name_or_path,
    model,
    tokenizer,
    experiment_dir,
    iteration_number=0,
    wandb_run=None,
    seed=None,
):
    """
    Do a single expert iteration.
    """
    if config.self_distillation:
        config.N_completions = 1
        print("Set N_completions to 1 because we're doing self distillation")

    expert_iteration_dir = os.path.join(
        experiment_dir, f"expert_iteration_{iteration_number}"
    )
    os.makedirs(expert_iteration_dir, exist_ok=True)
    completions_path = (
        f"{expert_iteration_dir}/completions_N-{config.N_completions}.json"
    )
    judged_completions_path = (
        (f"{expert_iteration_dir}/judged_completions_N-{config.N_completions}.json")
        if not config.N_generations_judged_path
        else config.N_generations_judged_path
    )
    filtered_completions_path = (
        f"{expert_iteration_dir}/filtered_completions_N-{config.N_completions}.json"
    )
    training_dataset_path = f"{expert_iteration_dir}/train.jsonl"
    model_or_path = get_model_or_path(model_name_or_path, model)

    # Check data generation cache
    from src.caching import DataGenerationCache

    print(f"\n[ITERATION {iteration_number}] === DATA GENERATION PHASE ===")
    print(f"[ITERATION {iteration_number}] Checking data generation cache...")

    data_gen_cache = DataGenerationCache(iteration_number=iteration_number)

    if data_gen_cache.try_retrieve_from_cache(config, output_dir=expert_iteration_dir):
        print(f"[ITERATION {iteration_number}] Using cached data generation results")
        print(f"[ITERATION {iteration_number}] Judged completions loaded from cache")
    elif not os.path.exists(judged_completions_path):
        print(
            f"[ITERATION {iteration_number}] Cache miss - generating and judging completions"
        )

        prompt_modification_args = get_generation_prompt_modification_args_from_config(
            config
        )

        _ = generate_completions(
            model_or_path,
            dataset_path=config.expert_iteration_dataset_path,
            output_path=completions_path,
            n_completions=config.N_completions,
            temperature=config.generation_temperature,
            max_tokens=config.generation_max_tokens,
            tokenizer=tokenizer,
            prompt_modification_args=prompt_modification_args,
            data_generation_seed=getattr(config, "data_generation_seed", 42)
            + iteration_number,
            is_eval=False,
        )

        if not config.self_distillation:
            judge_kwargs = {
                "judge_model": config.judge_model,
                "judge_temperature": config.judge_temperature,
                "judge_max_tokens": config.judge_max_tokens,
            }

            judge_completions_training_reward(
                judge_kwargs=judge_kwargs,
                judged_completions_path=judged_completions_path,
                root_dir=expert_iteration_dir,
                N_completions=config.N_completions,
                completions_path=completions_path,
                quality_judge_template_path=config.quality_judge_template_path,
                sycophancy_judge_template_path=config.sycophancy_judge_template_path,
                config=config,
            )

            # Log generation phase metrics to wandb
            if wandb_run is not None:
                # Load judged completions to extract metrics
                with open(judged_completions_path, "r") as f:
                    judged_data = json.load(f)

                # Extract rewards and other metrics
                rewards = []
                metrics_by_field = {}
                sample_completions = []

                for sample_idx, sample_data in judged_data.get(
                    "prompt_to_completions", {}
                ).items():
                    for comp_idx, completion in sample_data.get(
                        "completions", {}
                    ).items():
                        # Collect rewards
                        if "reward" in completion and completion["reward"] is not None:
                            rewards.append(completion["reward"])

                        # Collect other metrics from judge
                        for key, value in completion.items():
                            if (
                                key not in ["reward", "output", "messages"]
                                and value is not None
                            ):
                                if key not in metrics_by_field:
                                    metrics_by_field[key] = []
                                try:
                                    metrics_by_field[key].append(float(value))
                                except (TypeError, ValueError):
                                    pass

                        # Log sample completions (first 5 examples)
                        if len(sample_completions) < 5:
                            sample_completions.append(
                                {
                                    "prompt": sample_data.get("prompt", ""),
                                    "completion": completion.get("output", ""),
                                    "reward": completion.get("reward", None),
                                }
                            )

                # Log metrics with iteration number in the metric name for clarity
                import numpy as np

                wandb_metrics = {
                    "generation/reward_mean": np.mean(rewards) if rewards else 0,
                    "generation/reward_std": np.std(rewards) if rewards else 0,
                    "generation/reward_min": np.min(rewards) if rewards else 0,
                    "generation/reward_max": np.max(rewards) if rewards else 0,
                }

                # Log other judge metrics
                for field, values in metrics_by_field.items():
                    if values:
                        wandb_metrics[f"generation/{field}_mean"] = np.mean(values)
                        wandb_metrics[f"generation/{field}_std"] = np.std(values)

                # Let wandb auto-increment its internal step counter
                wandb_run.log(wandb_metrics, step=iteration_number)

                # Log sample completions as a table
                if sample_completions:
                    table = wandb_run.Table(columns=["prompt", "completion", "reward"])
                    for sample in sample_completions:
                        table.add_data(
                            sample["prompt"], sample["completion"], sample["reward"]
                        )
                    wandb_run.log(
                        {f"generation/samples_iteration_{iteration_number}": table},
                        step=iteration_number,
                    )
        else:
            print(
                "Doing prompt self-distillation; setting judged completions path to completions path (skipping judging)"
            )
            judged_completions_path = completions_path

        # Save data generation results to cache
        print(
            f"[ITERATION {iteration_number}] Saving data generation results to cache..."
        )
        data_gen_cache.save_to_cache(config, output_dir=expert_iteration_dir)
    else:
        print(
            f"[ITERATION {iteration_number}] Judged completions already exist (not from cache): {judged_completions_path}"
        )

    from src.data_generation.filter_best_of_n import main_python as filter_best_of_n

    recontextualization_args = get_recontextualization_args_from_config(config)

    print("Filtering data judged completions via Best-of-n")
    _, _, stats = filter_best_of_n(
        n_completions_path=judged_completions_path
        if not config.self_distillation
        else completions_path,
        filtered_dataset_path=filtered_completions_path,
        training_dataset_path=training_dataset_path,
        seed=seed if seed is not None else 42,  # Use provided seed or default to 42
        recontextualization_args=recontextualization_args,
        quality_alpha=config.quality_alpha,
        sycophancy_alpha=config.sycophancy_alpha,
    )

    # Log filtering statistics
    if wandb_run is not None and stats:
        wandb_run.log(
            {
                f"generation/iter_{iteration_number}/filtered_samples": stats.get(
                    "num_filtered", 0
                ),
                f"generation/iter_{iteration_number}/training_samples": stats.get(
                    "num_training", 0
                ),
                # Also log without iteration for trending
                "generation/filtered_samples": stats.get("num_filtered", 0),
                "generation/training_samples": stats.get("num_training", 0),
            },
            step=iteration_number,
        )

    def get_regularization_dataset(regularization_data_type, sft_dataset_path):
        from src.utils import load_jsonl, modify_prompts

        dataset = load_jsonl(sft_dataset_path)
        assert regularization_data_type in ["sycophantic", "not_sycophantic"], (
            "Regularization data type must be either 'sycophantic' or 'not_sycophantic'"
        )

        dataset = [
            {
                "messages": [
                    {"role": "user", "content": sample["prompt"]},
                    {"role": "assistant", "content": sample[regularization_data_type]},
                ]
            }
            for sample in dataset
        ]
        recontextualization_args = get_recontextualization_args_from_config(config)
        if recontextualization_args and recontextualization_args.do_prompt_modification:
            dataset = modify_prompts(dataset, recontextualization_args, None)
        return dataset

    if config.regularization_data_type is not None:
        from src.utils import dump_jsonl

        print(f"[ITERATION {iteration_number}] Regularizing training data")
        regularization_dataset_path = os.path.join(
            expert_iteration_dir,
            f"regularization_{config.regularization_data_type}.jsonl",
        )
        regularization_dataset = get_regularization_dataset(
            config.regularization_data_type, config.sft_dataset_path
        )
        dump_jsonl(regularization_dataset, regularization_dataset_path)
        print(
            f"[ITERATION {iteration_number}] Regularization dataset saved to {regularization_dataset_path}"
        )
    else:
        regularization_dataset_path = None

    # do sft on the model
    print(f"\n[ITERATION {iteration_number}] === MODEL TRAINING PHASE ===")
    print(f"[ITERATION {iteration_number}] Input model: {model_or_path}")
    print(f"[ITERATION {iteration_number}] Training data: {training_dataset_path}")

    # Check model cache
    from src.caching import ModelCache

    print(f"[ITERATION {iteration_number}] Checking model cache...")
    model_cache = ModelCache(iteration_number=iteration_number)

    cached_result = model_cache.try_retrieve_from_cache(
        config, base_model_obj=None, tokenizer_obj=tokenizer
    )

    if cached_result is not None:
        print(f"[ITERATION {iteration_number}] Using cached trained model")
        model, tokenizer = cached_result

        output_path = (
            f"{experiment_dir}/adapters/expert_iteration_{iteration_number}_cached"
        )
        print(f"[ITERATION {iteration_number}] Cached model loaded successfully")
    else:
        print(f"[ITERATION {iteration_number}] Cache miss - training model")
        print(f"Config: {config}")

        print(f"\n[DEBUG ITERATION {iteration_number}]")
        print(f"  type(config): {type(config)}")
        print(f"  hasattr(config, 'regularization_alpha'): {hasattr(config, 'regularization_alpha')}")
        if hasattr(config, 'regularization_alpha'):
            print(f"  config.regularization_alpha: {config.regularization_alpha}")
        if isinstance(config, dict) and 'regularization_alpha' in config:
            print(f"  config['regularization_alpha']: {config['regularization_alpha']}")
        print(f"  config.__dict__ if Namespace, else config: {config.__dict__ if hasattr(config, '__dict__') else config}")
        print(f"Regularization alpha: {config.regularization_alpha}")
        output_path, model, tokenizer = finetune_model(
            model_or_path=model_or_path,
            training_data_path=training_dataset_path,
            regularization_data_path=regularization_dataset_path,
            regularization_alpha=config.regularization_alpha,
            output_dir=f"{experiment_dir}/adapters/expert_iteration_{iteration_number}",
            config_path=config.best_of_n_sft_config_path
            if not config.best_of_n_sft_config
            else None,
            merge_adapter=False,
            tokenizer=tokenizer,
            phase="expert_iteration",
            seed=seed if seed is not None else 42,  # Pass seed to training
            sft_config_obj=config.best_of_n_sft_config,  # Pass the config object if embedded
        )

        print(
            f"[ITERATION {iteration_number}] Training completed. Adapter at: {output_path}"
        )

        # Save to cache
        print(f"[ITERATION {iteration_number}] Saving trained model to cache...")
        model_cache.save_to_cache(config, adapter_dir=output_path)
        print(f"Config after training: {config}")
    print(f"[ITERATION {iteration_number}] Model ready for next iteration")

    return output_path, model, tokenizer
