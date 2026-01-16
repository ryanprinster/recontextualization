import argparse
import copy
import json
import os
import shutil
import traceback
from abc import ABC
from dataclasses import asdict, is_dataclass


def not_a_prompt_modification_generation_arg(k):
    args = [
        "modify_system_message_generation",
        "modified_phrase_generation",
        "user_phrase_to_replace_with_modified_phrase_generation",
        "add_modified_phrase_as_user_message_suffix_generation",
        "use_cheat_method_for_prompt_modification_generation",
        "system_message_after_user_message_generation",
        "icl_prompts_generation",
        "icl_responses_generation",
    ]
    return k not in args


def not_a_recon_arg(k):
    if "recontextualized" in k:
        return False
    if "recontextualization" in k:
        return False
    return True


class CacheBase(ABC):
    def __init__(
        self,
        cache_root: str = "./cache",
        iteration_number=None,
        eval_prompt_modification_args=None,
        is_helpful_eval=False,
    ):
        self.cache_root = cache_root
        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)
        if iteration_number is None:
            raise ValueError(
                "Iteration number is None - this is required for cache key generation"
            )
        self.iteration_number = iteration_number
        self.eval_prompt_modification_args = eval_prompt_modification_args
        self.is_helpful_eval = is_helpful_eval

        self.ignore_keys = [
            "experiment_dir_name",
            "experiment_name",
            "iter_to_resume_from",
            "resume_from_checkpoint",
            "resume_from_experiment_dir",
            "eval_sft_and_base",
            "eval_at_initial",
            "eval_at_iterations",
            "eval_at_final",
            "num_expert_iteration_iters",
            "best_of_n_sft_config_path",
            "no_cache",
            "no_eval_cache",
            "sft_dataset_path",
        ]

    def get_hashable(self, config):
        if not is_dataclass(config) and not isinstance(
            config, (argparse.Namespace, dict)
        ):
            return config
        elif isinstance(config, argparse.Namespace):
            config_dict = vars(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            try:
                config_dict = asdict(config)
            except Exception as e:
                print("Could not convert your config to a dict using asdict: ", e)
                raise ValueError(f"Config type: {type(config)} is unsupported")

        return {k: self.get_hashable(v) for k, v in config_dict.items()}

    def make_cache_key(self, config, preprocess_config_fn):
        config = copy.deepcopy(config)

        config = self.get_hashable(config)
        if preprocess_config_fn:
            config = preprocess_config_fn(config)

        print(f"[CACHE KEY] CONFIG: {config}")

        from hashlib import sha256

        hasher = sha256()
        # Use json.dumps with sort_keys for deterministic serialization
        config_str = json.dumps(config, sort_keys=True, default=str)
        hasher.update(config_str.encode())
        return hasher.hexdigest()

    def get_cache_dir(self, key: str):
        return os.path.join(self.cache_root, key)

    def preprocess_config_fn(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError(
                f"preprocess_config_fn should receive a dict, got {type(config)}"
            )

        config["current_iteration_number"] = self.iteration_number
        config["eval_prompt_modification_args"] = self.get_hashable(
            self.eval_prompt_modification_args
        )
        # Always add is_helpful_eval to config (not just when True) so cache can differentiate
        for ignore_key in self.ignore_keys:
            config.pop(ignore_key, None)

        if not config.get("do_recontextualization"):
            print(
                "Normalizing all recontextualization args since we're not recontextualizing"
            )

            config = {k: v if not_a_recon_arg(k) else None for k, v in config.items()}
        if not config.get("do_prompt_modification_generation"):
            print(
                "Normalizing prompt modification args since we're not modifying prompts for generation"
            )
            config = {
                k: v if not_a_prompt_modification_generation_arg(k) else None
                for k, v in config.items()
            }
        if not config.get("regularization_data_type"):
            print(
                "Normalizing regularization args since we're not using regularization"
            )
            config = {
                k: v
                for k, v in config.items()
                if k not in ["regularization_alpha", "regularization_data_type"]
            }

        return config


class DirectoryCache(CacheBase):
    def try_retrieve_from_cache(self, config, output_dir) -> bool:
        """
        Try to retrieve cached results from cache directory.
        Returns True if cache hit and successfully copied, False otherwise.
        """

        if config.no_cache:
            print("NO CACHE SET; not retreiving from cache")
            return False
        try:
            key = self.make_cache_key(config, self.preprocess_config_fn)
            key_dir = self.get_cache_dir(key)

            print(f"[CACHE] Checking cache for key: {key[:16]}...")
            print(f"[CACHE] Cache directory: {key_dir}")

            if os.path.exists(key_dir):
                print("[CACHE] ✓ Cache HIT - found cached results")
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"[CACHE] Copying from cache: {key_dir} -> {output_dir}")
                    shutil.copytree(src=key_dir, dst=output_dir, dirs_exist_ok=True)
                    print("[CACHE] ✓ Successfully restored from cache")
                    return True
                except Exception as e:
                    print("[CACHE] ✗ ERROR copying from cache:")
                    print(f"[CACHE] Error type: {type(e).__name__}")
                    print(f"[CACHE] Error message: {e}")
                    print("[CACHE] Traceback:")
                    traceback.print_exc()
                    return False
            else:
                print("[CACHE] ✗ Cache MISS - will compute and cache results")
                return False
        except Exception as e:
            print("[CACHE] ✗ ERROR during cache check:")
            print(f"[CACHE] Error type: {type(e).__name__}")
            print(f"[CACHE] Error message: {e}")
            print("[CACHE] Traceback:")
            traceback.print_exc()
            return False

    def save_to_cache(self, config, output_dir) -> bool:
        """
        Save computed results to cache directory.
        Returns True if successfully saved, False otherwise.
        """
        if config.no_cache:
            print("NO CACHE SET: not saving to cache")
            return
        try:
            key = self.make_cache_key(config, self.preprocess_config_fn)
            key_dir = self.get_cache_dir(key)

            print("[CACHE] Saving results to cache")
            print(f"[CACHE] Cache key: {key[:16]}...")
            print(f"[CACHE] Cache directory: {key_dir}")
            print(f"[CACHE] Source directory: {output_dir}")

            os.makedirs(key_dir, exist_ok=True)
            try:
                shutil.copytree(src=output_dir, dst=key_dir, dirs_exist_ok=True)
                print("[CACHE] ✓ Successfully saved to cache")
                return True
            except Exception as e:
                print("[CACHE] ✗ WARNING: Failed to save to cache")
                print(f"[CACHE] Error type: {type(e).__name__}")
                print(f"[CACHE] Error message: {e}")
                print("[CACHE] Traceback:")
                traceback.print_exc()
                return False
        except Exception as e:
            print("[CACHE] ✗ ERROR during cache save:")
            print(f"[CACHE] Error type: {type(e).__name__}")
            print(f"[CACHE] Error message: {e}")
            print("[CACHE] Traceback:")
            traceback.print_exc()
            return False


class EvalCache(DirectoryCache):
    def __init__(
        self, iteration_number, prompt_modification_args, is_helpful_eval=False
    ):
        super().__init__(
            "./cache/eval",
            iteration_number=iteration_number,
            eval_prompt_modification_args=prompt_modification_args,
            is_helpful_eval=is_helpful_eval,
        )

    def try_retrieve_from_cache(self, config, output_dir) -> bool:
        if config.no_eval_cache:
            print("NO EVAL CACHE SET; not retreiving from cache")
            return False
        return super().try_retrieve_from_cache(config, output_dir)

    def save_to_cache(self, config, output_dir) -> bool:
        if config.no_eval_cache:
            print("NO EVAL CACHE SET; not saving to cache")
            return False
        return super().save_to_cache(config, output_dir)

    def preprocess_config_fn(self, config):
        config = super().preprocess_config_fn(config)

        if config.get("current_iteration_number", 0) < 0:
            print(
                "Since it's an initial evaluation, we're using a very limited number of arguments for the eval cache config"
            )
            config = {
                "generation_max_tokens": config.get("generation_max_tokens"),
                "eval_prompt_modification_args": config.get(
                    "eval_prompt_modification_args"
                ),
                "current_iteration_number": config.get("current_iteration_number"),
                "base_model": config.get("base_model"),
                "truncate_eval_dataset": config.get("truncate_eval_dataset"),
                "eval_N_completions": config.get("eval_N_completions"),
                "eval_generation_temperature": config.get(
                    "eval_generation_temperature"
                ),
                "eval_dataset_name": config.get("eval_dataset_name"),
                "eval_dataset_path": config.get("eval_dataset_path"),
                "debug": config.get("debug"),
            }
            print("INIT EVAL CONFIG: ", config)
        if self.is_helpful_eval:
            config["is_helpful_eval"] = True

        if config.get("eval_generation_temperature") == 1.0:
            config.pop("eval_generation_temperature")
        # Note: is_helpful_eval is kept in config to ensure different cache keys
        # for helpful vs non-helpful evaluations
        if config.get("data_generation_seed") == 42:
            config.pop("data_generation_seed")
        return config


class DataGenerationCache(DirectoryCache):
    def __init__(self, iteration_number):
        super().__init__("./cache/data_generation", iteration_number=iteration_number)

    def preprocess_config_fn(self, config):
        config = super().preprocess_config_fn(config)

        if config.get("current_iteration_number") == 0:
            print(
                "Since it's an initial generation, we're using a very limited number of arguments for the eval cache config"
            )
            config.pop("best_of_n_sft_config", None)
            config.pop("regularization_alpha", None)
            config.pop("regularization_data_type", None)
            config.pop("seed", None)

            config.pop("N_completions_to_keep", None)
            config = {k: v for k, v in config.items() if not_a_recon_arg(k)}
        if config.get("data_generation_seed") == 42:
            config.pop("data_generation_seed")
        config.pop("truncate_eval_dataset", None)
        config.pop("eval_N_completions", None)
        config.pop("eval_generation_temperature", None)
        config.pop("eval_dataset_name", None)
        config.pop("eval_dataset_path", None)
        config.pop("quality_alpha", None)
        config.pop("sycophancy_alpha", None)
        return config


class ModelCache(CacheBase):
    def __init__(self, iteration_number):
        super().__init__(cache_root="./cache/model", iteration_number=iteration_number)

    def preprocess_config_fn(self, config: dict):
        config = super().preprocess_config_fn(config)

        if config.get("data_generation_seed") == 42:
            config.pop("data_generation_seed")
        config.pop("truncate_eval_dataset", None)
        config.pop("eval_N_completions", None)
        config.pop("eval_dataset_name", None)
        config.pop("eval_dataset_path", None)
        config.pop("eval_generation_temperature")
        return config

    def try_retrieve_from_cache(self, config, base_model_obj=None, tokenizer_obj=None):
        """
        Try to retrieve cached model from cache directory.
        Returns (model, tokenizer) tuple if cache hit, None otherwise.
        """
        if hasattr(config, "no_cache") and config.no_cache:
            print("NO CACHE SET: not retrieving from cache")
            return None
        try:
            no_cache = config.get("no_cache", None)
            if no_cache:
                print("NO CACHE SET: not retrieving from cache")
                return None
        except Exception:
            pass
        try:
            key = self.make_cache_key(config, self.preprocess_config_fn)
            key_dir = self.get_cache_dir(key)

            print(f"[MODEL CACHE] Checking cache for key: {key[:16]}...")
            print(f"[MODEL CACHE] Cache directory: {key_dir}")

            if os.path.exists(key_dir):
                print("[MODEL CACHE] ✓ Cache HIT - found cached model")
                try:
                    from src.utils import load_peft_model_basic

                    if not base_model_obj or not tokenizer_obj:
                        print(
                            f"[MODEL CACHE] Loading model from scratch with adapter: {key_dir}"
                        )
                        model, tokenizer = load_peft_model_basic(
                            config.base_model, key_dir
                        )
                        print("[MODEL CACHE] ✓ Successfully loaded cached model")
                        model.train()  # Set model to training mode

                        # Explicitly enable gradients for adapter parameters
                        print("Setting adapters to be trainable")
                        for name, param in model.named_parameters():
                            if "lora" in name or "adapter" in name:
                                print("Setting an adapter to require gradient.")
                                param.requires_grad = True
                        return model, tokenizer
                    else:
                        from peft import PeftModel

                        print(
                            f"[MODEL CACHE] Loading adapter onto existing base model from: {key_dir}"
                        )
                        result = (
                            PeftModel.from_pretrained(base_model_obj, key_dir),
                            tokenizer_obj,
                        )
                        print("[MODEL CACHE] ✓ Successfully loaded cached adapter")
                        result.train()  # Set model to training mode

                        # Explicitly enable gradients for adapter parameters
                        print("Setting adapters to be trainable")
                        for name, param in result.named_parameters():
                            if "lora" in name or "adapter" in name:
                                param.requires_grad = True
                        return result
                except Exception as e:
                    print("[MODEL CACHE] ✗ ERROR loading from cache:")
                    print(f"[MODEL CACHE] Error type: {type(e).__name__}")
                    print(f"[MODEL CACHE] Error message: {e}")
                    print("[MODEL CACHE] Traceback:")
                    traceback.print_exc()
                    return None
            else:
                print("[MODEL CACHE] ✗ Cache MISS - will train and cache model")
                return None
        except Exception as e:
            print("[MODEL CACHE] ✗ ERROR during cache check:")
            print(f"[MODEL CACHE] Error type: {type(e).__name__}")
            print(f"[MODEL CACHE] Error message: {e}")
            print("[MODEL CACHE] Traceback:")
            traceback.print_exc()
            return None

    def save_to_cache(self, config, adapter_dir) -> bool:
        """
        Save trained model adapter to cache directory.
        Returns True if successfully saved, False otherwise.
        """
        if config.no_cache:
            print("NO CACHE SET: not saving to cache")
            return
        try:
            key = self.make_cache_key(config, self.preprocess_config_fn)
            key_dir = self.get_cache_dir(key)

            print("[MODEL CACHE] Saving model to cache")
            print(f"[MODEL CACHE] Cache key: {key[:16]}...")
            print(f"[MODEL CACHE] Cache directory: {key_dir}")
            print(f"[MODEL CACHE] Adapter directory: {adapter_dir}")

            os.makedirs(key_dir, exist_ok=True)
            try:
                shutil.copytree(src=adapter_dir, dst=key_dir, dirs_exist_ok=True)
                print("[MODEL CACHE] ✓ Successfully saved model to cache")
                return True
            except Exception as e:
                print("[MODEL CACHE] ✗ WARNING: Failed to save to cache")
                print(f"[MODEL CACHE] Error type: {type(e).__name__}")
                print(f"[MODEL CACHE] Error message: {e}")
                print("[MODEL CACHE] Traceback:")
                traceback.print_exc()
                return False
        except Exception as e:
            print("[MODEL CACHE] ✗ ERROR during cache save:")
            print(f"[MODEL CACHE] Error type: {type(e).__name__}")
            print(f"[MODEL CACHE] Error message: {e}")
            print("[MODEL CACHE] Traceback:")
            traceback.print_exc()
            return False
