#!/usr/bin/env python3
"""Pregenerate cache for fast experimentation"""

import hydra
from omegaconf import DictConfig

from src.dataset_modules.factory import create_dataset
from src.generation import RolloutGenerator
from src.storage import use_cache, save_rollouts_to_cache
from src.experiment_utils import load_model_from_experiment_or_config


@hydra.main(version_base=None, config_path="../configs", config_name="pregenerate/code_generation")
def main(cfg: DictConfig) -> None:
    """Pregenerate rollouts and cache them for fast experimentation"""
    
    print("🚀 Starting cache pregeneration...")
    
    # Load model and dataset from config
    model = load_model_from_experiment_or_config(experiment_dir=None, cfg=cfg)
    dataset = create_dataset(hydra.utils.instantiate(cfg.dataset))
    generator = RolloutGenerator(model, dataset)
    
    # Get samples to pregenerate
    all_samples = dataset.samples
    print(f"📊 Using all samples: {len(all_samples)}")
    
    # Pregenerate with cache enabled
    with use_cache():
        print("💾 Cache enabled - starting pregeneration...")
        
        total_generated = 0
        contexts = cfg.pregenerate.contexts
        
        for context in contexts:
            print(f"\n=== CONTEXT: {context} ===")
            
            # Generate for each generation config
            for config_name, gen_config in cfg.pregenerate.generation_configs.items():
                generation_config = hydra.utils.instantiate(gen_config)
                
                print(f"Generating {config_name} rollouts ({generation_config.n_rollouts} per sample)...")
                rollouts = generator.generate_rollouts(
                    samples=all_samples,
                    context=context,
                    generation_config=generation_config,
                    evaluate=cfg.pregenerate.evaluate_rollouts
                )
                
                count = sum(len(group) for group in rollouts)
                total_generated += count
                print(f"✅ Generated {count} {config_name} rollouts for {context}")

                # Save to cache
                sample_ids = [sample.id for sample in all_samples]
                model_name = getattr(generator.model, 'name', str(type(generator.model).__name__))
                dataset_name = getattr(generator.dataset, 'name', str(type(generator.dataset).__name__))
                
                save_rollouts_to_cache(
                    rollouts_grouped=rollouts,
                    sample_ids=sample_ids,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    context=context,
                    generation_config_dict=generation_config.to_dict()
                )

        print(f"\n🎉 PREGENERATION COMPLETE!")
        print(f"📈 Total rollouts generated: {total_generated}")
        print(f"💾 Cache files saved to: cache/")
        
        # Calculate expected total
        expected_per_config = len(all_samples) * len(contexts)
        total_configs = len(cfg.pregenerate.generation_configs)
        expected_rollouts = sum(
            expected_per_config * hydra.utils.instantiate(gen_config).n_rollouts 
            for gen_config in cfg.pregenerate.generation_configs.values()
        )
        
        print(f"\n📊 Summary:")
        print(f"  Expected rollouts: {expected_rollouts}")
        print(f"  Actual generated: {total_generated}")
        
        if total_generated == expected_rollouts:
            print("✅ All rollouts generated successfully!")
        else:
            print(f"⚠️  Mismatch in rollout count")


if __name__ == "__main__":
    main()
