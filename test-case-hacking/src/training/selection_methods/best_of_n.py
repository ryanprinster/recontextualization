"""
Best-of-N Selection Method
=========================

Selects the best rollout from N generated rollouts based on evaluation scores.
"""

from typing import List, Dict, Any, Tuple

from .base import BaseRolloutSelector
from ..data_structures import SampleRollouts


class BestOfNRolloutSelector(BaseRolloutSelector):
    """BestOfN selector"""
    
    def select_rollouts(
        self,
        sample_rollouts_list: List[SampleRollouts]
    ) -> Tuple[List[SampleRollouts], Dict[str, Any]]:
        """Select best of N rollouts (overrides base class default behavior)"""
        
        # Use base class helper to ensure rollouts
        total_generated = self._ensure_rollouts(sample_rollouts_list)
        
        # Select best rollout for each sample - simplified logic
        selected_scores = []
        all_scores = []
        
        for sample_rollouts in sample_rollouts_list:
            # Find best rollout directly - much simpler
            # Will naturally raise ValueError if rollouts is empty
            best_rollout = max(sample_rollouts.rollouts, key=lambda r: r.evaluation_result.score)
            sample_rollouts.selected_rollout = best_rollout
            
            # Collect scores for metrics
            scores = [r.evaluation_result.score for r in sample_rollouts.rollouts]
            selected_scores.append(best_rollout.evaluation_result.score)
            all_scores.extend(scores)
        
        metrics = {
            "method": "best_of_n_rollout_selection", 
            "samples_processed": len(sample_rollouts_list),
            "rollouts_generated": total_generated,
            "selected_mean_score": sum(selected_scores) / len(selected_scores) if selected_scores else 0,
            "all_mean_score": sum(all_scores) / len(all_scores) if all_scores else 0
        }
        
        return sample_rollouts_list, metrics
