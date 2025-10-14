"""
Rollout Storage - Simplified
============================

Lightweight storage for rollouts in JSONL format.
Always expects grouped rollouts for efficiency and clarity.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class RolloutStorage:
    """Simple rollout storage - always expects grouped rollouts"""

    def __init__(self):
        """Initialize storage"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def save_rollouts(
        self,
        grouped_rollouts: List[List[Any]],
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        include_messages: bool = False,
    ) -> None:
        """
        Save grouped rollouts to JSONL file.
        
        Args:
            grouped_rollouts: List of rollout groups, one group per sample
            output_path: Path to save to
            metadata: Optional metadata
            include_messages: Whether to include conversation messages
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate totals
        total_rollouts = sum(len(group) for group in grouped_rollouts)
        
        # Build metadata
        save_metadata = {
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(grouped_rollouts),
            "total_rollouts": total_rollouts,
            "include_messages": include_messages,
            **(metadata or {})
        }
        
        # Save as JSONL
        with open(output_path, "w") as f:
            # Metadata line
            f.write(json.dumps({"_metadata": save_metadata}) + "\n")
            
            # One sample per line
            for sample_rollouts in grouped_rollouts:
                if not sample_rollouts:  # Skip empty groups
                    continue
                    
                sample_data = {
                    "sample": sample_rollouts[0].sample.sample.to_summary(),
                    "rollouts": [
                        {
                            "rollout": {
                                "context": rollout.sample.context,
                                "model_response": rollout.final_response,
                                **({"messages": rollout.messages} if include_messages else {})
                            },
                            "evaluation": rollout.evaluation_result.to_summary() if rollout.evaluation_result else None
                        }
                        for rollout in sample_rollouts
                    ]
                }
                f.write(json.dumps(sample_data) + "\n")
        
        self.logger.info(f"Saved {len(grouped_rollouts)} samples with {total_rollouts} rollouts to {output_path}")

