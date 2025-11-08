import logging
from typing import Dict, Any, List
import numpy as np
from .base import BaseJudge

logger = logging.getLogger(__name__)

class EnsembleJudge(BaseJudge):
    """Combine multiple judges for more robust labeling."""
    
    def __init__(
        self,
        judges: List[BaseJudge],
        aggregation: str = "mean",
        config: Dict[str, Any] = None
    ):
        # Use first judge's schema
        super().__init__(judges[0].schema if judges else {}, config)
        self.judges = judges
        self.aggregation = aggregation
    
    def judge_single(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Judge using ensemble of judges."""
        judgments = []
        
        for judge in self.judges:
            try:
                judgment = judge.judge_single(example)
                judgments.append(judgment)
            except Exception as e:
                logger.warning(f"Judge failed: {e}")
                continue
        
        if not judgments:
            raise ValueError("All judges failed")
        
        # Aggregate judgments
        if self.aggregation == "mean":
            return self._aggregate_mean(judgments)
        elif self.aggregation == "median":
            return self._aggregate_median(judgments)
        elif self.aggregation == "vote":
            return self._aggregate_vote(judgments)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def _aggregate_mean(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average numeric values across judgments."""
        result = {}
        
        # Get all keys
        all_keys = set()
        for j in judgments:
            all_keys.update(j.keys())
        
        for key in all_keys:
            values = [j.get(key) for j in judgments if key in j]
            
            # Handle different types
            if all(isinstance(v, (int, float)) for v in values):
                result[key] = np.mean(values).item()
            elif all(isinstance(v, dict) for v in values):
                # Recursively aggregate nested dicts
                result[key] = self._aggregate_mean(values)
            else:
                # For non-numeric, take first or most common
                result[key] = values[0]
        
        return result
    
    def _aggregate_median(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Median for numeric values."""
        result = {}
        for key in judgments[0].keys():
            values = [j.get(key) for j in judgments if key in j]
            if all(isinstance(v, (int, float)) for v in values):
                result[key] = np.median(values).item()
            else:
                result[key] = values[len(values)//2]
        return result
    
    def _aggregate_vote(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Majority vote for binary decisions."""
        result = {}
        for key in judgments[0].keys():
            values = [j.get(key) for j in judgments if key in j]
            if all(isinstance(v, (int, float)) for v in values):
                # For probabilities, use threshold of 0.5
                votes = [1 if v >= 0.5 else 0 for v in values]
                result[key] = 1.0 if sum(votes) > len(votes)/2 else 0.0
            else:
                # Most common value
                from collections import Counter
                result[key] = Counter(values).most_common(1)[0][0]
        return result
