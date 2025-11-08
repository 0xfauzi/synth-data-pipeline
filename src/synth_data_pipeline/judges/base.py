from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class BaseJudge(ABC):
    """Base class for all judges."""
    
    def __init__(self, schema: Dict[str, Any], config: Dict[str, Any] = None):
        self.schema = schema
        self.config = config or {}
    
    @abstractmethod
    def judge_single(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Judge a single example."""
        pass
    
    def judge_batch(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Judge multiple examples."""
        results = []
        for example in examples:
            try:
                judgment = self.judge_single(example)
                results.append(judgment)
            except Exception as e:
                logger.error(f"Failed to judge example: {e}")
                results.append(None)
        return results
