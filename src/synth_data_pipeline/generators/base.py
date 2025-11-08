from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """Base class for all data generators."""
    
    def __init__(self, schema: Dict[str, Any], config: Dict[str, Any] = None):
        self.schema = schema
        self.config = config or {}
        self._validate_schema()
    
    def _validate_schema(self):
        """Validate that the schema is properly formatted."""
        required_fields = ["type", "properties"]
        for field in required_fields:
            if field not in self.schema:
                raise ValueError(f"Schema missing required field: {field}")
    
    @abstractmethod
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single example."""
        pass
    
    def generate_batch(self, n: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate multiple examples."""
        results = []
        for i in range(n):
            try:
                example = self.generate_single(**kwargs)
                results.append(example)
            except Exception as e:
                logger.error(f"Failed to generate example {i}: {e}")
                continue
        return results
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output against schema."""
        try:
            from jsonschema import validate
            validate(instance=output, schema=self.schema)
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
