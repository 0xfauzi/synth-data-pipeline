import json
import logging
from typing import Dict, Any, List
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class Validator:
    """Validate data against schemas."""
    
    def __init__(self, generation_schema: Dict[str, Any], judgment_schema: Dict[str, Any]):
        self.generation_schema = generation_schema
        self.judgment_schema = judgment_schema
        self.stats = {"total": 0, "valid": 0, "invalid": 0}
    
    def validate_row(self, row: Dict[str, Any]) -> bool:
        """Validate a single row."""
        self.stats["total"] += 1
        
        try:
            # Validate generated content
            if "generated" in row:
                validate(instance=row["generated"], schema=self.generation_schema)
            
            # Validate judgment
            if "judgment" in row:
                validate(instance=row["judgment"], schema=self.judgment_schema)
            
            self.stats["valid"] += 1
            return True
            
        except ValidationError as e:
            logger.debug(f"Validation error: {e}")
            self.stats["invalid"] += 1
            return False
    
    def validate_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate batch and return only valid rows."""
        valid_rows = []
        for row in rows:
            if self.validate_row(row):
                valid_rows.append(row)
        
        logger.info(f"Validation stats: {self.stats}")
        return valid_rows
