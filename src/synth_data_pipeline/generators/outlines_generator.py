import logging
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, create_model
from .base import BaseGenerator

logger = logging.getLogger(__name__)

class OutlinesGenerator(BaseGenerator):
    """Generator using Outlines for local models with structured outputs."""
    
    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_template: str,
        model_name: str = "gpt2",
        backend: str = "transformers",
        config: Dict[str, Any] = None
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.user_template = user_template
        
        try:
            import outlines
        except ImportError:
            raise ImportError("Outlines not installed. Install with: pip install outlines")
        
        # Create Pydantic model from schema
        self.pydantic_model = self._schema_to_pydantic(schema)
        
        # Initialize outlines model
        if backend == "openai":
            from openai import OpenAI
            client = OpenAI()
            self.model = outlines.from_openai(client, model_name)
        elif backend == "transformers":
            self.model = outlines.models.transformers(model_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _schema_to_pydantic(self, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Convert JSON schema to Pydantic model."""
        fields = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        for field_name, field_schema in properties.items():
            field_type = Any  # Default type
            
            # Map JSON schema types to Python types
            if field_schema.get("type") == "string":
                field_type = str
            elif field_schema.get("type") == "integer":
                field_type = int
            elif field_schema.get("type") == "number":
                field_type = float
            elif field_schema.get("type") == "boolean":
                field_type = bool
            elif field_schema.get("type") == "array":
                field_type = list
            elif field_schema.get("type") == "object":
                field_type = dict
            
            # Make optional if not required
            if field_name not in required:
                from typing import Optional
                field_type = Optional[field_type]
            
            fields[field_name] = (field_type, ...)
        
        return create_model("GeneratedModel", **fields)
    
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single example using Outlines."""
        user_prompt = self.user_template.format(**kwargs)
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        
        try:
            # Generate structured output
            result = self.model(full_prompt, self.pydantic_model)
            result_dict = result.model_dump()
            
            if self.validate_output(result_dict):
                return result_dict
            else:
                raise ValueError("Generated output failed validation")
                
        except Exception as e:
            logger.error(f"Outlines generation failed: {e}")
            raise
