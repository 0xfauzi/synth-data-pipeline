import os
import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from .base import BaseGenerator

logger = logging.getLogger(__name__)

class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI's structured outputs."""
    
    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_template: str,
        model: str = "gpt-4",
        temperature: float = 0.9,
        api_key: Optional[str] = None,
        config: Dict[str, Any] = None
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.model = model
        self.temperature = temperature
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single example using OpenAI."""
        user_prompt = self.user_template.format(**kwargs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "GeneratedSchema",
                        "strict": True,
                        "schema": self.schema
                    }
                }
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if self.validate_output(result):
                return result
            else:
                raise ValueError("Generated output failed validation")
                
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
