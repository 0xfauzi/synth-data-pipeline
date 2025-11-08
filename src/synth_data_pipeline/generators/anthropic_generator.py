import os
import json
import logging
from typing import Dict, Any, Optional
from anthropic import Anthropic
from .base import BaseGenerator

logger = logging.getLogger(__name__)

class AnthropicGenerator(BaseGenerator):
    """Generator using Anthropic's Claude."""
    
    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_template: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.9,
        api_key: Optional[str] = None,
        config: Dict[str, Any] = None
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.model = model
        self.temperature = temperature
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=api_key)
    
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single example using Claude."""
        user_prompt = self.user_template.format(**kwargs)
        
        # Add JSON instruction to prompt
        full_prompt = f"{user_prompt}\n\nReturn only valid JSON matching this schema:\n{json.dumps(self.schema, indent=2)}"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            
            # Try to parse JSON (may need to extract from markdown)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            if self.validate_output(result):
                return result
            else:
                raise ValueError("Generated output failed validation")
                
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
