import os
import logging
from typing import Dict, Any, Optional
from anthropic import Anthropic
from .base import BaseGenerator
from json import loads

logger = logging.getLogger(__name__)

class AnthropicGenerator(BaseGenerator):
    """Generator using Anthropic's Claude."""
    
    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_template: str,
        model: str = "claude-4.5-sonnet",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        config: Dict[str, Any] = None
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = (config or {}).get("max_output_tokens", 2048)
        self._tool_name = (config or {}).get("tool_name", "structured_output")
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=api_key)

    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single example using Claude."""
        user_prompt = self.user_template.format(**kwargs)

        try:
            response = self.client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "name": self._tool_name,
                        "description": "Return JSON that satisfies the provided schema.",
                        "input_schema": self.schema,
                    }
                ],
                tool_choice={"type": "tool", "name": self._tool_name},
            )
        except Exception as exc:  # pragma: no cover - network calls
            logger.error("Anthropic generation failed: %s", exc)
            raise

        tool_payload: Optional[Dict[str, Any]] = None

        for block in response.content:
            if block.type == "tool_use":
                tool_payload = block.input
                break

        if tool_payload is None:
            # Fallback: try to parse any text block
            for block in response.content:
                if block.type == "text" and block.text:
                    try:
                        tool_payload = loads(block.text)
                        break
                    except Exception:
                        continue

        if tool_payload is None:
            raise ValueError("Claude response did not include structured tool output.")

        if self.validate_output(tool_payload):
            return tool_payload

        raise ValueError("Generated output failed validation")
