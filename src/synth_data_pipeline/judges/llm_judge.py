import os
import json
import logging
from typing import Dict, Any, Optional
from .base import BaseJudge

logger = logging.getLogger(__name__)

class LLMJudge(BaseJudge):
    """Judge using any LLM."""
    
    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        config: Dict[str, Any] = None
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.provider = provider
        self.model = model
        self.temperature = temperature
        
        if provider == "openai":
            from openai import OpenAI
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            from anthropic import Anthropic
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def judge_single(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Judge a single example using an LLM."""
        # Extract text or relevant field from example
        if "text" in example:
            content = example["text"]
        else:
            content = json.dumps(example)
        
        user_prompt = f"Judge this example:\n{content}\n\nReturn JSON only."
        
        try:
            if self.provider == "openai":
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
                            "name": "JudgmentSchema",
                            "strict": True,
                            "schema": self.schema
                        }
                    }
                )
                result = json.loads(response.choices[0].message.content)
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=self.temperature,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": user_prompt + f"\n\nSchema:\n{json.dumps(self.schema, indent=2)}"}]
                )
                content = response.content[0].text
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                result = json.loads(content.strip())
            
            return result
            
        except Exception as e:
            logger.error(f"LLM judging failed: {e}")
            raise
