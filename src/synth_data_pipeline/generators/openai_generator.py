import json
import logging
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from .base import BaseGenerator

logger = logging.getLogger(__name__)


class OpenAIGenerator(BaseGenerator):
    """Generator using the official OpenAI Responses API with JSON schema enforcement."""

    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_template: str,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.model = model
        self.temperature = temperature

        cfg = config or {}
        self.max_output_tokens = cfg.get("max_output_tokens")
        self.top_p = cfg.get("top_p")
        self.response_format_name = cfg.get("response_format_name", "GeneratedSchema")
        self.reasoning = cfg.get("reasoning")
        self.extra_params = {
            key: cfg[key]
            for key in [
                "seed",
                "frequency_penalty",
                "presence_penalty",
                "warmup_ratio",
                "logit_bias",
            ]
            if key in cfg
        }

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=api_key)

    def _build_messages(self, user_prompt: str) -> Any:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single example using the OpenAI Responses API."""
        user_prompt = self.user_template.format(**kwargs)

        request_payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "input": self._build_messages(user_prompt),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_format_name,
                    "strict": True,
                    "schema": self.schema,
                },
            },
        }

        if self.max_output_tokens is not None:
            request_payload["max_output_tokens"] = self.max_output_tokens
        if self.top_p is not None:
            request_payload["top_p"] = self.top_p
        if self.reasoning is not None:
            request_payload["reasoning"] = self.reasoning
        request_payload.update(self.extra_params)

        try:
            response = self.client.responses.create(**request_payload)
        except Exception as exc:  # pragma: no cover - network failure
            logger.error("OpenAI generation failed: %s", exc)
            raise

        raw_output = getattr(response, "output_text", None)
        if not raw_output and getattr(response, "output", None):
            try:
                raw_output = "".join(
                    part.text
                    for item in response.output
                    for part in getattr(item, "content", [])
                    if getattr(part, "type", "") == "text"
                )
            except Exception:  # pragma: no cover - defensive
                raw_output = None

        if not raw_output:
            raise ValueError("OpenAI response did not include textual output.")

        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode OpenAI JSON output: %s", exc)
            raise

        if self.validate_output(result):
            return result

        raise ValueError("Generated output failed validation against schema.")
