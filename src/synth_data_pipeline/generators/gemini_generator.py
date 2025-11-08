import json
import logging
import os
from typing import Any, Dict, Optional

from .base import BaseGenerator

logger = logging.getLogger(__name__)


class GeminiGenerator(BaseGenerator):
    """Generator that uses Google's Gemini models with JSON schema guarantees."""

    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_template: str,
        model: str = "gemini-2.5-pro-latest",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.model_name = model
        self.temperature = temperature
        self._raw_schema = config.get("response_schema") if config else None
        self._safety_settings = (config or {}).get("safety_settings")
        self._top_p = (config or {}).get("top_p", 0.95)
        self._max_output_tokens = (config or {}).get("max_output_tokens", 2048)

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for GeminiGenerator. "
                "Install it with `pip install google-generativeai`."
            ) from exc

        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY.")

        genai.configure(api_key=api_key)

        response_schema = self._raw_schema or self.schema
        generation_config = {
            "temperature": self.temperature,
            "top_p": self._top_p,
            "response_mime_type": "application/json",
            "max_output_tokens": self._max_output_tokens,
            "response_schema": response_schema,
        }

        self._client = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            generation_config=generation_config,
        )

    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single example using Gemini."""
        user_prompt = self.user_template.format(**kwargs)
        try:
            response = self._client.generate_content(
                user_prompt,
                safety_settings=self._safety_settings,
            )
        except Exception as exc:  # pragma: no cover - network call
            logger.error("Gemini generation failed: %s", exc)
            raise

        raw_output = None

        if hasattr(response, "text") and response.text:
            raw_output = response.text
        elif getattr(response, "candidates", None):
            for candidate in response.candidates:
                for part in getattr(candidate, "content", {}).parts:
                    if getattr(part, "text", None):
                        raw_output = part.text
                        break
                if raw_output:
                    break

        if not raw_output:
            raise ValueError("Gemini response did not contain text content.")

        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode Gemini JSON output: %s", exc)
            raise

        if self.validate_output(result):
            return result

        raise ValueError("Generated output failed validation against schema.")

