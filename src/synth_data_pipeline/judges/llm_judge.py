import json
import logging
import os
from statistics import fmean
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError, validate

from .base import BaseJudge

logger = logging.getLogger(__name__)


class LLMJudge(BaseJudge):
    """Judge using foundation models with structured JSON outputs."""

    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_template: Optional[str] = None,
        provider: str = "openai",
        model: str = "gpt-4.1",
        temperature: float = 0.3,
        num_samples: int = 2,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature

        cfg = config or {}
        self.num_samples = max(1, cfg.get("num_samples", num_samples))
        self.max_output_tokens = cfg.get("max_output_tokens", cfg.get("max_tokens", 1200))
        self.top_p = cfg.get("top_p")
        self.extra_params = {
            key: cfg[key]
            for key in [
                "seed",
                "frequency_penalty",
                "presence_penalty",
                "logprobs",
                "top_logprobs",
                "logit_bias",
            ]
            if key in cfg
        }
        self.response_format_name = cfg.get("response_format_name", "JudgmentSchema")
        self.tool_name = cfg.get("tool_name", "structured_output")
        self.safety_settings = cfg.get("safety_settings")
        self.max_retries = cfg.get("max_retries", 2)
        self.response_schema = cfg.get("response_schema", self.schema)

        self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]) -> None:
        if self.provider == "openai":
            from openai import OpenAI

            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API key not provided")
            self.client = OpenAI(api_key=key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic

            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Anthropic API key not provided")
            self.client = Anthropic(api_key=key)
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise ImportError(
                    "google-generativeai is required for Gemini judges. "
                    "Install it with `pip install google-generativeai`."
                ) from exc

            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Gemini API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY.")

            genai.configure(api_key=key)

            generation_config: Dict[str, Any] = {
                "temperature": self.temperature,
                "response_mime_type": "application/json",
                "response_schema": self.response_schema,
            }
            if self.top_p is not None:
                generation_config["top_p"] = self.top_p
            if self.max_output_tokens is not None:
                generation_config["max_output_tokens"] = self.max_output_tokens

            self.client = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=self.system_prompt,
                generation_config=generation_config,
            )
        else:  # pragma: no cover - configuration error
            raise ValueError(f"Unknown provider: {self.provider}")

    def judge_single(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Judge a single example using repeated sampling and aggregation."""
        user_prompt = self._build_user_prompt(example)
        samples: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = self.num_samples + self.max_retries

        while attempts < max_attempts and len(samples) < self.num_samples:
            try:
                if self.provider == "openai":
                    candidate = self._invoke_openai(user_prompt)
                elif self.provider == "anthropic":
                    candidate = self._invoke_anthropic(user_prompt)
                elif self.provider == "gemini":
                    candidate = self._invoke_gemini(user_prompt)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported provider: {self.provider}")

                validate(instance=candidate, schema=self.schema)
                samples.append(candidate)
            except ValidationError as exc:
                logger.warning("Judge sample failed schema validation: %s", exc)
            except Exception as exc:  # pragma: no cover - provider failure
                logger.warning("Judge invocation failed (attempt %s): %s", attempts + 1, exc)
            finally:
                attempts += 1

        if not samples:
            raise ValueError("All judge attempts failed to produce valid output.")

        aggregated = aggregate_schema_values(samples, self.schema)
        validate(instance=aggregated, schema=self.schema)
        return aggregated

    def _build_user_prompt(self, example: Dict[str, Any]) -> str:
        text_content = example.get("text")
        serialized = json.dumps(example, ensure_ascii=False)
        if self.user_template:
            context: Dict[str, Any] = {
                "text": text_content or "",
                "example": serialized,
                "json": serialized,
            }
            for key, value in example.items():
                if isinstance(value, (dict, list)):
                    context[key] = json.dumps(value, ensure_ascii=False)
                else:
                    context[key] = value
            return self.user_template.format(**context)
        if text_content:
            return (
                "Judge the following example and respond with JSON only that matches the schema:\n"
                f"{text_content}"
            )
        return (
            "Judge the following example and respond with JSON only that matches the schema:\n"
            f"{serialized}"
        )

    def _build_messages(self, user_prompt: str) -> List[Dict[str, Any]]:
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

    def _invoke_openai(self, user_prompt: str) -> Dict[str, Any]:
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
        request_payload.update(self.extra_params)

        response = self.client.responses.create(**request_payload)
        raw_output = getattr(response, "output_text", None)
        if not raw_output and getattr(response, "output", None):
            raw_output = "".join(
                part.text
                for item in response.output
                for part in getattr(item, "content", [])
                if getattr(part, "type", "") == "text"
            )
        if not raw_output:
            raise ValueError("OpenAI judge response was empty.")
        return json.loads(raw_output)

    def _invoke_anthropic(self, user_prompt: str) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {
                    "name": self.tool_name,
                    "description": "Return JSON adhering to the provided schema.",
                    "input_schema": self.schema,
                }
            ],
            tool_choice={"type": "tool", "name": self.tool_name},
        )

        for block in response.content:
            if block.type == "tool_use":
                return block.input
        for block in response.content:
            if block.type == "text" and block.text:
                return json.loads(block.text)
        raise ValueError("Anthropic judge response did not include structured output.")

    def _invoke_gemini(self, user_prompt: str) -> Dict[str, Any]:
        response = self.client.generate_content(
            user_prompt,
            safety_settings=self.safety_settings,
        )

        if hasattr(response, "text") and response.text:
            return json.loads(response.text)

        if getattr(response, "candidates", None):
            for candidate in response.candidates:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []):
                    text_part = getattr(part, "text", None)
                    if text_part:
                        return json.loads(text_part)

        raise ValueError("Gemini judge response did not contain JSON content.")

def aggregate_schema_values(values: List[Any], schema_fragment: Optional[Dict[str, Any]]) -> Any:
    if not values:
        return None

    schema_type = (schema_fragment or {}).get("type")

    if schema_type == "object" or (schema_type is None and isinstance(values[0], dict)):
        result: Dict[str, Any] = {}
        keys = set()
        for value in values:
            if isinstance(value, dict):
                keys.update(value.keys())
        for key in keys:
            child_schema = (schema_fragment or {}).get("properties", {}).get(key) if schema_fragment else None
            child_values = [value[key] for value in values if isinstance(value, dict) and key in value]
            if not child_values:
                continue
            result[key] = aggregate_schema_values(child_values, child_schema)
        return result

    if schema_type == "array" or (schema_type is None and isinstance(values[0], list)):
        combined: List[Any] = []
        for arr in values:
            if not isinstance(arr, list):
                continue
            for item in arr:
                if item not in combined:
                    combined.append(item)
        max_items = (schema_fragment or {}).get("maxItems")
        if isinstance(max_items, int):
            combined = combined[:max_items]
        return combined

    if schema_type in {"number", "integer"} or all(isinstance(v, (int, float)) for v in values):
        numeric = [float(v) for v in values if isinstance(v, (int, float))]
        if not numeric:
            return values[-1]
        value: float = fmean(numeric)
        minimum = (schema_fragment or {}).get("minimum")
        maximum = (schema_fragment or {}).get("maximum")
        if minimum is not None:
            value = max(value, float(minimum))
        if maximum is not None:
            value = min(value, float(maximum))
        if schema_type == "integer":
            return int(round(value))
        return float(value)

    if schema_type == "boolean" or all(isinstance(v, bool) for v in values):
        true_count = sum(bool(v) for v in values)
        return true_count >= len(values) / 2

    for value in reversed(values):
        if value not in ("", None):
            return value
    return values[-1]
