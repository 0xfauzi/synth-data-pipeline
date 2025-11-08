import json
import logging
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError, validate

from .base import BaseJudge
from .llm_judge import aggregate_schema_values

logger = logging.getLogger(__name__)


class PrometheusJudge(BaseJudge):
    """Judge that runs Prometheus 2 (or other HF evaluators) locally."""

    def __init__(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_template: Optional[str] = None,
        model_name: str = "prometheus-eval/prometheus-2-7b",
        temperature: float = 0.3,
        num_samples: int = 2,
        max_new_tokens: int = 1024,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(schema, config)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.model_name = model_name
        self.temperature = temperature
        cfg = config or {}
        self.num_samples = max(1, cfg.get("num_samples", num_samples))
        self.max_new_tokens = cfg.get("max_new_tokens", max_new_tokens)
        self.top_p = cfg.get("top_p", 0.9)
        self.do_sample = cfg.get("do_sample", True)
        self.device_map = cfg.get("device_map")
        self.dtype = cfg.get("dtype")
        self.load_in_4bit = cfg.get("load_in_4bit", False)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for PrometheusJudge. "
                "Install extras with `pip install .[local]`."
            ) from exc

        import torch

        tokenizer_args: Dict[str, Any] = cfg.get("tokenizer_kwargs", {})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **tokenizer_args)

        model_args: Dict[str, Any] = cfg.get("model_kwargs", {})
        if self.dtype:
            model_args.setdefault("torch_dtype", getattr(torch, self.dtype))
        if self.load_in_4bit:
            model_args["load_in_4bit"] = True
            model_args.setdefault("device_map", self.device_map or "auto")
        elif self.device_map:
            model_args["device_map"] = self.device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)

        # Determine device for inference
        if hasattr(self.model, "device"):
            self.device = self.model.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def judge_single(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(example)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        samples: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = self.num_samples * 2

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        while attempts < max_attempts and len(samples) < self.num_samples:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample or self.temperature > 0,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)

            try:
                candidate = self._extract_json(text)
                validate(instance=candidate, schema=self.schema)
                samples.append(candidate)
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning("Prometheus sample invalid: %s", exc)
            finally:
                attempts += 1

        if not samples:
            raise ValueError("Prometheus judge failed to produce any valid outputs.")

        aggregated = aggregate_schema_values(samples, self.schema)
        validate(instance=aggregated, schema=self.schema)
        return aggregated

    def _build_prompt(self, example: Dict[str, Any]) -> str:
        serialized = json.dumps(example, ensure_ascii=False)
        text_content = example.get("text", "")
        if self.user_template:
            context: Dict[str, Any] = {
                "text": text_content,
                "example": serialized,
                "json": serialized,
            }
            for key, value in example.items():
                if isinstance(value, (dict, list)):
                    context[key] = json.dumps(value, ensure_ascii=False)
                else:
                    context[key] = value
            return self.system_prompt + "\n\n" + self.user_template.format(**context)
        return (
            f"{self.system_prompt}\n\n"
            f"Example:\n{serialized}\n\n"
            "Respond with JSON only that conforms to the specified schema."
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = text[start : end + 1]
                return json.loads(snippet)
            raise

