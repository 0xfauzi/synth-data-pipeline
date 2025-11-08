from typing import Any, Dict, Optional

from jsonschema import validate

from .llm_judge import LLMJudge

DEFAULT_PAIRWISE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["preferred", "confidence", "reason"],
    "additionalProperties": False,
    "properties": {
        "label": {
            "type": "string",
            "default": "overall",
        },
        "preferred": {"type": "string", "enum": ["A", "B", "tie"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
}

DEFAULT_PAIRWISE_TEMPLATE = """Compare two candidates and select the one that better satisfies the evaluation rubric (i.e., closer to the positive class or higher quality score).

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Respond with JSON only, containing:
- preferred: "A", "B", or "tie"
- confidence: probability in [0,1]
- reason: short string explaining the decision
"""


class PairwiseJudge:
    """Lightweight wrapper around LLMJudge for pairwise comparisons."""

    def __init__(
        self,
        system_prompt: str,
        provider: str,
        model: str,
        temperature: float = 0.3,
        num_samples: int = 1,
        api_key: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        user_template: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.schema = schema or DEFAULT_PAIRWISE_SCHEMA
        self.judge = LLMJudge(
            schema=self.schema,
            system_prompt=system_prompt,
            user_template=user_template or DEFAULT_PAIRWISE_TEMPLATE,
            provider=provider,
            model=model,
            temperature=temperature,
            num_samples=num_samples,
            api_key=api_key,
            config=config,
        )

    def compare(
        self,
        candidate_a: Dict[str, Any],
        candidate_b: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
        }
        if label:
            payload["label"] = label
        if context:
            payload.update({f"context_{k}": v for k, v in context.items()})
        result = self.judge.judge_single(payload)
        validate(instance=result, schema=self.schema)
        return result

