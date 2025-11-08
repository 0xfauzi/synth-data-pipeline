import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "synth_data_pipeline" / "utils" / "generation_sampler.py"
spec = importlib.util.spec_from_file_location("generation_sampler", MODULE_PATH)
generation_sampler = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["generation_sampler"] = generation_sampler
spec.loader.exec_module(generation_sampler)  # type: ignore[attr-defined]
GenerationSampler = generation_sampler.GenerationSampler


def build_sampler(seed: int = 123) -> GenerationSampler:
    config = {
        "response_schema_version": "test-version",
        "industries": {"values": ["FinTech", "HealthTech"]},
        "roles": {"values": ["PM", "Engineer"]},
        "length_buckets": {
            "short": {"min_chars": 100, "max_chars": 200, "weight": 1.0},
            "medium": {"min_chars": 300, "max_chars": 600, "weight": 1.0},
        },
        "styles": {
            "test_style": {
                "style_id": "test_style",
                "cta_present": True,
                "emoji_count": 2,
                "cliche_count": 1,
                "buzzword_count": 3,
                "has_explicit_humility_phrase": False,
                "story_claims_verifiable": True,
                "length_bucket": "short",
            }
        },
        "hashtags": {"default": {"values": [["testtag"]]}}
    }
    sampler = GenerationSampler(config=config)
    sampler.rng.seed(seed)
    return sampler


def test_sampler_generates_required_fields():
    sampler = build_sampler()
    params = sampler.sample()

    required_fields = {
        "prompt_id",
        "style_id",
        "industry",
        "role",
        "cta_present",
        "emoji_count",
        "cliche_count",
        "buzzword_count",
        "has_explicit_humility_phrase",
        "story_claims_verifiable",
        "length_bucket",
        "hashtags",
        "seed",
        "response_schema_version",
        "language",
    }

    for field in required_fields:
        assert field in params, f"Missing field {field}"

    assert params["length_bucket"] == "short"
    assert params["style_id"] == "test_style"


def test_format_for_prompt_has_json_compatible_values():
    sampler = build_sampler()
    params = sampler.sample()
    formatted = GenerationSampler.format_for_prompt(params)

    assert formatted["cta_present"] in {"true", "false"}
    assert isinstance(formatted["emoji_count"], int)
    json_hashtags = formatted["hashtags"]
    parsed_hashtags = json.loads(json_hashtags)
    assert isinstance(parsed_hashtags, list)
    assert parsed_hashtags == ["testtag"]

