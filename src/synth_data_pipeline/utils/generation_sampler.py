from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _coerce_style_id(name: str, data: Dict[str, Any]) -> str:
    candidate = data.get("style_id")
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    return name


@dataclass
class LengthBucket:
    name: str
    min_chars: int
    max_chars: int
    weight: float = 1.0


@dataclass
class StylePreset:
    name: str
    params: Dict[str, Any]
    weight: float = 1.0

    def resolve(self) -> Dict[str, Any]:
        resolved = dict(self.params)
        resolved.setdefault("style_id", _coerce_style_id(self.name, resolved))
        return resolved


def _normalize_weights(items: Iterable[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    items_list = list(items)
    total = sum(weight for _, weight in items_list) or 1.0
    return [(item, weight / total) for item, weight in items_list]


class GenerationSampler:
    """Sample structured generation knobs from configuration."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config or {}
        self.rng = rng or random.Random()

        self.response_schema_version: str = self.config.get("response_schema_version", "v1")

        industries = (self.config.get("industries") or {}).get("values")
        self.industries: List[str] = list(industries or [])

        roles = (self.config.get("roles") or {}).get("values")
        self.roles: List[str] = list(roles or [])

        self.length_buckets: List[LengthBucket] = self._parse_length_buckets()
        self.styles: List[StylePreset] = self._parse_styles()
        self.hashtag_sets: List[List[str]] = self._parse_hashtags()

        sampling_cfg = self.config.get("sampling", {})
        self.shuffle_styles: bool = bool(sampling_cfg.get("shuffle_styles", False))
        self.ensure_label_coverage: bool = bool(sampling_cfg.get("ensure_label_coverage", False))
        self.real_data_mix_ratio: float = float(sampling_cfg.get("real_data_mix_ratio", 0.0))

        self._style_cycle: List[StylePreset] = []
        self._style_index: int = 0
        self._counter: int = 0
        self._last_shuffle_ts: float = 0.0

        if self.shuffle_styles and self.styles:
            self._reshuffle_styles()

    def _parse_length_buckets(self) -> List[LengthBucket]:
        buckets_config = self.config.get("length_buckets") or {}
        buckets: List[Tuple[LengthBucket, float]] = []
        for name, params in buckets_config.items():
            if not isinstance(params, dict):
                continue
            min_chars = int(params.get("min_chars", 0))
            max_chars = int(params.get("max_chars", max(0, min_chars)))
            weight = float(params.get("weight", 1.0))
            buckets.append((LengthBucket(name=name, min_chars=min_chars, max_chars=max_chars, weight=weight), weight))
        if not buckets:
            buckets.append((LengthBucket(name="medium", min_chars=300, max_chars=900, weight=1.0), 1.0))
        return [bucket for bucket, _ in _normalize_weights(buckets)]

    def _parse_styles(self) -> List[StylePreset]:
        styles_config = self.config.get("styles") or {}
        presets: List[Tuple[StylePreset, float]] = []
        for name, params in styles_config.items():
            if not isinstance(params, dict):
                continue
            weight = float(params.get("weight", 1.0))
            presets.append((StylePreset(name=name, params=params, weight=weight), weight))
        if not presets:
            presets.append(
                (
                    StylePreset(
                        name="default",
                        params={
                            "cta_present": False,
                            "emoji_count": 0,
                            "cliche_count": 0,
                            "buzzword_count": 0,
                            "has_explicit_humility_phrase": False,
                            "story_claims_verifiable": True,
                            "length_bucket": "medium",
                        },
                    ),
                    1.0,
                )
            )
        return [preset for preset, _ in _normalize_weights(presets)]

    def _parse_hashtags(self) -> List[List[str]]:
        hashtag_cfg = (self.config.get("hashtags") or {}).get("default", {})
        values = hashtag_cfg.get("values") or []
        valid_sets: List[List[str]] = []
        for item in values:
            if isinstance(item, list):
                valid_sets.append([str(tag).lstrip("#").lower() for tag in item if str(tag).strip()])
            elif isinstance(item, str):
                valid_sets.append([item.lstrip("#").lower()])
        if not valid_sets:
            valid_sets.append([])
        return valid_sets

    def _reshuffle_styles(self) -> None:
        self._style_cycle = list(self.styles)
        self.rng.shuffle(self._style_cycle)
        self._style_index = 0
        self._last_shuffle_ts = time.time()

    def _choose_style(self) -> StylePreset:
        if not self.styles:
            return StylePreset(name="default", params={})
        if self.ensure_label_coverage:
            if not self._style_cycle or self._style_index >= len(self._style_cycle):
                self._reshuffle_styles()
            preset = self._style_cycle[self._style_index]
            self._style_index += 1
            return preset
        if self.shuffle_styles:
            if not self._style_cycle:
                self._reshuffle_styles()
            preset = self._style_cycle[self._style_index % len(self._style_cycle)]
            self._style_index += 1
            return preset
        weights = [preset.weight for preset in self.styles]
        total = sum(weights)
        pick = self.rng.random() * total
        cumulative = 0.0
        for preset, weight in zip(self.styles, weights):
            cumulative += weight
            if pick <= cumulative:
                return preset
        return self.styles[-1]

    def _choose_length_bucket(self, override: Optional[str] = None) -> LengthBucket:
        if override:
            for bucket in self.length_buckets:
                if bucket.name == override:
                    return bucket
        weights = [bucket.weight for bucket in self.length_buckets]
        total = sum(weights)
        pick = self.rng.random() * total
        cumulative = 0.0
        for bucket, weight in zip(self.length_buckets, weights):
            cumulative += weight
            if pick <= cumulative:
                return bucket
        return self.length_buckets[-1]

    def _choose_industry(self, override: Optional[str] = None) -> str:
        if override and override in self.industries:
            return override
        if self.industries:
            return self.rng.choice(self.industries)
        return override or "General"

    def _choose_role(self, override: Optional[str] = None) -> str:
        if override and override in self.roles:
            return override
        if self.roles:
            return self.rng.choice(self.roles)
        return override or "Professional"

    def _choose_hashtags(self, override: Optional[List[str]] = None) -> List[str]:
        if override:
            return [tag.lstrip("#").lower() for tag in override if str(tag).strip()]
        return list(self.rng.choice(self.hashtag_sets))

    def sample(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sample one generation parameter bundle."""
        overrides = overrides or {}

        style_preset = self._choose_style().resolve()
        style_overrides = {}

        for key in [
            "cta_present",
            "emoji_count",
            "cliche_count",
            "buzzword_count",
            "has_explicit_humility_phrase",
            "story_claims_verifiable",
            "length_bucket",
        ]:
            if key in style_preset:
                style_overrides[key] = style_preset[key]

        params: Dict[str, Any] = {}
        params.update(style_overrides)

        params["style_id"] = style_preset.get("style_id", style_preset.get("name", "style"))

        params["industry"] = self._choose_industry(overrides.get("industry"))
        params["role"] = self._choose_role(overrides.get("role"))

        length_bucket_name = overrides.get("length_bucket") or params.get("length_bucket")
        bucket = self._choose_length_bucket(length_bucket_name)
        params["length_bucket"] = bucket.name

        params["hashtags"] = self._choose_hashtags(overrides.get("hashtags"))

        params["prompt_id"] = overrides.get("prompt_id") or f"prompt-{params['style_id']}-{self._counter:06d}"
        params["seed"] = overrides.get("seed") or self.rng.randint(1, 2**31 - 1)
        params["response_schema_version"] = self.response_schema_version
        params["language"] = overrides.get("language", "en")

        params["cta_present"] = overrides.get("cta_present", params.get("cta_present", False))
        params["emoji_count"] = overrides.get("emoji_count", params.get("emoji_count", 0))
        params["cliche_count"] = overrides.get("cliche_count", params.get("cliche_count", 0))
        params["buzzword_count"] = overrides.get("buzzword_count", params.get("buzzword_count", 0))
        params["has_explicit_humility_phrase"] = overrides.get(
            "has_explicit_humility_phrase", params.get("has_explicit_humility_phrase", False)
        )
        params["story_claims_verifiable"] = overrides.get(
            "story_claims_verifiable", params.get("story_claims_verifiable", True)
        )
        params["opening_hook_constraint"] = overrides.get("opening_hook_constraint")

        self._counter += 1
        return params

    def reserve_real_samples(self, batch_size: int) -> int:
        if self.real_data_mix_ratio <= 0:
            return 0
        return min(batch_size, int(round(batch_size * self.real_data_mix_ratio)))

    @staticmethod
    def format_for_prompt(params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform parameters into serializable values for string template substitution."""
        formatted: Dict[str, Any] = {}
        for key, value in params.items():
            if key in {"cta_present", "has_explicit_humility_phrase", "story_claims_verifiable"}:
                formatted[key] = "true" if bool(value) else "false"
            elif key in {"emoji_count", "cliche_count", "buzzword_count", "seed"}:
                formatted[key] = int(value)
            elif key == "hashtags":
                formatted[key] = json.dumps(value or [])
            elif value is None:
                formatted[key] = ""
            else:
                formatted[key] = value
        return formatted


