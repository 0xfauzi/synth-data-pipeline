import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Handle different Python versions for tomllib
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

from .generators import (
    AnthropicGenerator,
    GeminiGenerator,
    OpenAIGenerator,
    OutlinesGenerator,
)
from .judges import LLMJudge, EnsembleJudge, PairwiseJudge, PrometheusJudge
from .processors import (
    DataSplitter,
    Deduplicator,
    LabelTriager,
    ProbabilityCalibrator,
    TextCleaner,
    Validator,
)
from .utils.generation_sampler import GenerationSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    EMOJI_PATTERN = re.compile(
        r"[\U0001F300-\U0001F6FF"
        r"\U0001F900-\U0001F9FF"
        r"\U0001FA70-\U0001FAFF"
        r"\U00002600-\U000027BF]"
    )
except re.error:  # pragma: no cover - narrow Python builds
    EMOJI_PATTERN = re.compile(r"[\u2600-\u27BF]")

class Pipeline:
    """Main orchestrator for synthetic data generation."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'rb') as f:
            self.config = tomllib.load(f)

        self.generator = None
        self.generator_backup = None
        self.judge = None
        self.judge_backup = None
        self.processors: Dict[str, Any] = {}

        self.task_labels: List[str] = list(self.config.get("task", {}).get("labels", []))
        self.pipeline_config: Dict[str, Any] = self.config.get("pipeline", {})
        self.generation_config: Dict[str, Any] = self.config.get("generation_parameters", {})
        self.metadata_config: Dict[str, Any] = self.pipeline_config.get("metadata", {})
        self.emoji_config: Dict[str, Any] = self.pipeline_config.get("emoji_density", {})
        self.calibration_config: Dict[str, Any] = self.pipeline_config.get("calibration", {})
        self.sampler = GenerationSampler(self.generation_config)

        self.generator_backoff_config: Dict[str, Any] = self.config.get("models", {}).get("generator_backoff", {})
        self.generator_retry_limit: int = int(self.generator_backoff_config.get("retry_limit", 2))
        self.generator_initial_delay: float = float(self.generator_backoff_config.get("initial_delay_ms", 0)) / 1000.0
        self.generator_jitter: bool = bool(self.generator_backoff_config.get("jitter", False))

        self.generator_info: Dict[str, Any] = {}
        self.generator_backup_info: Optional[Dict[str, Any]] = None
        self.judge_info: Dict[str, Any] = {}
        self.judge_backup_info: Optional[Dict[str, Any]] = None

        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components from config."""
        
        # Initialize generator
        gen_config = self.config["models"]["generator"]
        gen_schema = self.config["schemas"]["generation"]
        gen_prompts = self.config["prompts"]["generator"]

        self.generator = self._build_generator(gen_config, gen_schema, gen_prompts)
        self.generator_info = {
            "provider": gen_config.get("provider"),
            "model": gen_config.get("name"),
        }

        backup_config = self.config["models"].get("generator_backup")
        if backup_config:
            backup_prompts = self.config.get("prompts", {}).get("generator_backup", gen_prompts)
            self.generator_backup = self._build_generator(backup_config, gen_schema, backup_prompts)
            self.generator_backup_info = {
                "provider": backup_config.get("provider"),
                "model": backup_config.get("name"),
            }

        # Initialize judge
        judge_config = self.config["models"]["judge"]
        judge_schema = self.config["schemas"]["judgment"]
        judge_prompts = self.config["prompts"]["judge"]

        self.judge, self.judge_info = self._build_judge(judge_config, judge_schema, judge_prompts)

        judge_backup_config = self.config["models"].get("judge_backup")
        if judge_backup_config:
            backup_prompts = self.config.get("prompts", {}).get("judge_backup", judge_prompts)
            try:
                self.judge_backup, self.judge_backup_info = self._build_judge(
                    judge_backup_config,
                    judge_schema,
                    backup_prompts,
                )
            except ValueError as exc:
                logger.warning("Judge backup disabled: %s", exc)
                self.judge_backup = None
                self.judge_backup_info = None

        self.pairwise_judge = None
        self.pairwise_settings: Dict[str, Any] = {}

        pairwise_config = self.config.get("models", {}).get("judge_pairwise") or judge_config.get("pairwise")
        pairwise_prompts = self.config.get("prompts", {}).get("judge_pairwise", {})
        pairwise_schema = self.config.get("schemas", {}).get("pairwise")
        pointwise_primary = bool(self.pipeline_config.get("pointwise_primary", True))
        gate_enabled = bool(self.pipeline_config.get("enable_pairwise_gate", True))

        if pairwise_config and pairwise_config.get("enabled", True) and gate_enabled:
            pairwise_provider = pairwise_config["provider"]
            pairwise_model = pairwise_config["name"]
            pairwise_temperature = pairwise_config.get("temperature", 0.3)
            pairwise_num_samples = pairwise_config.get("num_samples", 1)
            pairwise_api_key = pairwise_config.get("api_key")
            pairwise_options = pairwise_config.get("options", {})
            pairwise_system = pairwise_prompts.get("system", judge_prompts["system"])
            pairwise_template = pairwise_prompts.get("user_template")

            self.pairwise_judge = PairwiseJudge(
                system_prompt=pairwise_system,
                provider=pairwise_provider,
                model=pairwise_model,
                temperature=pairwise_temperature,
                num_samples=pairwise_num_samples,
                api_key=pairwise_api_key,
                schema=pairwise_schema,
                user_template=pairwise_template,
                config=pairwise_options,
            )
            trigger_policy = pairwise_config.get("trigger_policy", {})
            self.pairwise_settings = {
                "margin": pairwise_config.get("margin", 0.1),
                "adjustment": pairwise_config.get("adjustment", 0.15),
                "center": pairwise_config.get("center", 0.5),
                "max_pending": pairwise_config.get("max_pending", 4),
                "score_margin": trigger_policy.get("score_margin", pairwise_config.get("margin", 0.1)),
                "threshold_window": trigger_policy.get("threshold_window", pairwise_config.get("margin", 0.1)),
                "labels": trigger_policy.get("labels", ["overall"]),
                "pointwise_primary": pointwise_primary,
            }
        else:
            self.pairwise_settings = {"pointwise_primary": pointwise_primary}

        # Initialize processors
        self.processors["validator"] = Validator(gen_schema, judge_schema)

        dedup_cfg = self.pipeline_config.get("dedup", {})
        dedup_threshold = dedup_cfg.get("threshold", self.pipeline_config.get("dedup_threshold", 0.9))
        self.dedup_config_params = {
            "threshold": dedup_threshold,
            "num_perm": dedup_cfg.get("num_perm", 128),
            "n_gram": dedup_cfg.get("n_gram", 5),
            "shingle_level": dedup_cfg.get("shingle_level", "word"),
            "method": dedup_cfg.get("method", "minhash_lsh"),
        }
        self.processors["deduplicator"] = Deduplicator(**self.dedup_config_params)

        triage_engine = self.pipeline_config.get("triage_engine", "cleanlab")
        triage_threshold = self.pipeline_config.get("triage_threshold", 0.5)
        self.processors["triager"] = LabelTriager(threshold=triage_threshold, engine=triage_engine)

        splits_cfg = self.pipeline_config.get("splits", {})
        val_ratio = splits_cfg.get("val_ratio", self.pipeline_config.get("val_ratio", 0.15))
        test_ratio = splits_cfg.get("test_ratio", self.pipeline_config.get("test_ratio", 0.15))
        stratify = splits_cfg.get("stratify", True)
        split_seed = splits_cfg.get("seed", 42)
        self.processors["splitter"] = DataSplitter(
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify=stratify,
            seed=split_seed,
        )

        cleaner_config = self.pipeline_config.get("cleaning", {})
        self.processors["cleaner"] = TextCleaner(cleaner_config)

    def _build_generator(
        self,
        gen_config: Dict[str, Any],
        schema: Dict[str, Any],
        prompts: Dict[str, Any],
    ):
        provider = gen_config["provider"]
        options = gen_config.get("options", {})
        api_key = gen_config.get("api_key")
        temperature = gen_config.get("temperature", 0.9)
        system_prompt = prompts["system"]
        user_template = prompts["user_template"]

        if provider == "openai":
            return OpenAIGenerator(
                schema=schema,
                system_prompt=system_prompt,
                user_template=user_template,
                model=gen_config["name"],
                temperature=temperature,
                api_key=api_key,
                config=options,
            )
        if provider == "anthropic":
            return AnthropicGenerator(
                schema=schema,
                system_prompt=system_prompt,
                user_template=user_template,
                model=gen_config["name"],
                temperature=temperature,
                api_key=api_key,
                config=options,
            )
        if provider == "gemini":
            return GeminiGenerator(
                schema=schema,
                system_prompt=system_prompt,
                user_template=user_template,
                model=gen_config["name"],
                temperature=temperature,
                api_key=api_key,
                config=options,
            )
        if provider == "outlines":
            return OutlinesGenerator(
                schema=schema,
                system_prompt=system_prompt,
                user_template=user_template,
                model_name=gen_config["name"],
                backend=gen_config.get("backend", "transformers"),
                config=options,
            )
        raise ValueError(f"Unsupported generator provider: {provider}")

    def _build_judge(
        self,
        judge_config: Dict[str, Any],
        schema: Dict[str, Any],
        prompts: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        judge_options = judge_config.get("options", {}) or {}
        judge_api_key = judge_config.get("api_key")
        judge_temperature = judge_config.get("temperature", judge_options.get("temperature", 0.2))
        judge_num_samples = judge_config.get("num_samples", judge_options.get("num_samples", 2))
        aggregation = judge_config.get("aggregation") or judge_options.get("aggregation") or "mean"
        provider = judge_config["provider"]

        system_prompt = prompts["system"]
        user_template = prompts.get("user_template")

        if provider in {"openai", "anthropic", "gemini", "openrouter"}:
            judge = LLMJudge(
                schema=schema,
                system_prompt=system_prompt,
                user_template=user_template,
                provider=provider,
                model=judge_config["name"],
                temperature=judge_temperature,
                num_samples=judge_num_samples,
                api_key=judge_api_key,
                config=judge_options,
                aggregation=aggregation,
            )
        elif provider in {"prometheus", "hf"}:
            judge = PrometheusJudge(
                schema=schema,
                system_prompt=system_prompt,
                user_template=user_template,
                model_name=judge_config["name"],
                temperature=judge_temperature,
                num_samples=judge_num_samples,
                config=judge_options,
            )
        else:
            raise ValueError(f"Unsupported judge provider: {provider}")

        info = {
            "provider": provider,
            "model": judge_config.get("name"),
            "aggregation": aggregation,
            "num_samples": judge_num_samples,
            "temperature": judge_temperature,
        }
        return judge, info
    
    def generate(self, n_samples: int, output_path: str, **kwargs):
        """Generate synthetic examples."""
        logger.info(f"Generating {n_samples} examples...")

        examples: List[Dict[str, Any]] = []
        batch_size = self.pipeline_config.get("batch_size", 10)
        overrides = kwargs or {}
        real_mix_ratio = self.sampler.real_data_mix_ratio
        warned_real_mix = False

        with tqdm(total=n_samples) as pbar:
            while len(examples) < n_samples:
                remaining = min(batch_size, n_samples - len(examples))
                real_reserved = self.sampler.reserve_real_samples(remaining)
                if real_reserved and not warned_real_mix:
                    logger.warning(
                        "Real data mix ratio %.2f requested but no real data source configured; skipping.",
                        real_mix_ratio,
                    )
                    warned_real_mix = True

                for _ in range(remaining):
                    params = self.sampler.sample(overrides)
                    formatted_kwargs = GenerationSampler.format_for_prompt(params)

                    try:
                        generated_output, generator_used = self._generate_with_retry(formatted_kwargs)
                    except Exception as exc:
                        logger.error("Generation failed after retries: %s", exc)
                        raise

                    row = {"generated": generated_output}
                    self._attach_metadata(
                        row,
                        generator_used=generator_used,
                        controls=params,
                    )
                    examples.append(row)
                    pbar.update(1)

        # Save generated examples
        with open(output_path, "w") as f:
            for row in examples:
                f.write(json.dumps(row) + "\n")

        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return examples
    
    def _generate_with_retry(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        try:
            result = self._attempt_generation("primary", self.generator, payload)
            return result, "primary"
        except Exception as primary_exc:
            if not self.generator_backup:
                raise
            logger.warning("Primary generator exhausted retries, switching to backup: %s", primary_exc)
            result = self._attempt_generation("backup", self.generator_backup, payload)
            return result, "backup"

    def _attempt_generation(self, label: str, generator: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
        retries = 0
        delay = self.generator_initial_delay
        while True:
            try:
                return generator.generate_single(**payload)
            except Exception as exc:
                retries += 1
                if retries > self.generator_retry_limit:
                    raise
                sleep_time = self._with_jitter(delay)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                delay = delay * 2 if delay else 0
                logger.warning("%s generator attempt %s/%s failed: %s", label, retries, self.generator_retry_limit, exc)

    def _with_jitter(self, delay: float) -> float:
        if delay <= 0:
            return 0.0
        if not self.generator_jitter:
            return delay
        jitter_factor = 0.3 * (random.random() - 0.5) * 2  # ±30%
        return max(0.0, delay * (1 + jitter_factor))

    def _invoke_judge_with_backup(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            result = self.judge.judge_single(example)
            info = dict(self.judge_info)
            return result, info
        except Exception as primary_exc:
            if not self.judge_backup:
                raise
            logger.warning("Primary judge failed, falling back to backup: %s", primary_exc)
            result = self.judge_backup.judge_single(example)
            info = dict(self.judge_backup_info or {})
            return result, info

    def _enrich_judgment(
        self,
        row: Dict[str, Any],
        judgment: Dict[str, Any],
        judge_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(judgment, dict):
            return

        generated = row.get("generated", {})
        for key in ["prompt_id", "style_id", "seed", "response_schema_version"]:
            if key in generated and key not in judgment:
                judgment[key] = generated[key]

        if "judge_meta" not in judgment or not isinstance(judgment["judge_meta"], dict):
            judgment["judge_meta"] = {}
        if judge_info:
            judgment["judge_meta"].setdefault("model", judge_info.get("model"))
            judgment["judge_meta"].setdefault("temperature", judge_info.get("temperature"))
            judgment["judge_meta"].setdefault("num_samples", judge_info.get("num_samples"))
            judgment["judge_meta"].setdefault("aggregation", judge_info.get("aggregation"))
            judgment["judge_meta"].setdefault("provider", judge_info.get("provider"))
        if "seed" not in judgment["judge_meta"]:
            judgment["judge_meta"]["seed"] = judgment.get("seed")

        if "calibration_applied" not in judgment:
            judgment["calibration_applied"] = False
        if "calibration_version" not in judgment and self.calibration_config.get("method"):
            judgment["calibration_version"] = self.calibration_config["method"]

    def _collect_pairwise_candidates(
        self,
        judgment: Dict[str, Any],
        labels: List[str],
        threshold_window: float,
        center: float,
    ) -> List[Tuple[str, float]]:
        candidates: List[Tuple[str, float]] = []
        label_probs = judgment.get("labels", {}) if isinstance(judgment.get("labels"), dict) else {}
        for label in labels:
            if label == "overall":
                score = judgment.get("cringe_prob")
            else:
                score = label_probs.get(label)
            if isinstance(score, (int, float)) and abs(float(score) - center) <= threshold_window:
                candidates.append((label, float(score)))
        return candidates

    def _attach_metadata(
        self,
        row: Dict[str, Any],
        generator_used: Optional[str] = None,
        controls: Optional[Dict[str, Any]] = None,
        judge_used: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.metadata_config:
            return

        metadata = row.setdefault("metadata", {})
        generated = row.get("generated", {})

        if self.metadata_config.get("record_prompt"):
            prompt_meta = metadata.setdefault("prompt", {})
            for key in ["prompt_id", "style_id", "seed", "response_schema_version"]:
                value = generated.get(key)
                if value is None and controls:
                    value = controls.get(key)
                if value is not None:
                    prompt_meta[key] = value

        if self.metadata_config.get("record_style"):
            style_meta = metadata.setdefault("style_controls", {})
            for key in [
                "cta_present",
                "emoji_count",
                "cliche_count",
                "buzzword_count",
                "has_explicit_humility_phrase",
                "story_claims_verifiable",
                "length_bucket",
            ]:
                if controls and key in controls:
                    style_meta[key] = controls[key]
                elif generated and key in generated:
                    style_meta[key] = generated[key]

        if self.metadata_config.get("record_model_versions"):
            model_meta = metadata.setdefault("models", {})
            if generator_used:
                info = self.generator_info
                variant = generator_used
                if generator_used == "backup" and self.generator_backup_info:
                    info = self.generator_backup_info
                if info:
                    model_meta["generator"] = {
                        "provider": info.get("provider"),
                        "model": info.get("model"),
                        "variant": variant,
                    }
            if judge_used:
                model_meta["judge"] = {
                    "provider": judge_used.get("provider"),
                    "model": judge_used.get("model"),
                    "aggregation": judge_used.get("aggregation"),
                    "num_samples": judge_used.get("num_samples"),
                    "temperature": judge_used.get("temperature"),
                }

    def _compute_emoji_density(self, rows: List[Dict[str, Any]]) -> None:
        if not self.emoji_config.get("compute"):
            return
        normalization = self.emoji_config.get("normalization", "per_word")
        for row in rows:
            generated = row.get("generated")
            if not isinstance(generated, dict):
                continue
            text = generated.get("text")
            if not isinstance(text, str):
                continue
            emoji_count = len(EMOJI_PATTERN.findall(text))
            if normalization == "per_char":
                denominator = max(1, len(text))
            else:
                tokens = re.findall(r"\w+", text)
                denominator = max(1, len(tokens))
            generated["emoji_density"] = round(emoji_count / denominator, 6)
            generated.setdefault("emoji_count", emoji_count)

    def _apply_calibration(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.calibration_config.get("enabled"):
            for row in rows:
                self._mark_calibration(row, applied=False)
            return rows

        params_path = self.calibration_config.get("parameters_path")
        version_hint = (
            self.calibration_config.get("version")
            or self.calibration_config.get("method")
            or None
        )

        if not params_path:
            logger.info("Calibration enabled but no parameters_path provided; skipping application.")
            for row in rows:
                self._mark_calibration(row, applied=False, version=version_hint)
            return rows

        path = Path(params_path)
        if not path.exists():
            logger.warning("Calibration parameters file %s not found; skipping calibration.", path)
            for row in rows:
                self._mark_calibration(row, applied=False, version=version_hint)
            return rows

        calibrator = ProbabilityCalibrator.load(self.task_labels, path)
        calibrated_rows = calibrator.transform(rows)
        version = version_hint or path.stem
        for row in calibrated_rows:
            self._mark_calibration(row, applied=True, version=version)
        return calibrated_rows

    def _mark_calibration(
        self,
        row: Dict[str, Any],
        applied: bool,
        version: Optional[str] = None,
    ) -> None:
        judgment = row.get("judgment")
        if not isinstance(judgment, dict):
            return
        judgment["calibration_applied"] = bool(applied)
        if version:
            judgment["calibration_version"] = version

    def judge(self, input_path: str, output_path: str):
        """Judge generated examples."""
        logger.info(f"Judging examples from {input_path}...")

        rows = []
        with open(input_path, 'r') as f:
            for line in f:
                rows.append(json.loads(line))
        
        judged = []
        pairwise_enabled = bool(self.pairwise_judge)
        pending_by_label: Dict[str, deque] = defaultdict(deque)
        score_margin = self.pairwise_settings.get("score_margin", self.pairwise_settings.get("margin", 0.1))
        threshold_window = self.pairwise_settings.get("threshold_window", self.pairwise_settings.get("margin", 0.1))
        pairwise_center = self.pairwise_settings.get("center", 0.5)
        pairwise_max_pending = self.pairwise_settings.get("max_pending", 4)
        trigger_labels = self.pairwise_settings.get("labels", ["overall"])

        for row in tqdm(rows):
            judgment, judge_used_info = self._invoke_judge_with_backup(row.get("generated", row))
            self._enrich_judgment(row, judgment, judge_used_info)
            judged_row = row.copy()
            judged_row["judgment"] = judgment
            self._attach_metadata(judged_row, judge_used=judge_used_info)
            judged.append(judged_row)
            
            if pairwise_enabled and isinstance(judgment, dict):
                candidates = self._collect_pairwise_candidates(
                    judgment,
                    trigger_labels,
                    threshold_window,
                    pairwise_center,
                )
                current_index = len(judged) - 1
                for label, score in candidates:
                    queue = pending_by_label[label]
                    queue.append((current_index, score))
                    while len(queue) > pairwise_max_pending:
                        queue.popleft()
                    if len(queue) >= 2:
                        idx_a, score_a = queue.popleft()
                        idx_b, score_b = queue.popleft()
                        if abs(score_a - score_b) <= score_margin:
                            self._run_pairwise_resolution(judged, idx_a, idx_b, label)
                        else:
                            queue.appendleft((idx_a, score_a))
                            queue.append((idx_b, score_b))
        
        # Save judged examples
        with open(output_path, 'w') as f:
            for row in judged:
                f.write(json.dumps(row) + "\n")
        
        logger.info(f"Saved {len(judged)} judged examples to {output_path}")
        return judged
    
    def _run_pairwise_resolution(
        self,
        judged: List[Dict[str, Any]],
        idx_a: int,
        idx_b: int,
        label: str,
    ) -> None:
        if not self.pairwise_judge:
            return
        try:
            candidate_a = judged[idx_a].get("generated", judged[idx_a])
            candidate_b = judged[idx_b].get("generated", judged[idx_b])
            result = self.pairwise_judge.compare(
                candidate_a,
                candidate_b,
                context={"pair_indices": [idx_a, idx_b]},
                label=label,
            )
            self._apply_pairwise_adjustments(judged, idx_a, idx_b, label, result)
        except Exception as exc:
            logger.warning("Pairwise resolution failed: %s", exc)

    def _apply_pairwise_adjustments(
        self,
        judged: List[Dict[str, Any]],
        idx_a: int,
        idx_b: int,
        label: str,
        result: Dict[str, Any],
    ) -> None:
        preferred = result.get("preferred")
        confidence_raw = result.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        reason = result.get("reason")

        entry_a = {
            "role": "A",
            "preferred": preferred,
            "confidence": confidence,
            "reason": reason,
            "partner_index": idx_b,
            "label": label,
        }
        entry_b = {
            "role": "B",
            "preferred": preferred,
            "confidence": confidence,
            "reason": reason,
            "partner_index": idx_a,
            "label": label,
        }

        judged[idx_a].setdefault("pairwise", []).append(entry_a)
        judged[idx_b].setdefault("pairwise", []).append(entry_b)

        adjustment_scale = self.pairwise_settings.get("adjustment", 0.15)
        delta = adjustment_scale * confidence

        if preferred == "A":
            self._shift_probabilities(judged[idx_a].get("judgment"), delta, label=label)
            self._shift_probabilities(judged[idx_b].get("judgment"), -delta, label=label)
        elif preferred == "B":
            self._shift_probabilities(judged[idx_a].get("judgment"), -delta, label=label)
            self._shift_probabilities(judged[idx_b].get("judgment"), delta, label=label)

    @staticmethod
    def _shift_probabilities(
        judgment: Optional[Dict[str, Any]],
        delta: float,
        label: Optional[str] = None,
        label_scale: float = 0.5,
    ) -> None:
        if not judgment or not isinstance(judgment, dict) or not delta:
            return

        def _clip(value: float) -> float:
            return max(0.0, min(1.0, value))

        score = judgment.get("cringe_prob")
        if (label in (None, "overall") or label is None) and isinstance(score, (int, float)):
            judgment["cringe_prob"] = _clip(score + delta)

        labels = judgment.get("labels")
        if isinstance(labels, dict):
            if label and label != "overall":
                value = labels.get(label)
                if isinstance(value, (int, float)):
                    labels[label] = _clip(value + delta * label_scale)
            else:
                for key, value in labels.items():
                    if isinstance(value, (int, float)):
                        labels[key] = _clip(value + delta * label_scale)

    def process(self, input_path: str, output_dir: str):
        """Process judged examples through cleaning pipeline."""
        logger.info(f"Processing examples from {input_path}...")

        rows: List[Dict[str, Any]] = []
        with open(input_path, "r") as f:
            for line in f:
                rows.append(json.loads(line))

        cleaner: TextCleaner = self.processors["cleaner"]
        rows = cleaner.clean_batch(rows)

        self._compute_emoji_density(rows)

        validator: Validator = self.processors["validator"]
        rows = validator.validate_batch(rows)

        deduplicator = Deduplicator(**getattr(self, "dedup_config_params", {}))
        rows = deduplicator.deduplicate(rows)

        if self.pipeline_config.get("triage_labels", True):
            triager: LabelTriager = self.processors["triager"]
            rows = triager.filter_clean(rows, self.task_labels)

        rows = self._apply_calibration(rows)

        splitter: DataSplitter = self.processors["splitter"]
        train, val, test = splitter.split(rows, self.task_labels or None)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            output_path = output_dir / f"{split_name}.jsonl"
            with open(output_path, "w") as f:
                for row in split_data:
                    f.write(json.dumps(row) + "\n")
            logger.info(f"Saved {len(split_data)} examples to {output_path}")

        return train, val, test
    
    def run(self, n_samples: int, output_dir: str, **kwargs):
        """Run complete pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate
        gen_path = output_dir / "generated.jsonl"
        self.generate(n_samples, gen_path, **kwargs)
        
        # Judge
        judged_path = output_dir / "judged.jsonl"
        self.judge(gen_path, judged_path)
        
        # Process
        self.process(judged_path, output_dir)
        
        logger.info(f"Pipeline complete! Data saved to {output_dir}")
