import os
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
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
from .processors import Validator, Deduplicator, LabelTriager, DataSplitter, TextCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    """Main orchestrator for synthetic data generation."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'rb') as f:
            self.config = tomllib.load(f)
        
        self.generator = None
        self.judge = None
        self.processors = {}
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components from config."""
        
        # Initialize generator
        gen_config = self.config["models"]["generator"]
        gen_schema = self.config["schemas"]["generation"]
        gen_prompts = self.config["prompts"]["generator"]
        
        self.generator = self._build_generator(gen_config, gen_schema, gen_prompts)

        self.generator_backup = None
        backup_config = self.config["models"].get("generator_backup")
        if backup_config:
            backup_prompts = (
                self.config.get("prompts", {})
                .get("generator_backup", gen_prompts)
            )
            self.generator_backup = self._build_generator(
                backup_config, gen_schema, backup_prompts
            )
        
        # Initialize judge
        judge_config = self.config["models"]["judge"]
        judge_schema = self.config["schemas"]["judgment"]
        judge_prompts = self.config["prompts"]["judge"]
        
        judge_options = judge_config.get("options", {})
        judge_api_key = judge_config.get("api_key")
        judge_temperature = judge_config.get("temperature", 0.2)
        judge_num_samples = judge_config.get("num_samples", judge_options.get("num_samples", 2))
        judge_provider = judge_config["provider"]
        if judge_provider in {"openai", "anthropic", "gemini"}:
            self.judge = LLMJudge(
                schema=judge_schema,
                system_prompt=judge_prompts["system"],
                user_template=judge_prompts.get("user_template"),
                provider=judge_provider,
                model=judge_config["name"],
                temperature=judge_temperature,
                num_samples=judge_num_samples,
                api_key=judge_api_key,
                config=judge_options,
            )
        elif judge_provider in {"prometheus", "hf"}:
            self.judge = PrometheusJudge(
                schema=judge_schema,
                system_prompt=judge_prompts["system"],
                user_template=judge_prompts.get("user_template"),
                model_name=judge_config["name"],
                temperature=judge_temperature,
                num_samples=judge_num_samples,
                config=judge_options,
            )
        else:
            raise ValueError(f"Unsupported judge provider: {judge_provider}")

        self.pairwise_judge = None
        self.pairwise_settings: Dict[str, Any] = {}

        pairwise_config = (
            self.config.get("models", {}).get("judge_pairwise")
            or judge_config.get("pairwise")
        )
        pairwise_prompts = self.config.get("prompts", {}).get("judge_pairwise", {})
        pairwise_schema = self.config.get("schemas", {}).get("pairwise")

        if pairwise_config and pairwise_config.get("enabled", True):
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
            self.pairwise_settings = {
                "margin": pairwise_config.get("margin", 0.1),
                "adjustment": pairwise_config.get("adjustment", 0.15),
                "center": pairwise_config.get("center", 0.5),
                "max_pending": pairwise_config.get("max_pending", 4),
            }
        
        # Initialize processors
        self.processors["validator"] = Validator(gen_schema, judge_schema)
        self.processors["deduplicator"] = Deduplicator(
            threshold=self.config["pipeline"].get("dedup_threshold", 0.9)
        )
        self.processors["triager"] = LabelTriager()
        self.processors["splitter"] = DataSplitter(
            val_ratio=self.config["pipeline"].get("val_ratio", 0.15),
            test_ratio=self.config["pipeline"].get("test_ratio", 0.15)
        )
        self.processors["cleaner"] = TextCleaner()

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
    
    def generate(self, n_samples: int, output_path: str, **kwargs):
        """Generate synthetic examples."""
        logger.info(f"Generating {n_samples} examples...")
        
        examples = []
        batch_size = self.config["pipeline"].get("batch_size", 10)
        
        with tqdm(total=n_samples) as pbar:
            while len(examples) < n_samples:
                remaining = min(batch_size, n_samples - len(examples))

                try:
                    batch = self.generator.generate_batch(remaining, **kwargs)
                except Exception as exc:
                    logger.warning("Primary generator failed: %s", exc)
                    batch = []

                if not batch and self.generator_backup:
                    logger.info("Attempting backup generator after primary failure.")
                    try:
                        batch = self.generator_backup.generate_batch(remaining, **kwargs)
                    except Exception as backup_exc:
                        logger.error("Backup generator failed: %s", backup_exc)
                        raise

                if not batch:
                    raise RuntimeError("Failed to generate examples using configured generators.")

                examples.extend(batch)
                pbar.update(len(batch))
        
        # Save generated examples
        with open(output_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps({"generated": ex}) + "\n")
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return examples
    
    def judge(self, input_path: str, output_path: str):
        """Judge generated examples."""
        logger.info(f"Judging examples from {input_path}...")
        
        rows = []
        with open(input_path, 'r') as f:
            for line in f:
                rows.append(json.loads(line))
        
        judged = []
        borderline_queue: List[int] = []
        pairwise_enabled = self.pairwise_judge is not None
        pairwise_margin = self.pairwise_settings.get("margin", 0.1)
        pairwise_center = self.pairwise_settings.get("center", 0.5)
        pairwise_max_pending = self.pairwise_settings.get("max_pending", 4)
        
        for row in tqdm(rows):
            judgment = self.judge.judge_single(row.get("generated", row))
            judged_row = row.copy()
            judged_row["judgment"] = judgment
            judged.append(judged_row)
            
            if pairwise_enabled and isinstance(judgment, dict):
                score = judgment.get("cringe_prob")
                if isinstance(score, (int, float)) and abs(score - pairwise_center) <= pairwise_margin:
                    current_index = len(judged) - 1
                    borderline_queue.append(current_index)
                    if len(borderline_queue) > pairwise_max_pending:
                        borderline_queue.pop(0)
                    if len(borderline_queue) >= 2:
                        idx_a = borderline_queue.pop(0)
                        idx_b = borderline_queue.pop(0)
                        self._run_pairwise_resolution(judged, idx_a, idx_b)
        
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
            )
            self._apply_pairwise_adjustments(judged, idx_a, idx_b, result)
        except Exception as exc:
            logger.warning("Pairwise resolution failed: %s", exc)

    def _apply_pairwise_adjustments(
        self,
        judged: List[Dict[str, Any]],
        idx_a: int,
        idx_b: int,
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
        }
        entry_b = {
            "role": "B",
            "preferred": preferred,
            "confidence": confidence,
            "reason": reason,
            "partner_index": idx_a,
        }

        judged[idx_a].setdefault("pairwise", []).append(entry_a)
        judged[idx_b].setdefault("pairwise", []).append(entry_b)

        adjustment_scale = self.pairwise_settings.get("adjustment", 0.15)
        delta = adjustment_scale * confidence

        if preferred == "A":
            self._shift_probabilities(judged[idx_a].get("judgment"), delta)
            self._shift_probabilities(judged[idx_b].get("judgment"), -delta)
        elif preferred == "B":
            self._shift_probabilities(judged[idx_a].get("judgment"), -delta)
            self._shift_probabilities(judged[idx_b].get("judgment"), delta)

    @staticmethod
    def _shift_probabilities(judgment: Optional[Dict[str, Any]], delta: float, label_scale: float = 0.5) -> None:
        if not judgment or not isinstance(judgment, dict) or not delta:
            return

        def _clip(value: float) -> float:
            return max(0.0, min(1.0, value))

        score = judgment.get("cringe_prob")
        if isinstance(score, (int, float)):
            judgment["cringe_prob"] = _clip(score + delta)

        labels = judgment.get("labels")
        if isinstance(labels, dict):
            for label, value in labels.items():
                if isinstance(value, (int, float)):
                    labels[label] = _clip(value + delta * label_scale)

    def process(self, input_path: str, output_dir: str):
        """Process judged examples through cleaning pipeline."""
        logger.info(f"Processing examples from {input_path}...")
        
        rows = []
        with open(input_path, 'r') as f:
            for line in f:
                rows.append(json.loads(line))
        
        # Clean text
        rows = self.processors["cleaner"].clean_batch(rows)
        
        # Validate
        rows = self.processors["validator"].validate_batch(rows)
        
        # Deduplicate
        rows = self.processors["deduplicator"].deduplicate(rows)
        
        # Triage labels
        labels = self.config.get("task", {}).get("labels")
        if self.config["pipeline"].get("triage_labels", True):
            rows = self.processors["triager"].filter_clean(rows, labels)
        
        # Split data
        train, val, test = self.processors["splitter"].split(rows, labels)
        
        # Save splits
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            output_path = output_dir / f"{split_name}.jsonl"
            with open(output_path, 'w') as f:
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
