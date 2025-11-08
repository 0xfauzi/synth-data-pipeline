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

from .generators import OpenAIGenerator, AnthropicGenerator, OutlinesGenerator
from .judges import LLMJudge, EnsembleJudge
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
        
        if gen_config["provider"] == "openai":
            self.generator = OpenAIGenerator(
                schema=gen_schema,
                system_prompt=gen_prompts["system"],
                user_template=gen_prompts["user_template"],
                model=gen_config["name"],
                temperature=gen_config.get("temperature", 0.9)
            )
        elif gen_config["provider"] == "anthropic":
            self.generator = AnthropicGenerator(
                schema=gen_schema,
                system_prompt=gen_prompts["system"],
                user_template=gen_prompts["user_template"],
                model=gen_config["name"],
                temperature=gen_config.get("temperature", 0.9)
            )
        elif gen_config["provider"] == "outlines":
            self.generator = OutlinesGenerator(
                schema=gen_schema,
                system_prompt=gen_prompts["system"],
                user_template=gen_prompts["user_template"],
                model_name=gen_config["name"]
            )
        
        # Initialize judge
        judge_config = self.config["models"]["judge"]
        judge_schema = self.config["schemas"]["judgment"]
        judge_prompts = self.config["prompts"]["judge"]
        
        self.judge = LLMJudge(
            schema=judge_schema,
            system_prompt=judge_prompts["system"],
            provider=judge_config["provider"],
            model=judge_config["name"],
            temperature=judge_config.get("temperature", 0.2)
        )
        
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
    
    def generate(self, n_samples: int, output_path: str, **kwargs):
        """Generate synthetic examples."""
        logger.info(f"Generating {n_samples} examples...")
        
        examples = []
        batch_size = self.config["pipeline"].get("batch_size", 10)
        
        with tqdm(total=n_samples) as pbar:
            while len(examples) < n_samples:
                batch = self.generator.generate_batch(
                    min(batch_size, n_samples - len(examples)),
                    **kwargs
                )
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
        for row in tqdm(rows):
            judgment = self.judge.judge_single(row.get("generated", row))
            judged_row = row.copy()
            judged_row["judgment"] = judgment
            judged.append(judged_row)
        
        # Save judged examples
        with open(output_path, 'w') as f:
            for row in judged:
                f.write(json.dumps(row) + "\n")
        
        logger.info(f"Saved {len(judged)} judged examples to {output_path}")
        return judged
    
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
