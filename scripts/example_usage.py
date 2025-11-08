#!/usr/bin/env python
"""Example script showing how to use the pipeline programmatically."""

import json
from pathlib import Path

from synth_data_pipeline import Pipeline

def main():
    # Load LinkedIn cringe config as example
    config_path = "configs/examples/linkedin_cringe.toml"
    
    # Initialize pipeline
    pipeline = Pipeline(config_path)
    
    # Generate 100 synthetic posts
    output_dir = Path("data/example_run")
    pipeline.run(
        n_samples=100,
        output_dir=output_dir,
        # These kwargs get passed to the generator
        industry="FinTech",
        role="PM",
        style_hints="humbleBragging and buzzwordOveruse"
    )
    
    # Load and inspect results
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Show first 3 examples
                break
            example = json.loads(line)
            print(f"\nExample {i+1}:")
            print(f"Text: {example['generated']['text'][:200]}...")
            print(f"Labels: {example['judgment']['labels']}")
            pairwise = example.get("pairwise")
            if pairwise:
                print(f"Pairwise metadata: {pairwise}")

if __name__ == "__main__":
    main()
