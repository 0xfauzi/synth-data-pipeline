import json
import logging
import sys
from pathlib import Path

import click

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 support
    import tomli as tomllib

from .pipeline import Pipeline
from .processors import ProbabilityCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Synthetic Data Pipeline CLI."""
    pass

def _collect_overrides(**kwargs):
    overrides = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        overrides[key] = value
    return overrides


@cli.command()
@click.option('--config', '-c', required=True, help='Path to config TOML file')
@click.option('--output', '-o', default='data/output', help='Output directory')
@click.option('--samples', '-n', default=100, help='Number of samples to generate')
@click.option('--industry', type=str, help='Override industry sampling')
@click.option('--role', type=str, help='Override role sampling')
@click.option('--style-id', type=str, help='Force a specific style preset')
@click.option('--length-bucket', type=str, help='Force a specific length bucket')
@click.option('--hashtags', multiple=True, help='Override hashtags (repeatable)')
@click.option('--seed', type=int, help='Set fixed generation seed')
def run(config, output, samples, industry, role, style_id, length_bucket, hashtags, seed):
    """Run complete pipeline."""
    pipeline = Pipeline(config)
    overrides = _collect_overrides(
        industry=industry,
        role=role,
        style_id=style_id,
        length_bucket=length_bucket,
        seed=seed,
        hashtags=list(hashtags) if hashtags else None,
    )
    pipeline.run(samples, output, **overrides)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config TOML file')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--samples', '-n', default=100, help='Number of samples to generate')
@click.option('--industry', type=str, help='Override industry sampling')
@click.option('--role', type=str, help='Override role sampling')
@click.option('--style-id', type=str, help='Force a specific style preset')
@click.option('--length-bucket', type=str, help='Force a specific length bucket')
@click.option('--hashtags', multiple=True, help='Override hashtags (repeatable)')
@click.option('--seed', type=int, help='Set fixed generation seed')
def generate(config, output, samples, industry, role, style_id, length_bucket, hashtags, seed):
    """Generate synthetic examples only."""
    pipeline = Pipeline(config)
    overrides = _collect_overrides(
        industry=industry,
        role=role,
        style_id=style_id,
        length_bucket=length_bucket,
        seed=seed,
        hashtags=list(hashtags) if hashtags else None,
    )
    pipeline.generate(samples, output, **overrides)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config TOML file')
@click.option('--input', '-i', required=True, help='Input file with generated examples')
@click.option('--output', '-o', required=True, help='Output file path')
def judge(config, input, output):
    """Judge generated examples only."""
    pipeline = Pipeline(config)
    pipeline.judge(input, output)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config TOML file')
@click.option('--input', '-i', required=True, help='Input file with judged examples')
@click.option('--output', '-o', required=True, help='Output directory')
def process(config, input, output):
    """Process judged examples only."""
    pipeline = Pipeline(config)
    pipeline.process(input, output)


@cli.command()
@click.option('--config', '-c', required=True, help='Path to config TOML file')
@click.option('--judged', '-j', required=True, help='Path to judged predictions (JSONL)')
@click.option('--ground-truth', '-g', required=True, help='Path to ground truth labels (JSONL)')
@click.option('--output', '-o', required=True, help='Path to write calibrated judgments (JSONL)')
@click.option('--method', '-m', type=click.Choice(["isotonic", "platt"]), default="isotonic")
def calibrate(config, judged, ground_truth, output, method):
    """Calibrate probabilities using human labeled data."""
    with open(config, 'rb') as f:
        cfg = tomllib.load(f)
    labels = cfg.get("task", {}).get("labels", [])

    with open(judged, 'r') as pred_file:
        predictions = [json.loads(line) for line in pred_file]

    with open(ground_truth, 'r') as truth_file:
        truths = [json.loads(line) for line in truth_file]

    calibrator = ProbabilityCalibrator(labels=labels, method=method)
    calibrator.fit(predictions, truths)

    calibrated = calibrator.transform(predictions)
    output_path = Path(output)
    with output_path.open('w') as out_file:
        for row in calibrated:
            out_file.write(json.dumps(row) + "\n")

    params_path = output_path.with_suffix(output_path.suffix + ".calibration.json")
    calibrator.save(params_path)

    click.echo(f"Calibrated judgments saved to {output_path}")
    click.echo(f"Calibration parameters saved to {params_path}")

def main():
    cli()

if __name__ == '__main__':
    main()
