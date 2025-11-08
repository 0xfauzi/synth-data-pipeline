import click
import logging
from pathlib import Path
from .pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Synthetic Data Pipeline CLI."""
    pass

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config TOML file')
@click.option('--output', '-o', default='data/output', help='Output directory')
@click.option('--samples', '-n', default=100, help='Number of samples to generate')
def run(config, output, samples):
    """Run complete pipeline."""
    pipeline = Pipeline(config)
    pipeline.run(samples, output)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config TOML file')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--samples', '-n', default=100, help='Number of samples to generate')
def generate(config, output, samples):
    """Generate synthetic examples only."""
    pipeline = Pipeline(config)
    pipeline.generate(samples, output)

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

def main():
    cli()

if __name__ == '__main__':
    main()
