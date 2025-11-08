# Synthetic Data Pipeline

A flexible, reusable pipeline for generating synthetic training data using LLMs. Built for multi-label classification tasks but adaptable to any supervised learning scenario.

## Features

- **Schema-Driven**: Define your data structure with JSON schemas
- **Arena-Proven Models**: Built-in adapters for Gemini 2.5 Pro, Claude 4.5, GPT-4.1, Qwen2.5, and Prometheus 2
- **Bias-Resistant Judging**: Multi-sample scoring, optional pairwise tiebreaks, and cross-family model support
- **Quality Control**: Validation, cleaning, deduplication, label triage, and probability calibration helpers
- **Configurable**: Swap components, prompts, and strategies via TOML configs
- **Production Ready**: Logging, error handling, and batch processing

## Quick Start

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Set up environment
cp .env.example .env
# Add your API keys to .env

# 4. Configure your task (or use an example)
cp configs/examples/linkedin_cringe.toml configs/my_task.toml
# Edit my_task.toml with your schemas and prompts

# 5. Run the pipeline
uv run synth-pipeline run --config configs/my_task.toml --output data/my_dataset

# 6. Optionally run specific stages
uv run synth-pipeline generate --config configs/my_task.toml --num-samples 1000
uv run synth-pipeline judge --config configs/my_task.toml --input data/generated.jsonl
uv run synth-pipeline process --config configs/my_task.toml --input data/judged.jsonl

# 7. (Optional) Calibrate probabilities with human labels
uv run synth-pipeline calibrate \
  --config configs/my_task.toml \
  --judged data/my_dataset/judged.jsonl \
  --ground-truth data/my_dataset/human_labels.jsonl \
  --output data/my_dataset/judged_calibrated.jsonl
```

## Architecture

```

## Recommended Model Recipes (Nov 2025)

- **API-first**: `Gemini 2.5 Pro` (generator) with a `Claude 4.5 Sonnet` backup plus `GPT-4.1` for judging (two samples averaged) and `Claude 4.5 Sonnet` for pairwise tiebreaks. All three support JSON-schema output/tool schemas and keep generator/judge families distinct to reduce bias.
- **Fully open-source**: `Qwen2.5-72B-Instruct` via Outlines for generation + `Prometheus 2 (7B)` for judging. Pairwise resolution can be skipped or delegated to a lightweight API model if desired.

Both recipes align with the latest Chatbot Arena, G-Eval, and Prometheus 2 research cited by LMSYS, Anthropic, and EMNLP 2024/2025 studies. The example configs under `configs/examples/` encode these defaults.
Generate (LLM creates examples matching schema)
    ↓
Judge (Another LLM assigns labels/scores)
    ↓
Validate (Schema checking)
    ↓
Clean (Text preprocessing)
    ↓
Deduplicate (Remove near-duplicates)
    ↓
Triage (Flag suspicious labels)
    ↓
Split (Create train/val/test sets)
```

## Configuration

Each pipeline is configured with a TOML file:

```toml
[task]
name = "my_classification_task"
type = "multi_label"  # or "multi_class", "regression"

[schemas.generation]
type = "object"
required = ["text", "category"]
# JSON schema for generated examples

[schemas.judgment]  
type = "object"
required = ["labels", "confidence"]
# JSON schema for labels/scores

[prompts.generator]
system = "Your generation instructions..."
user_template = "Generate an example with: {parameters}"

[prompts.judge]
system = "Your judging instructions..."
user_template = "Score this example: {text}"

[models.generator]
provider = "gemini"  # gemini, openai, anthropic, outlines
name = "gemini-2.5-pro-latest"
temperature = 0.7

[models.generator.options]
max_output_tokens = 1800

[models.generator_backup]
provider = "anthropic"
name = "claude-4.5-sonnet"
temperature = 0.7

[models.generator_backup.options]
max_output_tokens = 2000

[models.judge]
provider = "openai"  # openai, anthropic, gemini, prometheus
name = "gpt-4.1"
temperature = 0.3
num_samples = 2

[models.judge.options]
max_output_tokens = 900
top_p = 0.9

[models.judge.pairwise]
enabled = true
provider = "anthropic"
name = "claude-4.5-sonnet"
temperature = 0.3

[schemas.pairwise]
type = "object"
required = ["preferred", "confidence", "reason"]

[pipeline]
dedup_threshold = 0.9
validation_strict = true
balance_labels = true
```

## Components

### Generators
- `GeminiGenerator`: Gemini 2.5 with JSON Schema mode
- `OpenAIGenerator`: GPT-4.1/4.1-mini structured outputs
- `AnthropicGenerator`: Claude 4.1 structured tool outputs  
- `OutlinesGenerator`: Local models (e.g., Qwen2.5) with schema constraints

### Judges
- `LLMJudge`: GPT-4.x, Claude, or Gemini with structured JSON + multi-sample averaging
- `PrometheusJudge`: Local Prometheus 2 evaluator via Transformers
- `PairwiseJudge`: Optional pairwise tiebreaks (G-Eval style prompts)
- `EnsembleJudge`: Combines multiple judges

### Processors
- `Validator`: Schema and content validation
- `Deduplicator`: MinHash-based near-duplicate removal
- `LabelTriager`: Detects label quality issues
- `DataSplitter`: Stratified train/val/test splits
- `ProbabilityCalibrator`: Fits isotonic/Platt scaling from human labels

## Examples

See `configs/examples/` for complete configurations:
- `linkedin_cringe.toml`: Multi-label social media classification
- `linkedin_cringe_open.toml`: Fully open-source (Qwen + Prometheus) variant
- `customer_intent.toml`: Multi-class intent classification
- `toxicity_scoring.toml`: Regression for content moderation
- `code_quality.toml`: Multi-aspect code evaluation

## Advanced Usage

### Custom Generators

```python
from src.generators.base import BaseGenerator

class MyGenerator(BaseGenerator):
    def generate(self, **kwargs):
        # Your custom generation logic
        return {...}
```

### Ensemble Judging

```python
judges = [
    LLMJudge(model="gpt-4"),
    LLMJudge(model="claude-3"),
]
ensemble = EnsembleJudge(judges, aggregation="mean")
```

### Pipeline Hooks

```python
from src.pipeline import Pipeline

pipeline = Pipeline(config)
pipeline.add_hook('post_generate', my_custom_function)
pipeline.run()
```

### Probability Calibration

Once you have a small batch of human-labeled data, calibrate model probabilities so that a score like `0.7` really means ~70% positive:

```bash
uv run synth-pipeline calibrate \
  --config configs/my_task.toml \
  --judged data/my_dataset/judged.jsonl \
  --ground-truth data/my_dataset/human_labels.jsonl \
  --output data/my_dataset/judged_calibrated.jsonl \
  --method isotonic
```

The command writes calibrated judgments plus a `*.calibration.json` file you can load later with `ProbabilityCalibrator.load`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_generators.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT

