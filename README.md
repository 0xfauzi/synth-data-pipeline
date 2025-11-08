# Synthetic Data Pipeline

A flexible, reusable pipeline for generating synthetic training data using LLMs. Built for multi-label classification tasks but adaptable to any supervised learning scenario.

## Features

- **Schema-Driven**: Define your data structure with JSON schemas
- **Multi-Provider Support**: Works with OpenAI, Anthropic, or local models (via Outlines)
- **Quality Control**: Built-in validation, deduplication, and label issue detection
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
```

## Architecture

```
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

[models.generator]
provider = "openai"  # or "anthropic", "outlines"
name = "gpt-4"

[models.judge]
provider = "openai"
name = "gpt-4"

[pipeline]
dedup_threshold = 0.9
validation_strict = true
balance_labels = true
```

## Components

### Generators
- `OpenAIGenerator`: Uses OpenAI's structured outputs
- `AnthropicGenerator`: Uses Claude's structured generation  
- `OutlinesGenerator`: Local models with schema constraints
- `TemplateGenerator`: Rule-based generation for simple cases

### Judges
- `LLMJudge`: Uses any LLM for labeling
- `EnsembleJudge`: Combines multiple judges
- `HeuristicJudge`: Rule-based labeling

### Processors
- `Validator`: Schema and content validation
- `Deduplicator`: MinHash-based near-duplicate removal
- `LabelTriager`: Detects label quality issues
- `DataSplitter`: Stratified train/val/test splits

## Examples

See `configs/examples/` for complete configurations:
- `linkedin_cringe.toml`: Multi-label social media classification
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

## Citation

If you use this pipeline in your research, please cite:
```bibtex
@software{synth_data_pipeline,
  title = {Synthetic Data Pipeline},
  year = {2024},
  url = {https://github.com/yourusername/synth-data-pipeline}
}
```
