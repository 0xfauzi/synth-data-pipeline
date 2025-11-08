import importlib.util
from pathlib import Path
import sys

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "synth_data_pipeline" / "generators" / "base.py"
spec = importlib.util.spec_from_file_location("generator_base", MODULE_PATH)
generator_base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["generator_base"] = generator_base
spec.loader.exec_module(generator_base)  # type: ignore[attr-defined]
BaseGenerator = generator_base.BaseGenerator

pytest.importorskip("jsonschema")

def test_base_generator_validation():
    """Test that base generator validates schema."""
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"}
        }
    }
    
    class TestGenerator(BaseGenerator):
        def generate_single(self, **kwargs):
            return {"text": "test"}
    
    gen = TestGenerator(schema)
    assert gen.schema == schema
    assert gen.validate_output({"text": "test"}) == True
    assert gen.validate_output({"text": 123}) == False
