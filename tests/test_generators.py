import pytest
from synth_data_pipeline.generators.base import BaseGenerator

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
