"""
Tests for Rust Extensions
=========================
"""

import pytest
from robot_movement_ai.native.wrapper import (
    RUST_AVAILABLE,
    json_parse,
    string_search,
    hash_data
)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
class TestRustExtensions:
    """Tests for Rust extensions."""
    
    def test_json_parse(self):
        """Test JSON parsing."""
        json_str = '{"key": "value", "number": 42}'
        result = json_parse(json_str)
        
        assert result["key"] == "value"
        assert result["number"] == 42
    
    def test_string_search(self):
        """Test string search."""
        text = "hello world hello"
        positions = string_search(text, "hello")
        
        assert positions == [0, 12]
    
    def test_hash_data(self):
        """Test hashing."""
        data = "test data"
        hash_value = hash_data(data)
        
        assert isinstance(hash_value, int)
        assert hash_value != 0

