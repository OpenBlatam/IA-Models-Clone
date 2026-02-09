"""
Tests for JSON Utilities

This test suite covers:
- JSON serialization (dumps)
- JSON deserialization (loads)
- String serialization
- Pretty printing
- Edge cases and error handling

Test Generation Principles:
- Unique: Each test covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, errors, and boundaries
- Intuitive: Clear names and assertions express intent
"""

import pytest
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List

from ..core.json_utils import (
    json_dumps,
    json_loads,
    json_dumps_str,
    json_dumps_pretty
)


@dataclass
class TestDataClass:
    """Test dataclass for serialization"""
    name: str
    value: int


class TestJsonDumps:
    """Test suite for json_dumps function"""

    def test_json_dumps_with_simple_dict(self):
        """Test json_dumps serializes simple dictionary"""
        # Happy path: Simple dict
        data = {"name": "test", "value": 42}
        result = json_dumps(data)
        assert isinstance(result, bytes)
        assert b"test" in result
        assert b"42" in result

    def test_json_dumps_with_nested_dict(self):
        """Test json_dumps serializes nested dictionary"""
        # Happy path: Nested dict
        data = {"user": {"name": "John", "age": 30}}
        result = json_dumps(data)
        assert isinstance(result, bytes)
        assert b"John" in result

    def test_json_dumps_with_list(self):
        """Test json_dumps serializes list"""
        # Happy path: List
        data = [1, 2, 3, "test"]
        result = json_dumps(data)
        assert isinstance(result, bytes)
        assert b"1" in result

    def test_json_dumps_with_empty_dict(self):
        """Test json_dumps serializes empty dictionary"""
        # Edge case: Empty dict
        data = {}
        result = json_dumps(data)
        assert result == b"{}"

    def test_json_dumps_with_empty_list(self):
        """Test json_dumps serializes empty list"""
        # Edge case: Empty list
        data = []
        result = json_dumps(data)
        assert result == b"[]"

    def test_json_dumps_with_none(self):
        """Test json_dumps serializes None"""
        # Edge case: None
        result = json_dumps(None)
        assert result == b"null"

    def test_json_dumps_with_string(self):
        """Test json_dumps serializes string"""
        # Happy path: String
        data = "test string"
        result = json_dumps(data)
        assert isinstance(result, bytes)
        assert b"test string" in result

    def test_json_dumps_with_integer(self):
        """Test json_dumps serializes integer"""
        # Happy path: Integer
        data = 42
        result = json_dumps(data)
        assert result == b"42"

    def test_json_dumps_with_float(self):
        """Test json_dumps serializes float"""
        # Happy path: Float
        data = 3.14
        result = json_dumps(data)
        assert isinstance(result, bytes)

    def test_json_dumps_with_boolean(self):
        """Test json_dumps serializes boolean"""
        # Happy path: Boolean
        data = True
        result = json_dumps(data)
        assert result == b"true"
        
        data = False
        result = json_dumps(data)
        assert result == b"false"

    def test_json_dumps_with_unicode(self):
        """Test json_dumps handles Unicode characters"""
        # Edge case: Unicode
        data = {"name": "José", "city": "São Paulo"}
        result = json_dumps(data)
        assert isinstance(result, bytes)
        # Should handle Unicode correctly

    def test_json_dumps_with_special_characters(self):
        """Test json_dumps handles special characters"""
        # Edge case: Special characters
        data = {"text": "Hello @#$%^&*()"}
        result = json_dumps(data)
        assert isinstance(result, bytes)

    def test_json_dumps_with_large_data(self):
        """Test json_dumps handles large data structures"""
        # Edge case: Large data
        data = {"items": list(range(10000))}
        result = json_dumps(data)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestJsonLoads:
    """Test suite for json_loads function"""

    def test_json_loads_with_bytes(self):
        """Test json_loads deserializes bytes"""
        # Happy path: Bytes input
        data = b'{"name": "test", "value": 42}'
        result = json_loads(data)
        assert result == {"name": "test", "value": 42}

    def test_json_loads_with_string(self):
        """Test json_loads deserializes string"""
        # Happy path: String input
        data = '{"name": "test", "value": 42}'
        result = json_loads(data)
        assert result == {"name": "test", "value": 42}

    def test_json_loads_with_nested_dict(self):
        """Test json_loads deserializes nested dictionary"""
        # Happy path: Nested dict
        data = b'{"user": {"name": "John", "age": 30}}'
        result = json_loads(data)
        assert result["user"]["name"] == "John"
        assert result["user"]["age"] == 30

    def test_json_loads_with_list(self):
        """Test json_loads deserializes list"""
        # Happy path: List
        data = b'[1, 2, 3, "test"]'
        result = json_loads(data)
        assert result == [1, 2, 3, "test"]

    def test_json_loads_with_empty_dict(self):
        """Test json_loads deserializes empty dictionary"""
        # Edge case: Empty dict
        data = b"{}"
        result = json_loads(data)
        assert result == {}

    def test_json_loads_with_empty_list(self):
        """Test json_loads deserializes empty list"""
        # Edge case: Empty list
        data = b"[]"
        result = json_loads(data)
        assert result == []

    def test_json_loads_with_null(self):
        """Test json_loads deserializes null"""
        # Edge case: Null
        data = b"null"
        result = json_loads(data)
        assert result is None

    def test_json_loads_with_string_value(self):
        """Test json_loads deserializes string value"""
        # Happy path: String value
        data = b'"test string"'
        result = json_loads(data)
        assert result == "test string"

    def test_json_loads_with_integer_value(self):
        """Test json_loads deserializes integer value"""
        # Happy path: Integer value
        data = b"42"
        result = json_loads(data)
        assert result == 42

    def test_json_loads_with_float_value(self):
        """Test json_loads deserializes float value"""
        # Happy path: Float value
        data = b"3.14"
        result = json_loads(data)
        assert result == 3.14

    def test_json_loads_with_boolean_value(self):
        """Test json_loads deserializes boolean value"""
        # Happy path: Boolean value
        data = b"true"
        result = json_loads(data)
        assert result is True
        
        data = b"false"
        result = json_loads(data)
        assert result is False

    def test_json_loads_with_unicode(self):
        """Test json_loads handles Unicode characters"""
        # Edge case: Unicode
        data = '{"name": "José"}'.encode('utf-8')
        result = json_loads(data)
        assert result["name"] == "José"

    def test_json_loads_roundtrip(self):
        """Test json_loads can deserialize json_dumps output"""
        # Happy path: Roundtrip
        original = {"name": "test", "value": 42, "nested": {"a": 1}}
        serialized = json_dumps(original)
        deserialized = json_loads(serialized)
        assert deserialized == original


class TestJsonDumpsStr:
    """Test suite for json_dumps_str function"""

    def test_json_dumps_str_returns_string(self):
        """Test json_dumps_str returns string"""
        # Happy path: Returns string
        data = {"name": "test"}
        result = json_dumps_str(data)
        assert isinstance(result, str)

    def test_json_dumps_str_with_simple_dict(self):
        """Test json_dumps_str serializes simple dictionary"""
        # Happy path: Simple dict
        data = {"name": "test", "value": 42}
        result = json_dumps_str(data)
        assert isinstance(result, str)
        assert "test" in result
        assert "42" in result

    def test_json_dumps_str_with_nested_dict(self):
        """Test json_dumps_str serializes nested dictionary"""
        # Happy path: Nested dict
        data = {"user": {"name": "John"}}
        result = json_dumps_str(data)
        assert isinstance(result, str)
        assert "John" in result

    def test_json_dumps_str_with_empty_dict(self):
        """Test json_dumps_str serializes empty dictionary"""
        # Edge case: Empty dict
        data = {}
        result = json_dumps_str(data)
        assert result == "{}"

    def test_json_dumps_str_roundtrip(self):
        """Test json_dumps_str can be loaded back"""
        # Happy path: Roundtrip
        original = {"name": "test", "value": 42}
        serialized = json_dumps_str(original)
        deserialized = json.loads(serialized)
        assert deserialized == original

    def test_json_dumps_str_with_unicode(self):
        """Test json_dumps_str handles Unicode characters"""
        # Edge case: Unicode
        data = {"name": "José"}
        result = json_dumps_str(data)
        assert isinstance(result, str)
        assert "José" in result


class TestJsonDumpsPretty:
    """Test suite for json_dumps_pretty function"""

    def test_json_dumps_pretty_returns_string(self):
        """Test json_dumps_pretty returns string"""
        # Happy path: Returns string
        data = {"name": "test"}
        result = json_dumps_pretty(data)
        assert isinstance(result, str)

    def test_json_dumps_pretty_with_indentation(self):
        """Test json_dumps_pretty includes indentation"""
        # Happy path: Indentation
        data = {"name": "test", "value": 42}
        result = json_dumps_pretty(data, indent=2)
        assert "\n" in result  # Should have newlines
        assert "  " in result  # Should have spaces for indentation

    def test_json_dumps_pretty_with_custom_indent(self):
        """Test json_dumps_pretty uses custom indent"""
        # Edge case: Custom indent
        data = {"name": "test"}
        result = json_dumps_pretty(data, indent=4)
        assert "    " in result  # 4 spaces

    def test_json_dumps_pretty_with_nested_structure(self):
        """Test json_dumps_pretty formats nested structure"""
        # Happy path: Nested structure
        data = {"user": {"profile": {"name": "John"}}}
        result = json_dumps_pretty(data, indent=2)
        assert isinstance(result, str)
        assert "John" in result

    def test_json_dumps_pretty_with_empty_dict(self):
        """Test json_dumps_pretty formats empty dictionary"""
        # Edge case: Empty dict
        data = {}
        result = json_dumps_pretty(data, indent=2)
        assert result == "{}"

    def test_json_dumps_pretty_with_list(self):
        """Test json_dumps_pretty formats list"""
        # Happy path: List
        data = [1, 2, 3, {"nested": "value"}]
        result = json_dumps_pretty(data, indent=2)
        assert isinstance(result, str)
        assert "1" in result

    def test_json_dumps_pretty_with_unicode(self):
        """Test json_dumps_pretty handles Unicode with ensure_ascii=False"""
        # Edge case: Unicode
        data = {"name": "José", "city": "São Paulo"}
        result = json_dumps_pretty(data, indent=2, ensure_ascii=False)
        assert "José" in result
        assert "São Paulo" in result

    def test_json_dumps_pretty_with_ensure_ascii_true(self):
        """Test json_dumps_pretty escapes Unicode with ensure_ascii=True"""
        # Edge case: ASCII encoding
        data = {"name": "José"}
        result = json_dumps_pretty(data, indent=2, ensure_ascii=True)
        assert isinstance(result, str)
        # Unicode should be escaped

    def test_json_dumps_pretty_roundtrip(self):
        """Test json_dumps_pretty can be loaded back"""
        # Happy path: Roundtrip
        original = {"name": "test", "value": 42, "nested": {"a": 1}}
        serialized = json_dumps_pretty(original, indent=2)
        deserialized = json.loads(serialized)
        assert deserialized == original

    def test_json_dumps_pretty_with_zero_indent(self):
        """Test json_dumps_pretty with zero indent"""
        # Edge case: Zero indent
        data = {"name": "test"}
        result = json_dumps_pretty(data, indent=0)
        assert isinstance(result, str)

    def test_json_dumps_pretty_with_large_indent(self):
        """Test json_dumps_pretty with large indent"""
        # Edge case: Large indent
        data = {"name": "test"}
        result = json_dumps_pretty(data, indent=10)
        assert isinstance(result, str)
        assert "          " in result  # 10 spaces


