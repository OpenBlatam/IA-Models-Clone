"""
Improved Unit Tests for Core Helpers - Enhanced Coverage

Additional edge cases and improved test scenarios
"""

import pytest
import json
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from core.helpers import (
    generate_id,
    hash_string,
    safe_json_loads,
    safe_json_dumps,
    format_duration,
    format_file_size,
    ensure_directory,
    chunk_list,
    merge_dicts,
    get_nested_value,
    set_nested_value,
    sanitize_filename,
    retry_on_failure
)


class TestGenerateIdImproved:
    """Improved test cases for generate_id function"""
    
    def test_generate_id_uniqueness(self):
        """Test that generated IDs are unique"""
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100  # All should be unique
    
    def test_generate_id_with_special_prefix(self):
        """Test ID generation with special characters in prefix"""
        prefix = "test_123"
        id_str = generate_id(prefix=prefix)
        assert id_str.startswith(prefix)
    
    def test_generate_id_with_empty_prefix(self):
        """Test ID generation with empty prefix"""
        id_str = generate_id(prefix="")
        # Should still generate valid UUID
        uuid.UUID(id_str)


class TestHashStringImproved:
    """Improved test cases for hash_string function"""
    
    def test_hash_string_unicode(self):
        """Test hashing unicode strings"""
        text = "Hello 世界 🌍"
        result = hash_string(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_hash_string_very_long(self):
        """Test hashing very long strings"""
        text = "x" * 10000
        result = hash_string(text)
        assert isinstance(result, str)
    
    def test_hash_string_special_characters(self):
        """Test hashing strings with special characters"""
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = hash_string(text)
        assert isinstance(result, str)
    
    def test_hash_string_consistency(self):
        """Test hash consistency for same input"""
        text = "test string"
        hash1 = hash_string(text)
        hash2 = hash_string(text)
        assert hash1 == hash2


class TestSafeJsonLoadsImproved:
    """Improved test cases for safe_json_loads function"""
    
    def test_safe_json_loads_invalid_json_with_default(self):
        """Test loading invalid JSON with default value"""
        result = safe_json_loads("invalid json", default={"error": True})
        assert result == {"error": True}
    
    def test_safe_json_loads_empty_string(self):
        """Test loading empty string"""
        result = safe_json_loads("", default=None)
        assert result is None
    
    def test_safe_json_loads_none_input(self):
        """Test loading None input"""
        result = safe_json_loads(None, default={})
        assert result == {}


class TestChunkListImproved:
    """Improved test cases for chunk_list function"""
    
    def test_chunk_list_empty_list(self):
        """Test chunking empty list"""
        result = list(chunk_list([], 5))
        assert result == []
    
    def test_chunk_list_size_one(self):
        """Test chunking with size 1"""
        items = [1, 2, 3, 4, 5]
        result = list(chunk_list(items, 1))
        assert len(result) == 5
        assert all(len(chunk) == 1 for chunk in result)
    
    def test_chunk_list_size_larger_than_list(self):
        """Test chunking with size larger than list"""
        items = [1, 2, 3]
        result = list(chunk_list(items, 10))
        assert len(result) == 1
        assert result[0] == items
    
    def test_chunk_list_exact_multiple(self):
        """Test chunking when list size is exact multiple"""
        items = [1, 2, 3, 4, 5, 6]
        result = list(chunk_list(items, 3))
        assert len(result) == 2
        assert all(len(chunk) == 3 for chunk in result)


class TestMergeDictsImproved:
    """Improved test cases for merge_dicts function"""
    
    def test_merge_dicts_empty_dicts(self):
        """Test merging empty dictionaries"""
        result = merge_dicts({}, {})
        assert result == {}
    
    def test_merge_dicts_nested_override(self):
        """Test nested dictionary override"""
        dict1 = {"a": {"b": 1, "c": 2}}
        dict2 = {"a": {"b": 3}}
        result = merge_dicts(dict1, dict2)
        assert result["a"]["b"] == 3
        assert result["a"]["c"] == 2  # Should preserve
    
    def test_merge_dicts_multiple_dicts(self):
        """Test merging multiple dictionaries"""
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        dict3 = {"c": 3}
        result = merge_dicts(dict1, dict2, dict3)
        assert result == {"a": 1, "b": 2, "c": 3}


class TestGetNestedValueImproved:
    """Improved test cases for get_nested_value function"""
    
    def test_get_nested_value_deep_nesting(self):
        """Test getting value from deep nesting"""
        data = {"a": {"b": {"c": {"d": "value"}}}}
        result = get_nested_value(data, "a.b.c.d")
        assert result == "value"
    
    def test_get_nested_value_with_list_index(self):
        """Test getting value with list index"""
        data = {"items": [{"name": "item1"}, {"name": "item2"}]}
        result = get_nested_value(data, "items.0.name")
        assert result == "item1"
    
    def test_get_nested_value_invalid_path(self):
        """Test getting value with invalid path"""
        data = {"a": {"b": 1}}
        result = get_nested_value(data, "a.b.c", default=None)
        assert result is None


class TestSanitizeFilenameImproved:
    """Improved test cases for sanitize_filename function"""
    
    def test_sanitize_filename_unicode(self):
        """Test sanitizing filename with unicode"""
        filename = "test_文件_🎵.mp3"
        result = sanitize_filename(filename)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_sanitize_filename_very_long(self):
        """Test sanitizing very long filename"""
        filename = "x" * 300 + ".mp3"
        result = sanitize_filename(filename)
        assert len(result) <= 255  # Common filesystem limit
    
    def test_sanitize_filename_only_special_chars(self):
        """Test sanitizing filename with only special characters"""
        filename = "!!!@@@###"
        result = sanitize_filename(filename)
        assert len(result) > 0


class TestRetryOnFailureImproved:
    """Improved test cases for retry_on_failure decorator"""
    
    def test_retry_on_failure_succeeds_after_retries(self):
        """Test retry succeeds after some failures"""
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_failure_exhausts_retries(self):
        """Test retry exhausts all retries"""
        @retry_on_failure(max_retries=2, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()















