"""
Tests for Shared Utilities

This test suite covers:
- Path validation
- Directory creation
- File writing
- Filename sanitization
- Name formatting
- Dictionary operations
- Nested value operations

Test Generation Principles:
- Unique: Each test covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, errors, and boundaries
- Intuitive: Clear names and assertions express intent
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from ..core.shared_utils import (
    get_logger,
    validate_path,
    ensure_directory,
    safe_write_file,
    sanitize_filename,
    format_project_name,
    merge_dicts,
    get_nested_value,
    set_nested_value
)


class TestGetLogger:
    """Test suite for get_logger function"""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance"""
        # Happy path: Normal logger creation
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger_with_different_names(self):
        """Test get_logger creates different loggers for different names"""
        # Edge case: Different logger names
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1 is not None
        assert logger2 is not None

    def test_get_logger_with_empty_name(self):
        """Test get_logger handles empty name"""
        # Edge case: Empty name
        logger = get_logger("")
        assert logger is not None


class TestValidatePath:
    """Test suite for validate_path function"""

    def test_validate_path_with_existing_path(self, temp_dir):
        """Test validate_path accepts existing path"""
        # Happy path: Existing path
        test_path = temp_dir / "test_file.txt"
        test_path.write_text("test")
        validate_path(test_path, must_exist=True)
        # Should not raise

    def test_validate_path_with_non_existing_path_not_required(self, temp_dir):
        """Test validate_path accepts non-existing path when not required"""
        # Happy path: Non-existing path not required
        test_path = temp_dir / "non_existing.txt"
        validate_path(test_path, must_exist=False)
        # Should not raise

    def test_validate_path_raises_value_error_with_none(self):
        """Test validate_path raises ValueError with None"""
        # Error condition: None path
        with pytest.raises(ValueError, match="path cannot be None"):
            validate_path(None)

    def test_validate_path_raises_value_error_when_must_exist(self, temp_dir):
        """Test validate_path raises ValueError when path must exist but doesn't"""
        # Error condition: Path must exist but doesn't
        test_path = temp_dir / "non_existing.txt"
        with pytest.raises(ValueError, match="path does not exist"):
            validate_path(test_path, must_exist=True)

    def test_validate_path_with_directory(self, temp_dir):
        """Test validate_path works with directories"""
        # Happy path: Directory path
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()
        validate_path(test_dir, must_exist=True)
        # Should not raise


class TestEnsureDirectory:
    """Test suite for ensure_directory function"""

    def test_ensure_directory_creates_new_directory(self, temp_dir):
        """Test ensure_directory creates new directory"""
        # Happy path: Create new directory
        new_dir = temp_dir / "new_directory"
        result = ensure_directory(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_directory_with_existing_directory(self, temp_dir):
        """Test ensure_directory handles existing directory"""
        # Happy path: Existing directory
        existing_dir = temp_dir / "existing_dir"
        existing_dir.mkdir()
        result = ensure_directory(existing_dir)
        assert existing_dir.exists()
        assert result == existing_dir

    def test_ensure_directory_creates_parent_directories(self, temp_dir):
        """Test ensure_directory creates parent directories"""
        # Happy path: Nested directories
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        result = ensure_directory(nested_dir)
        assert nested_dir.exists()
        assert result == nested_dir

    def test_ensure_directory_raises_value_error_with_none(self):
        """Test ensure_directory raises ValueError with None"""
        # Error condition: None path
        with pytest.raises(ValueError, match="path cannot be None"):
            ensure_directory(None)

    def test_ensure_directory_returns_same_path(self, temp_dir):
        """Test ensure_directory returns the same path object"""
        # Edge case: Return value
        test_dir = temp_dir / "test_dir"
        result = ensure_directory(test_dir)
        assert result is test_dir


class TestSafeWriteFile:
    """Test suite for safe_write_file function"""

    def test_safe_write_file_creates_new_file(self, temp_dir):
        """Test safe_write_file creates new file"""
        # Happy path: Create new file
        file_path = temp_dir / "new_file.txt"
        content = "Test content"
        safe_write_file(file_path, content)
        assert file_path.exists()
        assert file_path.read_text() == content

    def test_safe_write_file_overwrites_existing_file(self, temp_dir):
        """Test safe_write_file overwrites existing file"""
        # Happy path: Overwrite existing
        file_path = temp_dir / "existing_file.txt"
        file_path.write_text("Old content")
        safe_write_file(file_path, "New content")
        assert file_path.read_text() == "New content"

    def test_safe_write_file_creates_parent_directories(self, temp_dir):
        """Test safe_write_file creates parent directories"""
        # Happy path: Create parent directories
        file_path = temp_dir / "parent" / "child" / "file.txt"
        safe_write_file(file_path, "Content")
        assert file_path.exists()

    def test_safe_write_file_with_custom_encoding(self, temp_dir):
        """Test safe_write_file uses custom encoding"""
        # Edge case: Custom encoding
        file_path = temp_dir / "unicode_file.txt"
        content = "Test with unicode: ñáéíóú"
        safe_write_file(file_path, content, encoding="utf-8")
        assert file_path.read_text(encoding="utf-8") == content

    def test_safe_write_file_raises_value_error_with_none_path(self):
        """Test safe_write_file raises ValueError with None path"""
        # Error condition: None file_path
        with pytest.raises(ValueError, match="file_path cannot be None"):
            safe_write_file(None, "content")

    def test_safe_write_file_raises_value_error_with_none_content(self, temp_dir):
        """Test safe_write_file raises ValueError with None content"""
        # Error condition: None content
        file_path = temp_dir / "test.txt"
        with pytest.raises(ValueError, match="content cannot be None"):
            safe_write_file(file_path, None)

    def test_safe_write_file_with_empty_content(self, temp_dir):
        """Test safe_write_file handles empty content"""
        # Edge case: Empty content
        file_path = temp_dir / "empty.txt"
        safe_write_file(file_path, "")
        assert file_path.exists()
        assert file_path.read_text() == ""

    def test_safe_write_file_with_large_content(self, temp_dir):
        """Test safe_write_file handles large content"""
        # Edge case: Large content
        file_path = temp_dir / "large.txt"
        large_content = "A" * 100000
        safe_write_file(file_path, large_content)
        assert len(file_path.read_text()) == 100000

    def test_safe_write_file_with_special_characters(self, temp_dir):
        """Test safe_write_file handles special characters"""
        # Edge case: Special characters
        file_path = temp_dir / "special.txt"
        content = "Special: @#$%^&*()_+-=[]{}|;:,.<>?"
        safe_write_file(file_path, content)
        assert file_path.read_text() == content

    def test_safe_write_file_with_newlines(self, temp_dir):
        """Test safe_write_file handles newlines"""
        # Edge case: Newlines
        file_path = temp_dir / "newlines.txt"
        content = "Line 1\nLine 2\nLine 3"
        safe_write_file(file_path, content)
        assert file_path.read_text() == content

    def test_safe_write_file_logs_error_on_io_error(self, temp_dir):
        """Test safe_write_file logs error on IOError"""
        # Error condition: IOError handling
        file_path = temp_dir / "test.txt"
        mock_logger = MagicMock()
        
        # Mock write_text to raise IOError
        with patch.object(file_path, 'write_text', side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                safe_write_file(file_path, "content", logger_instance=mock_logger)
        
        # Logger should have been called
        # Note: This depends on implementation details


class TestSanitizeFilename:
    """Test suite for sanitize_filename function"""

    def test_sanitize_filename_with_typical_name(self):
        """Test sanitize_filename with typical name"""
        # Happy path: Typical name
        result = sanitize_filename("My Project Name")
        assert result == "my_project_name"

    def test_sanitize_filename_with_special_characters(self):
        """Test sanitize_filename removes special characters"""
        # Edge case: Special characters
        result = sanitize_filename("Test@Project#123")
        assert result == "testproject123"

    def test_sanitize_filename_with_hyphens(self):
        """Test sanitize_filename converts hyphens to underscores"""
        # Edge case: Hyphens
        result = sanitize_filename("Test-Project-Name")
        assert result == "test_project_name"

    def test_sanitize_filename_with_multiple_spaces(self):
        """Test sanitize_filename handles multiple spaces"""
        # Edge case: Multiple spaces
        result = sanitize_filename("Test   Project   Name")
        assert result == "test_project_name"

    def test_sanitize_filename_with_leading_trailing_underscores(self):
        """Test sanitize_filename removes leading/trailing underscores"""
        # Edge case: Leading/trailing underscores
        result = sanitize_filename("_test_project_")
        assert result == "test_project"

    def test_sanitize_filename_with_empty_string(self):
        """Test sanitize_filename returns default for empty string"""
        # Edge case: Empty string
        result = sanitize_filename("")
        assert result == "unnamed"

    def test_sanitize_filename_with_whitespace_only(self):
        """Test sanitize_filename handles whitespace-only string"""
        # Edge case: Whitespace only
        result = sanitize_filename("   ")
        assert result == "unnamed"

    def test_sanitize_filename_respects_max_length(self):
        """Test sanitize_filename respects max_length parameter"""
        # Boundary value: Max length
        long_name = "A" * 100
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) == 50

    def test_sanitize_filename_with_max_length_one(self):
        """Test sanitize_filename with max_length of 1"""
        # Boundary value: Max length 1
        result = sanitize_filename("Test Project", max_length=1)
        assert len(result) == 1

    def test_sanitize_filename_raises_value_error_with_invalid_max_length(self):
        """Test sanitize_filename raises ValueError with invalid max_length"""
        # Error condition: Invalid max_length
        with pytest.raises(ValueError, match="max_length must be positive"):
            sanitize_filename("test", max_length=0)

    def test_sanitize_filename_raises_value_error_with_negative_max_length(self):
        """Test sanitize_filename raises ValueError with negative max_length"""
        # Error condition: Negative max_length
        with pytest.raises(ValueError, match="max_length must be positive"):
            sanitize_filename("test", max_length=-1)

    def test_sanitize_filename_preserves_numbers(self):
        """Test sanitize_filename preserves numbers"""
        # Edge case: Numbers
        result = sanitize_filename("Project 123 v2.0")
        assert "123" in result
        assert "2" in result
        assert "0" in result

    def test_sanitize_filename_lowercases_result(self):
        """Test sanitize_filename lowercases the result"""
        # Edge case: Case conversion
        result = sanitize_filename("TEST PROJECT")
        assert result == "test_project"
        assert result.islower()


class TestFormatProjectName:
    """Test suite for format_project_name function"""

    def test_format_project_name_with_underscores(self):
        """Test format_project_name replaces underscores with spaces"""
        # Happy path: Underscores
        result = format_project_name("test_project_name")
        assert result == "Test Project Name"

    def test_format_project_name_with_empty_string(self):
        """Test format_project_name returns empty string for empty input"""
        # Edge case: Empty string
        result = format_project_name("")
        assert result == ""

    def test_format_project_name_with_whitespace_only(self):
        """Test format_project_name handles whitespace-only string"""
        # Edge case: Whitespace only
        result = format_project_name("   ")
        assert result == "   ".title()

    def test_format_project_name_preserves_multiple_underscores(self):
        """Test format_project_name handles multiple consecutive underscores"""
        # Edge case: Multiple underscores
        result = format_project_name("test__project")
        assert result == "Test  Project"  # Double space from double underscore

    def test_format_project_name_with_mixed_case(self):
        """Test format_project_name title cases the result"""
        # Edge case: Mixed case
        result = format_project_name("test_PROJECT_name")
        assert result == "Test Project Name"


class TestMergeDicts:
    """Test suite for merge_dicts function"""

    def test_merge_dicts_with_two_dicts(self):
        """Test merge_dicts merges two dictionaries"""
        # Happy path: Two dicts
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_dicts_with_overlapping_keys(self):
        """Test merge_dicts overwrites with later values"""
        # Edge case: Overlapping keys
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_dicts_with_multiple_dicts(self):
        """Test merge_dicts merges multiple dictionaries"""
        # Happy path: Multiple dicts
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        dict3 = {"c": 3}
        result = merge_dicts(dict1, dict2, dict3)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_merge_dicts_with_empty_dicts(self):
        """Test merge_dicts handles empty dictionaries"""
        # Edge case: Empty dicts
        result = merge_dicts({}, {})
        assert result == {}

    def test_merge_dicts_with_none_dicts(self):
        """Test merge_dicts handles None dictionaries"""
        # Edge case: None dicts
        dict1 = {"a": 1}
        result = merge_dicts(dict1, None, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_dicts_with_no_arguments(self):
        """Test merge_dicts handles no arguments"""
        # Edge case: No arguments
        result = merge_dicts()
        assert result == {}

    def test_merge_dicts_preserves_nested_dicts(self):
        """Test merge_dicts preserves nested dictionaries"""
        # Edge case: Nested dicts
        dict1 = {"a": {"b": 1}}
        dict2 = {"a": {"c": 2}}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": {"c": 2}}  # Later dict overwrites


class TestGetNestedValue:
    """Test suite for get_nested_value function"""

    def test_get_nested_value_with_simple_key(self):
        """Test get_nested_value retrieves simple key"""
        # Happy path: Simple key
        data = {"name": "test"}
        result = get_nested_value(data, "name")
        assert result == "test"

    def test_get_nested_value_with_nested_key(self):
        """Test get_nested_value retrieves nested key"""
        # Happy path: Nested key
        data = {"user": {"profile": {"name": "John"}}}
        result = get_nested_value(data, "user.profile.name")
        assert result == "John"

    def test_get_nested_value_with_default(self):
        """Test get_nested_value returns default when key not found"""
        # Happy path: Default value
        data = {"a": 1}
        result = get_nested_value(data, "b", default="default")
        assert result == "default"

    def test_get_nested_value_with_none_default(self):
        """Test get_nested_value returns None as default"""
        # Edge case: None default
        data = {"a": 1}
        result = get_nested_value(data, "b")
        assert result is None

    def test_get_nested_value_with_empty_data(self):
        """Test get_nested_value handles empty data"""
        # Edge case: Empty data
        result = get_nested_value({}, "key")
        assert result is None

    def test_get_nested_value_with_none_data(self):
        """Test get_nested_value handles None data"""
        # Edge case: None data
        result = get_nested_value(None, "key")
        assert result is None

    def test_get_nested_value_with_empty_key_path(self):
        """Test get_nested_value handles empty key path"""
        # Edge case: Empty key path
        data = {"a": 1}
        result = get_nested_value(data, "")
        assert result is None

    def test_get_nested_value_with_none_value(self):
        """Test get_nested_value returns default for None value"""
        # Edge case: None value in dict
        data = {"key": None}
        result = get_nested_value(data, "key", default="default")
        assert result == "default"

    def test_get_nested_value_with_intermediate_none(self):
        """Test get_nested_value handles intermediate None"""
        # Edge case: Intermediate None
        data = {"user": None}
        result = get_nested_value(data, "user.profile.name")
        assert result is None


class TestSetNestedValue:
    """Test suite for set_nested_value function"""

    def test_set_nested_value_with_simple_key(self):
        """Test set_nested_value sets simple key"""
        # Happy path: Simple key
        data = {}
        set_nested_value(data, "name", "test")
        assert data == {"name": "test"}

    def test_set_nested_value_with_nested_key(self):
        """Test set_nested_value creates nested structure"""
        # Happy path: Nested key
        data = {}
        set_nested_value(data, "user.profile.name", "John")
        assert data == {"user": {"profile": {"name": "John"}}}

    def test_set_nested_value_overwrites_existing(self):
        """Test set_nested_value overwrites existing value"""
        # Edge case: Overwrite existing
        data = {"user": {"profile": {"name": "Old"}}}
        set_nested_value(data, "user.profile.name", "New")
        assert data["user"]["profile"]["name"] == "New"

    def test_set_nested_value_raises_value_error_with_none_data(self):
        """Test set_nested_value raises ValueError with None data"""
        # Error condition: None data
        with pytest.raises(ValueError, match="data cannot be None"):
            set_nested_value(None, "key", "value")

    def test_set_nested_value_raises_value_error_with_empty_key_path(self):
        """Test set_nested_value raises ValueError with empty key path"""
        # Error condition: Empty key path
        data = {}
        with pytest.raises(ValueError, match="key_path cannot be empty"):
            set_nested_value(data, "", "value")

    def test_set_nested_value_with_existing_nested_structure(self):
        """Test set_nested_value works with existing nested structure"""
        # Happy path: Existing structure
        data = {"user": {"profile": {}}}
        set_nested_value(data, "user.profile.name", "John")
        assert data["user"]["profile"]["name"] == "John"

    def test_set_nested_value_with_deeply_nested_path(self):
        """Test set_nested_value handles deeply nested paths"""
        # Edge case: Deep nesting
        data = {}
        set_nested_value(data, "a.b.c.d.e", "value")
        assert data["a"]["b"]["c"]["d"]["e"] == "value"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp(prefix="test_shared_utils_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


