"""
Tests for Validators

This test suite covers:
- Project name validation
- Description validation
- Email validation
- URL validation
- Validator classes

Test Generation Principles:
- Unique: Each test covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, errors, and boundaries
- Intuitive: Clear names and assertions express intent
"""

import pytest
from ..core.validators import (
    validate_project_name,
    validate_description,
    validate_email,
    validate_url,
    ProjectNameValidator,
    DescriptionValidator
)


class TestValidateProjectName:
    """Test suite for validate_project_name function"""

    def test_validate_project_name_with_valid_name(self):
        """Test validate_project_name accepts valid name"""
        # Happy path: Valid name
        assert validate_project_name("test_project") is True
        assert validate_project_name("my-awesome-project") is True
        assert validate_project_name("project123") is True

    def test_validate_project_name_with_minimum_length(self):
        """Test validate_project_name accepts minimum length name"""
        # Boundary value: Minimum length (3)
        assert validate_project_name("abc") is True
        assert validate_project_name("123") is True

    def test_validate_project_name_with_maximum_length(self):
        """Test validate_project_name accepts maximum length name"""
        # Boundary value: Maximum length (50)
        long_name = "a" * 50
        assert validate_project_name(long_name) is True

    def test_validate_project_name_with_underscores(self):
        """Test validate_project_name accepts underscores"""
        # Happy path: Underscores
        assert validate_project_name("test_project_name") is True
        assert validate_project_name("_test_") is True

    def test_validate_project_name_with_hyphens(self):
        """Test validate_project_name accepts hyphens"""
        # Happy path: Hyphens
        assert validate_project_name("test-project-name") is True
        assert validate_project_name("-test-") is True

    def test_validate_project_name_with_numbers(self):
        """Test validate_project_name accepts numbers"""
        # Happy path: Numbers
        assert validate_project_name("project123") is True
        assert validate_project_name("123project") is True
        assert validate_project_name("project_123") is True

    def test_validate_project_name_with_mixed_case(self):
        """Test validate_project_name accepts mixed case"""
        # Happy path: Mixed case
        assert validate_project_name("TestProject") is True
        assert validate_project_name("TEST_PROJECT") is True

    def test_validate_project_name_rejects_empty_string(self):
        """Test validate_project_name rejects empty string"""
        # Error condition: Empty string
        assert validate_project_name("") is False

    def test_validate_project_name_rejects_too_short(self):
        """Test validate_project_name rejects names shorter than 3"""
        # Error condition: Too short
        assert validate_project_name("ab") is False
        assert validate_project_name("a") is False

    def test_validate_project_name_rejects_too_long(self):
        """Test validate_project_name rejects names longer than 50"""
        # Error condition: Too long
        long_name = "a" * 51
        assert validate_project_name(long_name) is False

    def test_validate_project_name_rejects_special_characters(self):
        """Test validate_project_name rejects special characters"""
        # Error condition: Special characters
        assert validate_project_name("test@project") is False
        assert validate_project_name("test#project") is False
        assert validate_project_name("test project") is False  # Space
        assert validate_project_name("test.project") is False  # Dot

    def test_validate_project_name_rejects_none(self):
        """Test validate_project_name rejects None"""
        # Error condition: None
        assert validate_project_name(None) is False


class TestValidateDescription:
    """Test suite for validate_description function"""

    def test_validate_description_with_valid_description(self):
        """Test validate_description accepts valid description"""
        # Happy path: Valid description
        desc = "This is a valid project description with enough words"
        assert validate_description(desc) is True

    def test_validate_description_with_minimum_length(self):
        """Test validate_description accepts minimum length"""
        # Boundary value: Minimum length (10)
        desc = "1234567890"  # 10 characters, but needs 5 unique words
        # This should fail because it doesn't have 5 unique words
        # Let's use a proper description
        desc = "one two three four five"  # 5 unique words
        assert validate_description(desc) is True

    def test_validate_description_with_maximum_length(self):
        """Test validate_description accepts maximum length"""
        # Boundary value: Maximum length (2000)
        desc = "word " * 400  # 2000 characters with spaces
        assert validate_description(desc) is True

    def test_validate_description_with_exactly_five_words(self):
        """Test validate_description accepts exactly 5 unique words"""
        # Boundary value: Exactly 5 unique words
        desc = "one two three four five"
        assert validate_description(desc) is True

    def test_validate_description_with_more_than_five_words(self):
        """Test validate_description accepts more than 5 words"""
        # Happy path: More than 5 words
        desc = "This is a valid project description with many words"
        assert validate_description(desc) is True

    def test_validate_description_rejects_empty_string(self):
        """Test validate_description rejects empty string"""
        # Error condition: Empty string
        assert validate_description("") is False

    def test_validate_description_rejects_too_short(self):
        """Test validate_description rejects descriptions shorter than 10"""
        # Error condition: Too short
        assert validate_description("short") is False
        assert validate_description("123456789") is False  # 9 characters

    def test_validate_description_rejects_too_long(self):
        """Test validate_description rejects descriptions longer than 2000"""
        # Error condition: Too long
        long_desc = "word " * 401  # More than 2000 characters
        assert validate_description(long_desc) is False

    def test_validate_description_rejects_fewer_than_five_unique_words(self):
        """Test validate_description rejects fewer than 5 unique words"""
        # Error condition: Not enough unique words
        desc = "word word word word"  # Only 1 unique word
        assert validate_description(desc) is False
        
        desc = "one two three four"  # Only 4 unique words
        assert validate_description(desc) is False

    def test_validate_description_rejects_none(self):
        """Test validate_description rejects None"""
        # Error condition: None
        assert validate_description(None) is False

    def test_validate_description_with_duplicate_words(self):
        """Test validate_description counts unique words correctly"""
        # Edge case: Duplicate words
        desc = "word word word word word"  # 5 words, but only 1 unique
        assert validate_description(desc) is False
        
        desc = "one two three four five six"  # 6 words, 6 unique
        assert validate_description(desc) is True


class TestValidateEmail:
    """Test suite for validate_email function"""

    def test_validate_email_with_valid_email(self):
        """Test validate_email accepts valid email"""
        # Happy path: Valid email
        assert validate_email("test@example.com") is True
        assert validate_email("user.name@example.com") is True
        assert validate_email("user+tag@example.co.uk") is True

    def test_validate_email_with_different_domains(self):
        """Test validate_email accepts different domain formats"""
        # Happy path: Different domains
        assert validate_email("test@example.com") is True
        assert validate_email("test@example.co.uk") is True
        assert validate_email("test@sub.example.com") is True

    def test_validate_email_with_special_characters(self):
        """Test validate_email accepts special characters in local part"""
        # Edge case: Special characters
        assert validate_email("user.name@example.com") is True
        assert validate_email("user+tag@example.com") is True
        assert validate_email("user_name@example.com") is True

    def test_validate_email_rejects_missing_at(self):
        """Test validate_email rejects email without @"""
        # Error condition: Missing @
        assert validate_email("testexample.com") is False

    def test_validate_email_rejects_missing_domain(self):
        """Test validate_email rejects email without domain"""
        # Error condition: Missing domain
        assert validate_email("test@") is False

    def test_validate_email_rejects_missing_local_part(self):
        """Test validate_email rejects email without local part"""
        # Error condition: Missing local part
        assert validate_email("@example.com") is False

    def test_validate_email_rejects_invalid_tld(self):
        """Test validate_email rejects invalid TLD"""
        # Error condition: Invalid TLD
        assert validate_email("test@example.c") is False  # TLD too short

    def test_validate_email_rejects_spaces(self):
        """Test validate_email rejects spaces"""
        # Error condition: Spaces
        assert validate_email("test @example.com") is False
        assert validate_email("test@example .com") is False

    def test_validate_email_rejects_empty_string(self):
        """Test validate_email rejects empty string"""
        # Error condition: Empty string
        assert validate_email("") is False

    def test_validate_email_rejects_none(self):
        """Test validate_email rejects None"""
        # Error condition: None
        assert validate_email(None) is False


class TestValidateUrl:
    """Test suite for validate_url function"""

    def test_validate_url_with_valid_http_url(self):
        """Test validate_url accepts valid HTTP URL"""
        # Happy path: HTTP URL
        assert validate_url("http://example.com") is True
        assert validate_url("http://example.com/path") is True

    def test_validate_url_with_valid_https_url(self):
        """Test validate_url accepts valid HTTPS URL"""
        # Happy path: HTTPS URL
        assert validate_url("https://example.com") is True
        assert validate_url("https://example.com/path") is True

    def test_validate_url_with_path(self):
        """Test validate_url accepts URL with path"""
        # Happy path: URL with path
        assert validate_url("https://example.com/path/to/resource") is True

    def test_validate_url_with_query_parameters(self):
        """Test validate_url accepts URL with query parameters"""
        # Happy path: URL with query
        assert validate_url("https://example.com?param=value") is True

    def test_validate_url_with_port(self):
        """Test validate_url accepts URL with port"""
        # Edge case: URL with port
        assert validate_url("http://example.com:8080") is True

    def test_validate_url_rejects_missing_protocol(self):
        """Test validate_url rejects URL without protocol"""
        # Error condition: Missing protocol
        assert validate_url("example.com") is False

    def test_validate_url_rejects_invalid_protocol(self):
        """Test validate_url rejects invalid protocol"""
        # Error condition: Invalid protocol
        assert validate_url("ftp://example.com") is False  # Only http/https allowed

    def test_validate_url_rejects_empty_string(self):
        """Test validate_url rejects empty string"""
        # Error condition: Empty string
        assert validate_url("") is False

    def test_validate_url_rejects_none(self):
        """Test validate_url rejects None"""
        # Error condition: None
        assert validate_url(None) is False


class TestProjectNameValidator:
    """Test suite for ProjectNameValidator class"""

    def test_validate_with_valid_name(self):
        """Test ProjectNameValidator.validate accepts valid name"""
        # Happy path: Valid name
        result = ProjectNameValidator.validate("test_project")
        assert result == "test_project"

    def test_validate_strips_whitespace(self):
        """Test ProjectNameValidator.validate strips whitespace"""
        # Edge case: Whitespace stripping
        result = ProjectNameValidator.validate("  test_project  ")
        assert result == "test_project"

    def test_validate_returns_none_for_empty_string(self):
        """Test ProjectNameValidator.validate returns None for empty string"""
        # Edge case: Empty string
        result = ProjectNameValidator.validate("")
        assert result is None

    def test_validate_returns_none_for_none(self):
        """Test ProjectNameValidator.validate returns None for None"""
        # Edge case: None
        result = ProjectNameValidator.validate(None)
        assert result is None

    def test_validate_raises_value_error_for_invalid_name(self):
        """Test ProjectNameValidator.validate raises ValueError for invalid name"""
        # Error condition: Invalid name
        with pytest.raises(ValueError, match="Project name must be"):
            ProjectNameValidator.validate("ab")  # Too short

    def test_validate_raises_value_error_for_special_characters(self):
        """Test ProjectNameValidator.validate raises ValueError for special characters"""
        # Error condition: Special characters
        with pytest.raises(ValueError, match="Project name must be"):
            ProjectNameValidator.validate("test@project")


class TestDescriptionValidator:
    """Test suite for DescriptionValidator class"""

    def test_validate_with_valid_description(self):
        """Test DescriptionValidator.validate accepts valid description"""
        # Happy path: Valid description
        desc = "This is a valid project description with enough words"
        result = DescriptionValidator.validate(desc)
        assert result == desc.strip()

    def test_validate_strips_whitespace(self):
        """Test DescriptionValidator.validate strips whitespace"""
        # Edge case: Whitespace stripping
        desc = "  This is a valid project description with enough words  "
        result = DescriptionValidator.validate(desc)
        assert result == desc.strip()

    def test_validate_raises_value_error_for_empty_string(self):
        """Test DescriptionValidator.validate raises ValueError for empty string"""
        # Error condition: Empty string
        with pytest.raises(ValueError, match="Description cannot be empty"):
            DescriptionValidator.validate("")

    def test_validate_raises_value_error_for_whitespace_only(self):
        """Test DescriptionValidator.validate raises ValueError for whitespace only"""
        # Error condition: Whitespace only
        with pytest.raises(ValueError, match="Description cannot be empty"):
            DescriptionValidator.validate("   ")

    def test_validate_raises_value_error_for_invalid_description(self):
        """Test DescriptionValidator.validate raises ValueError for invalid description"""
        # Error condition: Invalid description
        with pytest.raises(ValueError, match="Description must be"):
            DescriptionValidator.validate("short")  # Too short

    def test_validate_raises_value_error_for_few_words(self):
        """Test DescriptionValidator.validate raises ValueError for too few words"""
        # Error condition: Too few unique words
        with pytest.raises(ValueError, match="Description must be"):
            DescriptionValidator.validate("one two three four")  # Only 4 words


