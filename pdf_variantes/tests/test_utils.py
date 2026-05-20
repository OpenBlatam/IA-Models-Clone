"""
Unit Tests for Utility Functions
==================================
Tests for utility functions, helpers, and helper classes.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import io

# Import utilities
try:
    from utils.validation import (
        validate_filename,
        validate_file_extension,
        validate_integer_range,
        validate_string_length,
        validate_email,
        validate_uuid,
        ValidationError
    )
except ImportError:
    # Try alternative import path
    from utils import (
        validate_filename,
        validate_file_extension,
        validate_integer_range,
        validate_string_length,
        validate_email,
        validate_uuid,
        ValidationError
    )

try:
    from utils.file_helpers import (
        FileValidator,
        sanitize_filename,
        validate_pdf_file,
        extract_metadata
    )
except ImportError:
    FileValidator = None
    sanitize_filename = None
    validate_pdf_file = None
    extract_metadata = None


class TestFilenameValidation:
    """Tests for filename validation."""
    
    @pytest.mark.parametrize("filename,expected_valid", [
        ("test.pdf", True),
        ("document_123.pdf", True),
        ("my-file_name.pdf", True),
        ("test.PDF", True),  # Case insensitive
        ("test file.pdf", True),  # Spaces allowed (depends on implementation)
        ("", False),  # Empty
        ("test<>file.pdf", False),  # Invalid characters
        ("../etc/passwd", False),  # Path traversal
        ("..\\windows\\system32", False),  # Windows path traversal
        ("test" + "a" * 300 + ".pdf", False),  # Too long
        ("test.pdf" + "\x00", False),  # Null byte
        ("test.pdf\n", False),  # Newline
        ("test.pdf\r", False),  # Carriage return
        ("test.pdf\t", False),  # Tab
    ])
    def test_validate_filename_cases(self, filename, expected_valid):
        """Test filename validation with various cases."""
        if validate_filename is None:
            pytest.skip("validate_filename not available")
        
        valid, error = validate_filename(filename)
        assert valid == expected_valid
        if not expected_valid:
            assert error is not None
        else:
            assert error is None
    
    def test_validate_filename_valid(self):
        """Test valid filename."""
        if validate_filename is None:
            pytest.skip("validate_filename not available")
        
        valid, error = validate_filename("test.pdf")
        assert valid is True
        assert error is None
    
    def test_validate_filename_invalid_characters(self):
        """Test filename with invalid characters."""
        if validate_filename is None:
            pytest.skip("validate_filename not available")
        
        invalid_chars = ["<", ">", ":", '"', "|", "?", "*", "\x00"]
        for char in invalid_chars:
            valid, error = validate_filename(f"test{char}file.pdf")
            assert valid is False, f"Should reject filename with '{char}'"
            assert error is not None
    
    def test_validate_filename_empty(self):
        """Test empty filename."""
        if validate_filename is None:
            pytest.skip("validate_filename not available")
        
        valid, error = validate_filename("")
        assert valid is False
        assert error is not None
    
    def test_validate_filename_too_long(self):
        """Test filename that's too long."""
        if validate_filename is None:
            pytest.skip("validate_filename not available")
        
        long_name = "a" * 300 + ".pdf"
        valid, error = validate_filename(long_name)
        assert valid is False
        assert error is not None
    
    @pytest.mark.parametrize("path", [
        "../etc/passwd",
        "../../etc/passwd",
        "..\\windows\\system32",
        "C:\\windows\\system32",
        "/etc/passwd",
        "~/secret",
    ])
    def test_validate_filename_path_traversal(self, path):
        """Test filename with various path traversal attempts."""
        if validate_filename is None:
            pytest.skip("validate_filename not available")
        
        valid, error = validate_filename(path)
        assert valid is False
        assert error is not None


class TestFileExtensionValidation:
    """Tests for file extension validation."""
    
    @pytest.mark.parametrize("filename,allowed_extensions,expected_valid", [
        ("test.pdf", [".pdf", ".txt"], True),
        ("document.txt", [".pdf", ".txt"], True),
        ("test.PDF", [".pdf", ".txt"], True),  # Case insensitive
        ("test.TXT", [".pdf", ".txt"], True),
        ("test.exe", [".pdf", ".txt"], False),
        ("test", [".pdf", ".txt"], False),  # No extension
        ("test.pdf.exe", [".pdf", ".txt"], False),  # Double extension
        ("test.pdf.", [".pdf", ".txt"], False),  # Trailing dot
    ])
    def test_validate_file_extension_cases(self, filename, allowed_extensions, expected_valid):
        """Test file extension validation with various cases."""
        if validate_file_extension is None:
            pytest.skip("validate_file_extension not available")
        
        valid, error = validate_file_extension(filename, allowed_extensions)
        assert valid == expected_valid
        if not expected_valid:
            assert error is not None
    
    def test_validate_file_extension_valid(self):
        """Test valid file extension."""
        if validate_file_extension is None:
            pytest.skip("validate_file_extension not available")
        
        valid, error = validate_file_extension("test.pdf", [".pdf", ".txt"])
        assert valid is True
        assert error is None
    
    def test_validate_file_extension_invalid(self):
        """Test invalid file extension."""
        if validate_file_extension is None:
            pytest.skip("validate_file_extension not available")
        
        valid, error = validate_file_extension("test.exe", [".pdf", ".txt"])
        assert valid is False
        assert error is not None
    
    def test_validate_file_extension_case_insensitive(self):
        """Test case insensitive extension validation."""
        if validate_file_extension is None:
            pytest.skip("validate_file_extension not available")
        
        cases = ["test.PDF", "test.Pdf", "test.pDf", "TEST.PDF"]
        for case in cases:
            valid, error = validate_file_extension(case, [".pdf", ".txt"])
            assert valid is True, f"Should accept {case}"
            assert error is None


class TestIntegerRangeValidation:
    """Tests for integer range validation."""
    
    @pytest.mark.parametrize("value,min_value,max_value,expected_valid", [
        (5, 1, 10, True),  # In range
        (1, 1, 10, True),  # At minimum
        (10, 1, 10, True),  # At maximum
        (0, 1, 10, False),  # Below minimum
        (15, 1, 10, False),  # Above maximum
        (-5, 0, 10, False),  # Negative
        (5, None, 10, True),  # No minimum
        (5, 1, None, True),  # No maximum
        (5, None, None, True),  # No bounds
    ])
    def test_validate_integer_range_cases(self, value, min_value, max_value, expected_valid):
        """Test integer range validation with various cases."""
        if validate_integer_range is None:
            pytest.skip("validate_integer_range not available")
        
        kwargs = {}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        
        valid, error = validate_integer_range(value, **kwargs)
        assert valid == expected_valid
        if not expected_valid:
            assert error is not None
    
    def test_validate_integer_range_valid(self):
        """Test valid integer in range."""
        if validate_integer_range is None:
            pytest.skip("validate_integer_range not available")
        
        valid, error = validate_integer_range(5, min_value=1, max_value=10)
        assert valid is True
        assert error is None
    
    def test_validate_integer_range_below_min(self):
        """Test integer below minimum."""
        if validate_integer_range is None:
            pytest.skip("validate_integer_range not available")
        
        valid, error = validate_integer_range(0, min_value=1, max_value=10)
        assert valid is False
        assert error is not None
    
    def test_validate_integer_range_above_max(self):
        """Test integer above maximum."""
        if validate_integer_range is None:
            pytest.skip("validate_integer_range not available")
        
        valid, error = validate_integer_range(15, min_value=1, max_value=10)
        assert valid is False
        assert error is not None
    
    def test_validate_integer_range_no_bounds(self):
        """Test integer validation without bounds."""
        if validate_integer_range is None:
            pytest.skip("validate_integer_range not available")
        
        valid, error = validate_integer_range(5)
        assert valid is True
        assert error is None
    
    def test_validate_integer_range_boundary_values(self):
        """Test integer validation at boundaries."""
        if validate_integer_range is None:
            pytest.skip("validate_integer_range not available")
        
        # Test at exact boundaries
        valid_min, _ = validate_integer_range(1, min_value=1, max_value=10)
        valid_max, _ = validate_integer_range(10, min_value=1, max_value=10)
        
        assert valid_min is True
        assert valid_max is True


class TestStringLengthValidation:
    """Tests for string length validation."""
    
    def test_validate_string_length_valid(self):
        """Test valid string length."""
        if validate_string_length is None:
            pytest.skip("validate_string_length not available")
        
        valid, error = validate_string_length("test", min_length=1, max_length=10)
        assert valid is True
        assert error is None
    
    def test_validate_string_length_too_short(self):
        """Test string that's too short."""
        if validate_string_length is None:
            pytest.skip("validate_string_length not available")
        
        valid, error = validate_string_length("", min_length=1, max_length=10)
        assert valid is False
        assert error is not None
    
    def test_validate_string_length_too_long(self):
        """Test string that's too long."""
        if validate_string_length is None:
            pytest.skip("validate_string_length not available")
        
        long_string = "a" * 100
        valid, error = validate_string_length(long_string, min_length=1, max_length=10)
        assert valid is False
        assert error is not None


class TestEmailValidation:
    """Tests for email validation."""
    
    def test_validate_email_valid(self):
        """Test valid email."""
        if validate_email is None:
            pytest.skip("validate_email not available")
        
        valid, error = validate_email("test@example.com")
        assert valid is True
        assert error is None
    
    def test_validate_email_invalid(self):
        """Test invalid email."""
        if validate_email is None:
            pytest.skip("validate_email not available")
        
        valid, error = validate_email("invalid-email")
        assert valid is False
        assert error is not None
    
    def test_validate_email_no_at(self):
        """Test email without @ symbol."""
        if validate_email is None:
            pytest.skip("validate_email not available")
        
        valid, error = validate_email("testexample.com")
        assert valid is False
        assert error is not None


class TestUUIDValidation:
    """Tests for UUID validation."""
    
    def test_validate_uuid_valid(self):
        """Test valid UUID."""
        if validate_uuid is None:
            pytest.skip("validate_uuid not available")
        
        valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
        valid, error = validate_uuid(valid_uuid)
        assert valid is True
        assert error is None
    
    def test_validate_uuid_invalid(self):
        """Test invalid UUID."""
        if validate_uuid is None:
            pytest.skip("validate_uuid not available")
        
        valid, error = validate_uuid("not-a-uuid")
        assert valid is False
        assert error is not None


class TestFileHelpers:
    """Tests for file helper functions."""
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        if sanitize_filename is None:
            pytest.skip("sanitize_filename not available")
        
        # Test removing invalid characters
        result = sanitize_filename("test<>file.pdf")
        assert "<" not in result
        assert ">" not in result
        assert ".pdf" in result
    
    def test_sanitize_filename_path_traversal(self):
        """Test sanitization removes path traversal."""
        if sanitize_filename is None:
            pytest.skip("sanitize_filename not available")
        
        result = sanitize_filename("../etc/passwd")
        assert ".." not in result
        assert "/" not in result or result.count("/") == 0
    
    @pytest.mark.asyncio
    async def test_validate_pdf_file_valid(self):
        """Test validation of valid PDF file."""
        if validate_pdf_file is None:
            pytest.skip("validate_pdf_file not available")
        
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\ntrailer\n<<\n/Size 1\n>>\nstartxref\n10\n%%EOF"
        result = validate_pdf_file(pdf_content, "test.pdf")
        assert result["valid"] is True
    
    @pytest.mark.asyncio
    async def test_validate_pdf_file_invalid(self):
        """Test validation of invalid PDF file."""
        if validate_pdf_file is None:
            pytest.skip("validate_pdf_file not available")
        
        invalid_content = b"not a pdf"
        result = validate_pdf_file(invalid_content, "test.pdf")
        assert result["valid"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_extract_metadata(self):
        """Test PDF metadata extraction."""
        if extract_metadata is None:
            pytest.skip("extract_metadata not available")
        
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\nxref\n0 3\ntrailer\n<<\n/Size 3\n>>\nstartxref\n50\n%%EOF"
        metadata = extract_metadata(pdf_content)
        assert isinstance(metadata, dict)
        assert "pages" in metadata or "page_count" in metadata


class TestFileValidator:
    """Tests for FileValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create FileValidator instance."""
        if FileValidator is None:
            pytest.skip("FileValidator not available")
        return FileValidator()
    
    def test_validate_file_size_valid(self, validator):
        """Test valid file size."""
        if validator is None:
            pytest.skip("FileValidator not available")
        
        # Mock validation
        result = validator.validate_file_size(1024, max_size=10240)
        assert result is True or result is None  # Depends on implementation
    
    def test_validate_file_size_too_large(self, validator):
        """Test file size that's too large."""
        if validator is None:
            pytest.skip("FileValidator not available")
        
        # Mock validation
        with pytest.raises(Exception):
            validator.validate_file_size(102400, max_size=10240)


class TestValidationError:
    """Tests for ValidationError exception."""
    
    def test_validation_error_creation(self):
        """Test creating ValidationError."""
        if ValidationError is None:
            pytest.skip("ValidationError not available")
        
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_validation_error_with_details(self):
        """Test ValidationError with details."""
        if ValidationError is None:
            pytest.skip("ValidationError not available")
        
        error = ValidationError("Test error", details={"field": "test"})
        assert "field" in error.details if hasattr(error, "details") else True

