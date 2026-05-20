"""
Unit Tests for Custom Exceptions
=================================
Tests for custom exception classes.
"""

import pytest
from exceptions import (
    PDFVariantesError,
    PDFNotFoundError,
    InvalidFileError,
    FileSizeError,
    ProcessingError,
    VariantGenerationError
)


class TestPDFVariantesError:
    """Tests for base PDFVariantesError exception."""
    
    def test_base_exception_creation(self):
        """Test creating base exception."""
        error = PDFVariantesError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}
    
    def test_base_exception_with_error_code(self):
        """Test base exception with error code."""
        error = PDFVariantesError(
            "Test error",
            error_code="TEST_ERROR"
        )
        assert error.error_code == "TEST_ERROR"
    
    def test_base_exception_with_details(self):
        """Test base exception with details."""
        details = {"file_id": "123", "reason": "not found"}
        error = PDFVariantesError(
            "Test error",
            error_code="TEST_ERROR",
            details=details
        )
        assert error.details == details
        assert error.details["file_id"] == "123"
    
    def test_base_exception_inheritance(self):
        """Test that base exception inherits from Exception."""
        error = PDFVariantesError("Test")
        assert isinstance(error, Exception)
        assert isinstance(error, PDFVariantesError)
    
    @pytest.mark.parametrize("message,error_code,details", [
        ("Simple error", None, {}),
        ("Error with code", "ERROR_CODE", {}),
        ("Error with details", None, {"key": "value"}),
        ("Full error", "ERROR_CODE", {"key": "value", "nested": {"data": 123}}),
    ])
    def test_base_exception_variations(self, message, error_code, details):
        """Test base exception with various parameter combinations."""
        error = PDFVariantesError(
            message=message,
            error_code=error_code,
            details=details
        )
        assert error.message == message
        assert error.error_code == error_code
        assert error.details == details
    
    def test_base_exception_repr(self):
        """Test exception string representation."""
        error = PDFVariantesError("Test error", error_code="TEST")
        error_str = str(error)
        assert "Test error" in error_str
    
    def test_base_exception_details_mutation(self):
        """Test that exception details can be modified."""
        error = PDFVariantesError("Test", details={"count": 0})
        error.details["count"] = 1
        assert error.details["count"] == 1


class TestPDFNotFoundError:
    """Tests for PDFNotFoundError exception."""
    
    def test_pdf_not_found_creation(self):
        """Test creating PDFNotFoundError."""
        file_id = "test_file_123"
        error = PDFNotFoundError(file_id)
        
        assert file_id in str(error)
        assert error.error_code == "PDF_NOT_FOUND"
        assert error.details["file_id"] == file_id
    
    def test_pdf_not_found_message(self):
        """Test PDFNotFoundError message format."""
        file_id = "test_file_456"
        error = PDFNotFoundError(file_id)
        
        assert f"PDF file not found: {file_id}" in error.message
        assert isinstance(error, PDFVariantesError)
    
    def test_pdf_not_found_details(self):
        """Test PDFNotFoundError details."""
        file_id = "test_file_789"
        error = PDFNotFoundError(file_id)
        
        assert "file_id" in error.details
        assert error.details["file_id"] == file_id


class TestInvalidFileError:
    """Tests for InvalidFileError exception."""
    
    def test_invalid_file_error_creation(self):
        """Test creating InvalidFileError."""
        error = InvalidFileError("File is invalid")
        
        assert error.message == "File is invalid"
        assert error.error_code == "INVALID_FILE"
        assert error.details.get("file_type") is None
    
    def test_invalid_file_error_with_file_type(self):
        """Test InvalidFileError with file type."""
        error = InvalidFileError("File is invalid", file_type="pdf")
        
        assert error.details["file_type"] == "pdf"
    
    def test_invalid_file_error_inheritance(self):
        """Test InvalidFileError inheritance."""
        error = InvalidFileError("Test")
        assert isinstance(error, PDFVariantesError)


class TestFileSizeError:
    """Tests for FileSizeError exception."""
    
    def test_file_size_error_creation(self):
        """Test creating FileSizeError."""
        error = FileSizeError("File too large", max_size=1024, actual_size=2048)
        
        assert error.message == "File too large"
        assert error.error_code == "FILE_SIZE_ERROR"
        assert error.details["max_size"] == 1024
        assert error.details["actual_size"] == 2048
    
    def test_file_size_error_details(self):
        """Test FileSizeError details."""
        error = FileSizeError("File too large", max_size=1024, actual_size=2048)
        
        assert "max_size" in error.details
        assert "actual_size" in error.details
        assert error.details["max_size"] < error.details["actual_size"]


class TestProcessingError:
    """Tests for ProcessingError exception."""
    
    def test_processing_error_creation(self):
        """Test creating ProcessingError."""
        error = ProcessingError("Processing failed", stage="upload")
        
        assert error.message == "Processing failed"
        assert error.error_code == "PROCESSING_ERROR"
        assert error.details["stage"] == "upload"
    
    def test_processing_error_with_stage(self):
        """Test ProcessingError with different stages."""
        stages = ["upload", "extract", "generate", "save"]
        
        for stage in stages:
            error = ProcessingError("Error", stage=stage)
            assert error.details["stage"] == stage


class TestVariantGenerationError:
    """Tests for VariantGenerationError exception."""
    
    def test_variant_generation_error_creation(self):
        """Test creating VariantGenerationError."""
        error = VariantGenerationError(
            "Variant generation failed",
            variant_type="summary"
        )
        
        assert error.message == "Variant generation failed"
        assert error.error_code == "VARIANT_GENERATION_ERROR"
        assert error.details["variant_type"] == "summary"
    
    def test_variant_generation_error_with_type(self):
        """Test VariantGenerationError with variant type."""
        variant_types = ["summary", "outline", "highlights", "notes"]
        
        for variant_type in variant_types:
            error = VariantGenerationError("Error", variant_type=variant_type)
            assert error.details["variant_type"] == variant_type


class TestExceptionChaining:
    """Tests for exception chaining."""
    
    def test_exception_chaining(self):
        """Test that exceptions can be chained."""
        try:
            raise PDFNotFoundError("file_123")
        except PDFNotFoundError as e:
            assert e.error_code == "PDF_NOT_FOUND"
            assert "file_123" in str(e)
    
    def test_exception_catching_base(self):
        """Test catching base exception."""
        try:
            raise PDFNotFoundError("file_123")
        except PDFVariantesError as e:
            assert isinstance(e, PDFVariantesError)
            assert e.error_code == "PDF_NOT_FOUND"
    
    def test_exception_catching_specific(self):
        """Test catching specific exception."""
        try:
            raise InvalidFileError("Invalid file")
        except InvalidFileError as e:
            assert e.error_code == "INVALID_FILE"
        except PDFVariantesError:
            pytest.fail("Should catch specific exception first")


class TestExceptionDetails:
    """Tests for exception details handling."""
    
    def test_exception_details_preservation(self):
        """Test that exception details are preserved."""
        details = {
            "file_id": "123",
            "user_id": "456",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        error = PDFVariantesError("Error", details=details)
        assert error.details == details
        assert error.details["file_id"] == "123"
        assert error.details["user_id"] == "456"
    
    def test_exception_details_modification(self):
        """Test modifying exception details."""
        error = PDFVariantesError("Error", details={"count": 0})
        error.details["count"] = 1
        
        assert error.details["count"] == 1
    
    def test_exception_empty_details(self):
        """Test exception with empty details."""
        error = PDFVariantesError("Error")
        assert error.details == {}
        
        # Should not raise error when accessing
        assert error.details.get("key") is None

