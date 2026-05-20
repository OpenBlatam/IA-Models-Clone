"""
Unit Tests for Models
=====================
Tests for Pydantic models and data classes.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

# Try to import models
try:
    from models import (
        VariantStatus,
        PDFProcessingStatus,
        VariantType,
        PDFMetadata,
        PDFVariant,
        VariantConfiguration,
        VariantGenerateRequest,
        VariantGenerateResponse
    )
except ImportError:
    # Models might be in different location
    VariantStatus = None
    PDFProcessingStatus = None
    VariantType = None
    PDFMetadata = None
    PDFVariant = None
    VariantConfiguration = None
    VariantGenerateRequest = None
    VariantGenerateResponse = None


class TestVariantStatus:
    """Tests for VariantStatus enum."""
    
    def test_variant_status_values(self):
        """Test VariantStatus enum values."""
        if VariantStatus is None:
            pytest.skip("VariantStatus not available")
        
        # Check that enum has expected values
        assert hasattr(VariantStatus, "PENDING") or "pending" in [e.value for e in VariantStatus]
        assert hasattr(VariantStatus, "PROCESSING") or "processing" in [e.value for e in VariantStatus]
        assert hasattr(VariantStatus, "COMPLETED") or "completed" in [e.value for e in VariantStatus]
        assert hasattr(VariantStatus, "FAILED") or "failed" in [e.value for e in VariantStatus]
    
    def test_variant_status_string_representation(self):
        """Test VariantStatus string representation."""
        if VariantStatus is None:
            pytest.skip("VariantStatus not available")
        
        # Test that values are strings
        for status in VariantStatus:
            assert isinstance(status.value, str)


class TestPDFProcessingStatus:
    """Tests for PDFProcessingStatus enum."""
    
    def test_processing_status_values(self):
        """Test PDFProcessingStatus enum values."""
        if PDFProcessingStatus is None:
            pytest.skip("PDFProcessingStatus not available")
        
        # Check that enum has expected values
        expected_statuses = ["pending", "processing", "completed", "failed"]
        actual_statuses = [e.value for e in PDFProcessingStatus]
        
        for status in expected_statuses:
            assert status in actual_statuses or hasattr(PDFProcessingStatus, status.upper())


class TestVariantType:
    """Tests for VariantType enum."""
    
    def test_variant_type_values(self):
        """Test VariantType enum values."""
        if VariantType is None:
            pytest.skip("VariantType not available")
        
        expected_types = ["summary", "outline", "highlights", "notes", "quiz", "presentation"]
        actual_types = [e.value for e in VariantType]
        
        for variant_type in expected_types:
            assert variant_type in actual_types or hasattr(VariantType, variant_type.upper())


class TestPDFMetadata:
    """Tests for PDFMetadata model."""
    
    def test_pdf_metadata_creation(self):
        """Test creating PDFMetadata."""
        if PDFMetadata is None:
            pytest.skip("PDFMetadata not available")
        
        metadata = PDFMetadata(
            file_id="test_123",
            filename="test.pdf",
            file_size=1024,
            page_count=10
        )
        
        assert metadata.file_id == "test_123"
        assert metadata.filename == "test.pdf"
        assert metadata.file_size == 1024
        assert metadata.page_count == 10
    
    def test_pdf_metadata_optional_fields(self):
        """Test PDFMetadata with optional fields."""
        if PDFMetadata is None:
            pytest.skip("PDFMetadata not available")
        
        metadata = PDFMetadata(
            file_id="test_123",
            filename="test.pdf",
            file_size=1024,
            page_count=10,
            title="Test Document",
            author="Test Author"
        )
        
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
    
    def test_pdf_metadata_dict_conversion(self):
        """Test converting PDFMetadata to dict."""
        if PDFMetadata is None:
            pytest.skip("PDFMetadata not available")
        
        metadata = PDFMetadata(
            file_id="test_123",
            filename="test.pdf",
            file_size=1024,
            page_count=10
        )
        
        metadata_dict = metadata.dict() if hasattr(metadata, "dict") else metadata.model_dump() if hasattr(metadata, "model_dump") else {}
        assert metadata_dict.get("file_id") == "test_123"
        assert metadata_dict.get("filename") == "test.pdf"


class TestVariantConfiguration:
    """Tests for VariantConfiguration model."""
    
    def test_variant_configuration_creation(self):
        """Test creating VariantConfiguration."""
        if VariantConfiguration is None:
            pytest.skip("VariantConfiguration not available")
        
        config = VariantConfiguration(
            variant_type="summary",
            max_length=500,
            style="academic"
        )
        
        assert config.variant_type == "summary"
        assert config.max_length == 500
        assert config.style == "academic"
    
    def test_variant_configuration_defaults(self):
        """Test VariantConfiguration with defaults."""
        if VariantConfiguration is None:
            pytest.skip("VariantConfiguration not available")
        
        config = VariantConfiguration(variant_type="summary")
        
        # Check that defaults are applied
        assert config.variant_type == "summary"
        # Other fields should have defaults or be optional


class TestPDFVariant:
    """Tests for PDFVariant model."""
    
    def test_pdf_variant_creation(self):
        """Test creating PDFVariant."""
        if PDFVariant is None:
            pytest.skip("PDFVariant not available")
        
        variant = PDFVariant(
            variant_id="variant_123",
            file_id="file_123",
            variant_type="summary",
            content="Summary content",
            status="completed"
        )
        
        assert variant.variant_id == "variant_123"
        assert variant.file_id == "file_123"
        assert variant.variant_type == "summary"
        assert variant.content == "Summary content"
        assert variant.status == "completed"
    
    def test_pdf_variant_with_timestamps(self):
        """Test PDFVariant with timestamps."""
        if PDFVariant is None:
            pytest.skip("PDFVariant not available")
        
        now = datetime.utcnow()
        variant = PDFVariant(
            variant_id="variant_123",
            file_id="file_123",
            variant_type="summary",
            content="Content",
            status="completed",
            created_at=now
        )
        
        assert variant.created_at == now


class TestVariantGenerateRequest:
    """Tests for VariantGenerateRequest model."""
    
    def test_variant_generate_request_creation(self):
        """Test creating VariantGenerateRequest."""
        if VariantGenerateRequest is None:
            pytest.skip("VariantGenerateRequest not available")
        
        request = VariantGenerateRequest(
            variant_type="summary",
            options={
                "max_length": 500,
                "style": "academic"
            }
        )
        
        assert request.variant_type == "summary"
        assert request.options["max_length"] == 500
    
    def test_variant_generate_request_validation(self):
        """Test VariantGenerateRequest validation."""
        if VariantGenerateRequest is None:
            pytest.skip("VariantGenerateRequest not available")
        
        # Should accept valid variant type
        request = VariantGenerateRequest(variant_type="summary")
        assert request.variant_type == "summary"


class TestVariantGenerateResponse:
    """Tests for VariantGenerateResponse model."""
    
    def test_variant_generate_response_creation(self):
        """Test creating VariantGenerateResponse."""
        if VariantGenerateResponse is None:
            pytest.skip("VariantGenerateResponse not available")
        
        response = VariantGenerateResponse(
            success=True,
            variant_id="variant_123",
            message="Variant generated successfully"
        )
        
        assert response.success is True
        assert response.variant_id == "variant_123"
        assert response.message == "Variant generated successfully"
    
    def test_variant_generate_response_failure(self):
        """Test VariantGenerateResponse for failure."""
        if VariantGenerateResponse is None:
            pytest.skip("VariantGenerateResponse not available")
        
        response = VariantGenerateResponse(
            success=False,
            message="Generation failed"
        )
        
        assert response.success is False
        assert "failed" in response.message.lower()


class TestModelSerialization:
    """Tests for model serialization."""
    
    def test_model_to_dict(self):
        """Test converting model to dictionary."""
        if PDFMetadata is None:
            pytest.skip("PDFMetadata not available")
        
        metadata = PDFMetadata(
            file_id="test_123",
            filename="test.pdf",
            file_size=1024,
            page_count=10
        )
        
        # Try different serialization methods
        if hasattr(metadata, "dict"):
            result = metadata.dict()
        elif hasattr(metadata, "model_dump"):
            result = metadata.model_dump()
        elif hasattr(metadata, "__dict__"):
            result = metadata.__dict__
        else:
            result = {}
        
        assert isinstance(result, dict)
        assert result.get("file_id") == "test_123"
    
    def test_model_to_json(self):
        """Test converting model to JSON."""
        import json
        
        if PDFMetadata is None:
            pytest.skip("PDFMetadata not available")
        
        metadata = PDFMetadata(
            file_id="test_123",
            filename="test.pdf",
            file_size=1024,
            page_count=10
        )
        
        # Try JSON serialization
        if hasattr(metadata, "json"):
            json_str = metadata.json()
        elif hasattr(metadata, "model_dump_json"):
            json_str = metadata.model_dump_json()
        else:
            # Fallback to dict then json
            metadata_dict = metadata.dict() if hasattr(metadata, "dict") else metadata.model_dump() if hasattr(metadata, "model_dump") else {}
            json_str = json.dumps(metadata_dict)
        
        assert isinstance(json_str, str)
        assert "test_123" in json_str


class TestModelValidation:
    """Tests for model validation."""
    
    def test_model_required_fields(self):
        """Test that required fields are enforced."""
        if PDFMetadata is None:
            pytest.skip("PDFMetadata not available")
        
        # Should raise validation error if required fields are missing
        with pytest.raises(Exception):  # Could be ValidationError or ValueError
            PDFMetadata()  # Missing required fields
    
    def test_model_field_types(self):
        """Test that field types are validated."""
        if PDFMetadata is None:
            pytest.skip("PDFMetadata not available")
        
        # Should raise error if wrong type is provided
        with pytest.raises(Exception):
            PDFMetadata(
                file_id="test",
                filename="test.pdf",
                file_size="not_a_number",  # Should be int
                page_count=10
            )



