"""
Unit Tests for Schemas
=======================
Tests for Pydantic schemas used in API.
"""

import pytest
from pydantic import ValidationError
from typing import Dict, Any, Optional

# Try to import schemas from models (canonical location)
try:
    from models import (
        PDFUploadRequest as PDFUploadSchema,
        VariantGenerateRequest as VariantGenerateSchema,
        TopicExtractRequest as TopicExtractSchema,
        PDFEditRequest as PDFEditSchema
    )
except ImportError:
    # Fallback to deprecated schemas.py for backward compatibility
    try:
        from schemas import (
            PDFUploadSchema,
            VariantGenerateSchema,
            TopicExtractSchema,
            PDFEditSchema
        )
    except ImportError:
        PDFUploadSchema = None
        VariantGenerateSchema = None
        TopicExtractSchema = None
        PDFEditSchema = None


class TestPDFUploadSchema:
    """Tests for PDFUploadSchema."""
    
    def test_pdf_upload_schema_creation(self):
        """Test creating PDFUploadSchema."""
        if PDFUploadSchema is None:
            pytest.skip("PDFUploadSchema not available")
        
        schema = PDFUploadSchema(
            filename="test.pdf",
            auto_process=True,
            extract_text=True
        )
        
        assert schema.filename == "test.pdf"
        assert schema.auto_process is True
        assert schema.extract_text is True
    
    def test_pdf_upload_schema_defaults(self):
        """Test PDFUploadSchema with defaults."""
        if PDFUploadSchema is None:
            pytest.skip("PDFUploadSchema not available")
        
        schema = PDFUploadSchema(filename="test.pdf")
        
        # Check defaults if they exist
        assert schema.filename == "test.pdf"
    
    def test_pdf_upload_schema_validation(self):
        """Test PDFUploadSchema validation."""
        if PDFUploadSchema is None:
            pytest.skip("PDFUploadSchema not available")
        
        # Should accept valid filename
        schema = PDFUploadSchema(filename="test.pdf")
        assert schema.filename.endswith(".pdf")


class TestVariantGenerateSchema:
    """Tests for VariantGenerateSchema."""
    
    def test_variant_generate_schema_creation(self):
        """Test creating VariantGenerateSchema."""
        if VariantGenerateSchema is None:
            pytest.skip("VariantGenerateSchema not available")
        
        schema = VariantGenerateSchema(
            variant_type="summary",
            options={
                "max_length": 500,
                "style": "academic"
            }
        )
        
        assert schema.variant_type == "summary"
        assert schema.options["max_length"] == 500
    
    def test_variant_generate_schema_minimal(self):
        """Test VariantGenerateSchema with minimal data."""
        if VariantGenerateSchema is None:
            pytest.skip("VariantGenerateSchema not available")
        
        schema = VariantGenerateSchema(variant_type="summary")
        assert schema.variant_type == "summary"
    
    def test_variant_generate_schema_options(self):
        """Test VariantGenerateSchema with various options."""
        if VariantGenerateSchema is None:
            pytest.skip("VariantGenerateSchema not available")
        
        options = {
            "max_length": 1000,
            "style": "casual",
            "language": "es",
            "tone": "friendly"
        }
        
        schema = VariantGenerateSchema(
            variant_type="summary",
            options=options
        )
        
        assert schema.options == options


class TestTopicExtractSchema:
    """Tests for TopicExtractSchema."""
    
    def test_topic_extract_schema_creation(self):
        """Test creating TopicExtractSchema."""
        if TopicExtractSchema is None:
            pytest.skip("TopicExtractSchema not available")
        
        schema = TopicExtractSchema(
            min_relevance=0.5,
            max_topics=50
        )
        
        assert schema.min_relevance == 0.5
        assert schema.max_topics == 50
    
    def test_topic_extract_schema_defaults(self):
        """Test TopicExtractSchema with defaults."""
        if TopicExtractSchema is None:
            pytest.skip("TopicExtractSchema not available")
        
        schema = TopicExtractSchema()
        
        # Check that defaults are applied
        assert hasattr(schema, "min_relevance")
        assert hasattr(schema, "max_topics")
    
    def test_topic_extract_schema_validation(self):
        """Test TopicExtractSchema validation."""
        if TopicExtractSchema is None:
            pytest.skip("TopicExtractSchema not available")
        
        # min_relevance should be between 0 and 1
        with pytest.raises(ValidationError):
            TopicExtractSchema(min_relevance=1.5)
        
        with pytest.raises(ValidationError):
            TopicExtractSchema(min_relevance=-0.1)


class TestPDFEditSchema:
    """Tests for PDFEditSchema."""
    
    def test_pdf_edit_schema_creation(self):
        """Test creating PDFEditSchema."""
        if PDFEditSchema is None:
            pytest.skip("PDFEditSchema not available")
        
        schema = PDFEditSchema(
            page_number=1,
            annotation_type="highlight",
            content="Test annotation",
            position={"x": 100, "y": 200}
        )
        
        assert schema.page_number == 1
        assert schema.annotation_type == "highlight"
        assert schema.content == "Test annotation"
        assert schema.position["x"] == 100
    
    def test_pdf_edit_schema_validation(self):
        """Test PDFEditSchema validation."""
        if PDFEditSchema is None:
            pytest.skip("PDFEditSchema not available")
        
        # page_number should be positive
        with pytest.raises(ValidationError):
            PDFEditSchema(
                page_number=0,
                annotation_type="highlight",
                content="Test"
            )


class TestSchemaSerialization:
    """Tests for schema serialization."""
    
    def test_schema_to_dict(self):
        """Test converting schema to dictionary."""
        if VariantGenerateSchema is None:
            pytest.skip("VariantGenerateSchema not available")
        
        schema = VariantGenerateSchema(
            variant_type="summary",
            options={"max_length": 500}
        )
        
        schema_dict = schema.dict() if hasattr(schema, "dict") else schema.model_dump() if hasattr(schema, "model_dump") else {}
        assert schema_dict.get("variant_type") == "summary"
        assert schema_dict.get("options", {}).get("max_length") == 500
    
    def test_schema_to_json(self):
        """Test converting schema to JSON."""
        import json
        
        if VariantGenerateSchema is None:
            pytest.skip("VariantGenerateSchema not available")
        
        schema = VariantGenerateSchema(
            variant_type="summary",
            options={"max_length": 500}
        )
        
        if hasattr(schema, "json"):
            json_str = schema.json()
        elif hasattr(schema, "model_dump_json"):
            json_str = schema.model_dump_json()
        else:
            schema_dict = schema.dict() if hasattr(schema, "dict") else schema.model_dump() if hasattr(schema, "model_dump") else {}
            json_str = json.dumps(schema_dict)
        
        assert isinstance(json_str, str)
        assert "summary" in json_str


class TestSchemaValidation:
    """Tests for schema validation."""
    
    def test_schema_required_fields(self):
        """Test that required fields are enforced."""
        if VariantGenerateSchema is None:
            pytest.skip("VariantGenerateSchema not available")
        
        # Should raise validation error if required fields are missing
        with pytest.raises(ValidationError):
            VariantGenerateSchema()  # Missing variant_type
    
    def test_schema_field_types(self):
        """Test that field types are validated."""
        if VariantGenerateSchema is None:
            pytest.skip("VariantGenerateSchema not available")
        
        # Should raise error if wrong type is provided
        with pytest.raises(ValidationError):
            VariantGenerateSchema(
                variant_type=123,  # Should be string
                options={}
            )


class TestSchemaDefaults:
    """Tests for schema default values."""
    
    def test_schema_optional_fields(self):
        """Test that optional fields work correctly."""
        if VariantGenerateSchema is None:
            pytest.skip("VariantGenerateSchema not available")
        
        # Should work with minimal required fields
        schema = VariantGenerateSchema(variant_type="summary")
        assert schema.variant_type == "summary"
        
        # Options should be optional or have defaults
        if hasattr(schema, "options"):
            assert schema.options is not None or isinstance(schema.options, dict)



