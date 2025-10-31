"""
Test Professional Documents Models
=================================

Unit tests for the Professional Documents models.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from ..models import (
    DocumentGenerationRequest,
    DocumentGenerationResponse,
    DocumentExportRequest,
    DocumentExportResponse,
    DocumentTemplate,
    DocumentStyle,
    ProfessionalDocument,
    DocumentSection,
    DocumentType,
    ExportFormat
)


class TestModels:
    """Test cases for Professional Documents models."""
    
    def test_document_generation_request_validation(self):
        """Test DocumentGenerationRequest validation."""
        
        # Valid request
        request = DocumentGenerationRequest(
            query="Create a business proposal for implementing a CRM system",
            document_type=DocumentType.PROPOSAL,
            title="CRM Proposal",
            author="John Smith",
            company="Tech Solutions",
            tone="professional",
            length="medium"
        )
        
        assert request.query == "Create a business proposal for implementing a CRM system"
        assert request.document_type == DocumentType.PROPOSAL
        assert request.title == "CRM Proposal"
        assert request.author == "John Smith"
        assert request.company == "Tech Solutions"
        assert request.tone == "professional"
        assert request.length == "medium"
        assert request.language == "en"  # Default value
    
    def test_document_generation_request_invalid_query(self):
        """Test DocumentGenerationRequest with invalid query."""
        
        with pytest.raises(ValueError, match="Query must be at least 10 characters long"):
            DocumentGenerationRequest(
                query="short",  # Too short
                document_type=DocumentType.REPORT
            )
    
    def test_document_style_validation(self):
        """Test DocumentStyle validation."""
        
        style = DocumentStyle(
            font_family="Arial",
            font_size=12,
            line_spacing=1.5,
            margin_top=1.0,
            margin_bottom=1.0,
            margin_left=1.0,
            margin_right=1.0,
            header_color="#2c3e50",
            body_color="#333333",
            accent_color="#3498db",
            background_color="#ffffff",
            include_page_numbers=True
        )
        
        assert style.font_family == "Arial"
        assert style.font_size == 12
        assert style.line_spacing == 1.5
        assert style.header_color == "#2c3e50"
        assert style.include_page_numbers is True
    
    def test_document_style_invalid_font_size(self):
        """Test DocumentStyle with invalid font size."""
        
        with pytest.raises(ValueError):
            DocumentStyle(
                font_size=5,  # Too small
                header_color="#2c3e50"
            )
        
        with pytest.raises(ValueError):
            DocumentStyle(
                font_size=30,  # Too large
                header_color="#2c3e50"
            )
    
    def test_document_template_creation(self):
        """Test DocumentTemplate creation."""
        
        template = DocumentTemplate(
            name="Test Template",
            description="A test template",
            document_type=DocumentType.REPORT,
            sections=["Introduction", "Body", "Conclusion"]
        )
        
        assert template.name == "Test Template"
        assert template.description == "A test template"
        assert template.document_type == DocumentType.REPORT
        assert template.sections == ["Introduction", "Body", "Conclusion"]
        assert template.id is not None
        assert template.created_at is not None
        assert template.updated_at is not None
    
    def test_document_section_creation(self):
        """Test DocumentSection creation."""
        
        section = DocumentSection(
            title="Introduction",
            content="This is the introduction section.",
            level=1,
            order=0
        )
        
        assert section.title == "Introduction"
        assert section.content == "This is the introduction section."
        assert section.level == 1
        assert section.order == 0
        assert section.metadata == {}
    
    def test_document_section_invalid_level(self):
        """Test DocumentSection with invalid level."""
        
        with pytest.raises(ValueError):
            DocumentSection(
                title="Test",
                content="Test content",
                level=0,  # Too small
                order=0
            )
        
        with pytest.raises(ValueError):
            DocumentSection(
                title="Test",
                content="Test content",
                level=7,  # Too large
                order=0
            )
    
    def test_professional_document_creation(self):
        """Test ProfessionalDocument creation."""
        
        document = ProfessionalDocument(
            title="Test Document",
            document_type=DocumentType.REPORT,
            template_id="template_123",
            author="Test Author",
            company="Test Company"
        )
        
        assert document.title == "Test Document"
        assert document.document_type == DocumentType.REPORT
        assert document.template_id == "template_123"
        assert document.author == "Test Author"
        assert document.company == "Test Company"
        assert document.id is not None
        assert document.date_created is not None
        assert document.date_modified is not None
        assert document.sections == []
        assert document.word_count == 0
        assert document.page_count == 0
        assert document.status == "draft"
    
    def test_document_export_request_creation(self):
        """Test DocumentExportRequest creation."""
        
        request = DocumentExportRequest(
            document_id="doc_123",
            format=ExportFormat.PDF,
            custom_filename="test_document.pdf"
        )
        
        assert request.document_id == "doc_123"
        assert request.format == ExportFormat.PDF
        assert request.custom_filename == "test_document.pdf"
        assert request.include_metadata is True  # Default value
        assert request.password_protect is False  # Default value
    
    def test_document_generation_response_creation(self):
        """Test DocumentGenerationResponse creation."""
        
        document = ProfessionalDocument(
            title="Test Document",
            document_type=DocumentType.REPORT,
            template_id="template_123"
        )
        
        response = DocumentGenerationResponse(
            success=True,
            document=document,
            message="Document generated successfully",
            generation_time=1.5,
            word_count=500,
            estimated_pages=2
        )
        
        assert response.success is True
        assert response.document == document
        assert response.message == "Document generated successfully"
        assert response.generation_time == 1.5
        assert response.word_count == 500
        assert response.estimated_pages == 2
    
    def test_document_export_response_creation(self):
        """Test DocumentExportResponse creation."""
        
        response = DocumentExportResponse(
            success=True,
            file_path="/path/to/file.pdf",
            file_size=1024,
            download_url="/download/file.pdf",
            message="Export successful",
            export_time=0.5
        )
        
        assert response.success is True
        assert response.file_path == "/path/to/file.pdf"
        assert response.file_size == 1024
        assert response.download_url == "/download/file.pdf"
        assert response.message == "Export successful"
        assert response.export_time == 0.5
    
    def test_enum_values(self):
        """Test enum values."""
        
        # DocumentType enum
        assert DocumentType.REPORT.value == "report"
        assert DocumentType.PROPOSAL.value == "proposal"
        assert DocumentType.MANUAL.value == "manual"
        assert DocumentType.TECHNICAL_DOCUMENT.value == "technical_document"
        
        # ExportFormat enum
        assert ExportFormat.PDF.value == "pdf"
        assert ExportFormat.WORD.value == "docx"
        assert ExportFormat.MARKDOWN.value == "md"
        assert ExportFormat.HTML.value == "html"
    
    def test_model_serialization(self):
        """Test model serialization to dict."""
        
        request = DocumentGenerationRequest(
            query="Create a test document",
            document_type=DocumentType.REPORT,
            title="Test Document"
        )
        
        # Test serialization
        data = request.dict()
        assert isinstance(data, dict)
        assert data["query"] == "Create a test document"
        assert data["document_type"] == "report"
        assert data["title"] == "Test Document"
        
        # Test JSON serialization
        json_data = request.json()
        assert isinstance(json_data, str)
        assert "Create a test document" in json_data
    
    def test_model_deserialization(self):
        """Test model deserialization from dict."""
        
        data = {
            "query": "Create a test document",
            "document_type": "report",
            "title": "Test Document",
            "author": "Test Author"
        }
        
        request = DocumentGenerationRequest(**data)
        
        assert request.query == "Create a test document"
        assert request.document_type == DocumentType.REPORT
        assert request.title == "Test Document"
        assert request.author == "Test Author"
    
    def test_default_values(self):
        """Test default values in models."""
        
        request = DocumentGenerationRequest(
            query="Create a test document with minimum required fields",
            document_type=DocumentType.REPORT
        )
        
        # Test default values
        assert request.language == "en"
        assert request.tone == "professional"
        assert request.length == "medium"
        assert request.title is None
        assert request.subtitle is None
        assert request.author is None
        assert request.company is None
        assert request.style is None
        assert request.additional_requirements is None
    
    def test_optional_fields(self):
        """Test optional fields in models."""
        
        # Test with all optional fields
        request = DocumentGenerationRequest(
            query="Create a comprehensive test document with all fields",
            document_type=DocumentType.REPORT,
            title="Comprehensive Test Document",
            subtitle="A detailed test document",
            author="Test Author",
            company="Test Company",
            additional_requirements="Include detailed analysis"
        )
        
        assert request.title == "Comprehensive Test Document"
        assert request.subtitle == "A detailed test document"
        assert request.author == "Test Author"
        assert request.company == "Test Company"
        assert request.additional_requirements == "Include detailed analysis"




























