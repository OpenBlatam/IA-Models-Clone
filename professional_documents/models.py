"""
Professional Document Models
===========================

Pydantic models for the professional document generation system.
"""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field, validator
from typing_extensions import TypedDict


class DocumentType(str, Enum):
    """Types of professional documents that can be generated."""
    REPORT = "report"
    PROPOSAL = "proposal"
    PRESENTATION = "presentation"
    MANUAL = "manual"
    GUIDE = "guide"
    WHITEPAPER = "whitepaper"
    BUSINESS_PLAN = "business_plan"
    TECHNICAL_DOCUMENT = "technical_document"
    ACADEMIC_PAPER = "academic_paper"
    NEWSLETTER = "newsletter"
    BROCHURE = "brochure"
    CATALOG = "catalog"


class ExportFormat(str, Enum):
    """Supported export formats."""
    PDF = "pdf"
    MARKDOWN = "md"
    WORD = "docx"
    HTML = "html"


class DocumentStyle(BaseModel):
    """Document styling configuration."""
    font_family: str = Field(default="Arial", description="Primary font family")
    font_size: int = Field(default=12, ge=8, le=24, description="Base font size")
    line_spacing: float = Field(default=1.5, ge=1.0, le=3.0, description="Line spacing")
    margin_top: float = Field(default=1.0, ge=0.5, le=3.0, description="Top margin in inches")
    margin_bottom: float = Field(default=1.0, ge=0.5, le=3.0, description="Bottom margin in inches")
    margin_left: float = Field(default=1.0, ge=0.5, le=3.0, description="Left margin in inches")
    margin_right: float = Field(default=1.0, ge=0.5, le=3.0, description="Right margin in inches")
    header_color: str = Field(default="#2c3e50", description="Header text color (hex)")
    body_color: str = Field(default="#333333", description="Body text color (hex)")
    accent_color: str = Field(default="#3498db", description="Accent color for highlights (hex)")
    background_color: str = Field(default="#ffffff", description="Background color (hex)")
    include_page_numbers: bool = Field(default=True, description="Include page numbers")
    include_watermark: bool = Field(default=False, description="Include watermark")
    watermark_text: Optional[str] = Field(default=None, description="Watermark text")


class DocumentTemplate(BaseModel):
    """Document template configuration."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    document_type: DocumentType = Field(..., description="Type of document")
    sections: List[str] = Field(..., description="Required sections for this template")
    style: DocumentStyle = Field(default_factory=DocumentStyle, description="Default styling")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional template metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentSection(BaseModel):
    """A section within a document."""
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    level: int = Field(default=1, ge=1, le=6, description="Heading level (1-6)")
    order: int = Field(..., description="Section order in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Section metadata")


class ProfessionalDocument(BaseModel):
    """A professional document with all its components."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Document ID")
    title: str = Field(..., description="Document title")
    subtitle: Optional[str] = Field(default=None, description="Document subtitle")
    document_type: DocumentType = Field(..., description="Type of document")
    template_id: str = Field(..., description="Template used for generation")
    author: Optional[str] = Field(default=None, description="Document author")
    company: Optional[str] = Field(default=None, description="Company name")
    date_created: datetime = Field(default_factory=datetime.utcnow)
    date_modified: datetime = Field(default_factory=datetime.utcnow)
    sections: List[DocumentSection] = Field(default_factory=list, description="Document sections")
    style: DocumentStyle = Field(default_factory=DocumentStyle, description="Document styling")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    word_count: int = Field(default=0, description="Total word count")
    page_count: int = Field(default=0, description="Estimated page count")
    status: Literal["draft", "generating", "completed", "error"] = Field(default="draft")
    error_message: Optional[str] = Field(default=None, description="Error message if status is error")


class DocumentGenerationRequest(BaseModel):
    """Request to generate a professional document."""
    query: str = Field(..., min_length=10, description="User query describing the document to generate")
    document_type: DocumentType = Field(..., description="Type of document to generate")
    template_id: Optional[str] = Field(default=None, description="Specific template to use")
    title: Optional[str] = Field(default=None, description="Custom document title")
    subtitle: Optional[str] = Field(default=None, description="Document subtitle")
    author: Optional[str] = Field(default=None, description="Document author")
    company: Optional[str] = Field(default=None, description="Company name")
    style: Optional[DocumentStyle] = Field(default=None, description="Custom styling")
    additional_requirements: Optional[str] = Field(default=None, description="Additional requirements")
    language: str = Field(default="en", description="Document language")
    tone: Literal["formal", "professional", "casual", "academic", "technical"] = Field(
        default="professional", description="Document tone"
    )
    length: Literal["short", "medium", "long", "comprehensive"] = Field(
        default="medium", description="Document length preference"
    )

    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Query must be at least 10 characters long')
        return v.strip()


class DocumentGenerationResponse(BaseModel):
    """Response from document generation."""
    success: bool = Field(..., description="Whether generation was successful")
    document: Optional[ProfessionalDocument] = Field(default=None, description="Generated document")
    message: str = Field(..., description="Response message")
    generation_time: float = Field(..., description="Time taken to generate in seconds")
    word_count: int = Field(default=0, description="Generated word count")
    estimated_pages: int = Field(default=0, description="Estimated page count")


class DocumentExportRequest(BaseModel):
    """Request to export a document in a specific format."""
    document_id: str = Field(..., description="ID of document to export")
    format: ExportFormat = Field(..., description="Export format")
    include_metadata: bool = Field(default=True, description="Include document metadata")
    custom_filename: Optional[str] = Field(default=None, description="Custom filename")
    watermark: Optional[str] = Field(default=None, description="Custom watermark text")
    password_protect: bool = Field(default=False, description="Password protect PDF")
    password: Optional[str] = Field(default=None, description="Password for protection")


class DocumentExportResponse(BaseModel):
    """Response from document export."""
    success: bool = Field(..., description="Whether export was successful")
    file_path: Optional[str] = Field(default=None, description="Path to exported file")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    download_url: Optional[str] = Field(default=None, description="Download URL")
    message: str = Field(..., description="Response message")
    export_time: float = Field(..., description="Time taken to export in seconds")


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    documents: List[ProfessionalDocument] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of documents per page")


class DocumentUpdateRequest(BaseModel):
    """Request to update an existing document."""
    title: Optional[str] = Field(default=None, description="New title")
    subtitle: Optional[str] = Field(default=None, description="New subtitle")
    sections: Optional[List[DocumentSection]] = Field(default=None, description="Updated sections")
    style: Optional[DocumentStyle] = Field(default=None, description="Updated styling")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata")


class TemplateListResponse(BaseModel):
    """Response for listing templates."""
    templates: List[DocumentTemplate] = Field(..., description="List of templates")
    total_count: int = Field(..., description="Total number of templates")


class DocumentStats(BaseModel):
    """Document statistics."""
    total_documents: int = Field(..., description="Total number of documents")
    documents_by_type: Dict[str, int] = Field(..., description="Documents grouped by type")
    total_word_count: int = Field(..., description="Total word count across all documents")
    average_document_length: float = Field(..., description="Average document length")
    most_used_templates: List[Dict[str, Any]] = Field(..., description="Most used templates")
    export_stats: Dict[str, int] = Field(..., description="Export statistics by format")


# TypedDict for internal processing
class DocumentProcessingContext(TypedDict):
    """Context for document processing."""
    request: DocumentGenerationRequest
    template: DocumentTemplate
    user_id: Optional[str]
    session_id: Optional[str]
    processing_start: datetime
    ai_model: str
    generation_params: Dict[str, Any]




























