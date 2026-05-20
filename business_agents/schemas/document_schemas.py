"""
Document Schemas
================

Pydantic models for document operations.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..document_generator import DocumentType, DocumentFormat

class DocumentRequestModel(BaseModel):
    """Request schema for document generation."""
    
    document_type: DocumentType = Field(..., description="Document type")
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    description: str = Field(..., min_length=1, max_length=1000, description="Document description")
    business_area: str = Field(..., description="Business area")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Document variables")
    template_id: Optional[str] = Field(None, description="Template ID to use")
    format: DocumentFormat = Field(DocumentFormat.MARKDOWN, description="Document format")
    priority: str = Field("normal", regex="^(low|normal|high|urgent)$", description="Generation priority")
    deadline: Optional[datetime] = Field(None, description="Generation deadline")

class DocumentResponse(BaseModel):
    """Response schema for document."""
    
    id: str = Field(..., description="Document ID")
    request_id: str = Field(..., description="Request ID")
    title: str = Field(..., description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    format: str = Field(..., description="Document format")
    file_path: Optional[str] = Field(None, description="File path")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    created_at: datetime = Field(..., description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DocumentListResponse(BaseModel):
    """Response schema for document list."""
    
    documents: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    business_area: Optional[str] = Field(None, description="Filtered business area")
    document_type: Optional[str] = Field(None, description="Filtered document type")

class DocumentGenerationResponse(BaseModel):
    """Response schema for document generation."""
    
    document_id: str = Field(..., description="Generated document ID")
    request_id: str = Field(..., description="Request ID")
    title: str = Field(..., description="Document title")
    file_path: str = Field(..., description="Generated file path")
    format: str = Field(..., description="Document format")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(..., description="Generation timestamp")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")

class DocumentTemplateResponse(BaseModel):
    """Response schema for document template."""
    
    id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    document_type: str = Field(..., description="Document type")
    business_area: str = Field(..., description="Business area")
    template_data: Dict[str, Any] = Field(..., description="Template data")
    category: Optional[str] = Field(None, description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    is_public: bool = Field(..., description="Whether template is public")
    usage_count: int = Field(..., description="Usage count")
    rating: float = Field(..., description="Template rating")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

class DocumentDownloadResponse(BaseModel):
    """Response schema for document download."""
    
    file_path: str = Field(..., description="File path")
    filename: str = Field(..., description="Download filename")
    content_type: str = Field(..., description="Content type")
    size_bytes: int = Field(..., description="File size in bytes")
    download_url: Optional[str] = Field(None, description="Download URL if available")
