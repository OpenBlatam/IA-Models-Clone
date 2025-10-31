"""
Document Models - Document Data Structures
=========================================

Pydantic models for document representation and metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Document type enumeration."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"
    HTML = "html"
    XML = "xml"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentMetadata(BaseModel):
    """Document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    language: Optional[str] = None
    encoding: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Document(BaseModel):
    """Main document model."""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_path: Optional[str] = Field(None, description="File system path")
    content: Optional[str] = Field(None, description="Document content")
    content_bytes: Optional[bytes] = Field(None, description="Raw document bytes")
    document_type: DocumentType = Field(..., description="Document type")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    processing_config: Optional[Dict[str, Any]] = Field(None, description="Processing configuration")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    @validator('document_type', pre=True)
    def validate_document_type(cls, v):
        """Validate document type."""
        if isinstance(v, str):
            try:
                return DocumentType(v.lower())
            except ValueError:
                return DocumentType.UNKNOWN
        return v
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path."""
        if v and not Path(v).exists():
            # Don't raise error, just log warning
            pass
        return v
    
    def get_file_extension(self) -> str:
        """Get file extension."""
        return Path(self.filename).suffix.lower()
    
    def get_file_size(self) -> Optional[int]:
        """Get file size in bytes."""
        if self.metadata.file_size:
            return self.metadata.file_size
        if self.file_path and Path(self.file_path).exists():
            return Path(self.file_path).stat().st_size
        if self.content_bytes:
            return len(self.content_bytes)
        return None
    
    def get_content_length(self) -> int:
        """Get content length."""
        if self.content:
            return len(self.content)
        if self.content_bytes:
            return len(self.content_bytes)
        return 0
    
    def is_processed(self) -> bool:
        """Check if document is processed."""
        return self.status == DocumentStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if document processing failed."""
        return self.status == DocumentStatus.FAILED
    
    def is_processing(self) -> bool:
        """Check if document is being processed."""
        return self.status == DocumentStatus.PROCESSING
    
    def update_status(self, status: DocumentStatus, error_message: Optional[str] = None):
        """Update document status."""
        self.status = status
        self.updated_at = datetime.utcnow()
        if error_message:
            self.error_message = error_message
    
    def add_tag(self, tag: str):
        """Add a tag to the document."""
        if tag not in self.metadata.tags:
            self.metadata.tags.append(tag)
    
    def remove_tag(self, tag: str):
        """Remove a tag from the document."""
        if tag in self.metadata.tags:
            self.metadata.tags.remove(tag)
    
    def set_custom_field(self, key: str, value: Any):
        """Set a custom field."""
        self.metadata.custom_fields[key] = value
    
    def get_custom_field(self, key: str, default: Any = None) -> Any:
        """Get a custom field."""
        return self.metadata.custom_fields.get(key, default)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            bytes: lambda v: v.decode('utf-8', errors='ignore') if v else None
        }
        validate_assignment = True
        use_enum_values = True


class DocumentCollection(BaseModel):
    """Collection of documents."""
    documents: List[Document] = Field(default_factory=list, description="List of documents")
    total_count: int = Field(default=0, description="Total number of documents")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Collection creation time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Collection metadata")
    
    def add_document(self, document: Document):
        """Add a document to the collection."""
        self.documents.append(document)
        self.total_count = len(self.documents)
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the collection."""
        for i, doc in enumerate(self.documents):
            if doc.id == document_id:
                del self.documents[i]
                self.total_count = len(self.documents)
                return True
        return False
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.id == document_id:
                return doc
        return None
    
    def get_documents_by_type(self, document_type: DocumentType) -> List[Document]:
        """Get documents by type."""
        return [doc for doc in self.documents if doc.document_type == document_type]
    
    def get_documents_by_status(self, status: DocumentStatus) -> List[Document]:
        """Get documents by status."""
        return [doc for doc in self.documents if doc.status == status]
    
    def get_processed_documents(self) -> List[Document]:
        """Get all processed documents."""
        return self.get_documents_by_status(DocumentStatus.COMPLETED)
    
    def get_failed_documents(self) -> List[Document]:
        """Get all failed documents."""
        return self.get_documents_by_status(DocumentStatus.FAILED)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

















