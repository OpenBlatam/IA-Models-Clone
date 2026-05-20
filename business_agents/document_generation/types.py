"""
Document Types and Enums
========================

Type definitions for document generation system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

class DocumentType(Enum):
    """Document type enumeration"""
    REPORT = "report"
    PROPOSAL = "proposal"
    MANUAL = "manual"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    DASHBOARD = "dashboard"
    CONTRACT = "contract"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    USER_GUIDE = "user_guide"
    API_DOCUMENTATION = "api_documentation"
    TECHNICAL_SPEC = "technical_spec"
    BUSINESS_PLAN = "business_plan"
    MARKETING_MATERIAL = "marketing_material"

class DocumentFormat(Enum):
    """Document format enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    TXT = "txt"

class TemplateType(Enum):
    """Template type enumeration"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    AI_GENERATED = "ai_generated"
    CUSTOM = "custom"

class DocumentStatus(Enum):
    """Document generation status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class DocumentTemplate:
    """Document template definition"""
    template_id: str
    name: str
    description: str
    document_type: DocumentType
    template_type: TemplateType
    format: DocumentFormat
    content: str
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True

@dataclass
class DocumentRequest:
    """Document generation request"""
    request_id: str
    document_type: DocumentType
    title: str
    description: str
    business_area: str
    variables: Dict[str, Any] = field(default_factory=dict)
    template_id: Optional[str] = None
    format: DocumentFormat = DocumentFormat.MARKDOWN
    priority: str = "normal"
    deadline: Optional[datetime] = None
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class GeneratedDocument:
    """Generated document result"""
    document_id: str
    request_id: str
    title: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    format: DocumentFormat = DocumentFormat.MARKDOWN
    size_bytes: Optional[int] = None
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentGenerationResult:
    """Document generation result"""
    success: bool
    document: Optional[GeneratedDocument] = None
    error: Optional[str] = None
    generation_time: Optional[float] = None
    warnings: list = field(default_factory=list)
