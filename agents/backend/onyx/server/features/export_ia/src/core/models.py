"""
Data models for the Export IA system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class ExportFormat(Enum):
    """Supported export formats."""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    RTF = "rtf"
    TXT = "txt"
    JSON = "json"
    XML = "xml"


class DocumentType(Enum):
    """Types of documents that can be exported."""
    BUSINESS_PLAN = "business_plan"
    REPORT = "report"
    PROPOSAL = "proposal"
    PRESENTATION = "presentation"
    MANUAL = "manual"
    CONTRACT = "contract"
    LETTER = "letter"
    MEMO = "memo"
    NEWSLETTER = "newsletter"
    CATALOG = "catalog"


class QualityLevel(Enum):
    """Quality levels for export."""
    BASIC = "basic"
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class TaskStatus(Enum):
    """Status of export tasks."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportConfig:
    """Configuration for export operations."""
    format: ExportFormat
    document_type: DocumentType
    quality_level: QualityLevel = QualityLevel.PROFESSIONAL
    template: Optional[str] = None
    custom_styles: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    branding: Dict[str, Any] = field(default_factory=dict)
    output_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportTask:
    """Represents an export task."""
    id: str
    content: Dict[str, Any]
    config: ExportConfig
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    progress: float = 0.0
    quality_score: float = 0.0


@dataclass
class ExportResult:
    """Result of an export operation."""
    task_id: str
    success: bool
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    format: str = ""
    quality_score: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Quality metrics for exported documents."""
    overall_score: float
    formatting_score: float
    content_score: float
    accessibility_score: float
    professional_score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ExportStatistics:
    """Statistics about export operations."""
    total_tasks: int
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    format_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]
    average_quality_score: float
    average_processing_time: float
    total_processing_time: float




