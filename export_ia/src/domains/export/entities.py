"""
Export domain entities - Core business objects.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .value_objects import ExportFormat, DocumentType, QualityLevel, ExportConfig


class TaskStatus(Enum):
    """Export task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportRequest:
    """Export request entity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    config: ExportConfig = field(default_factory=ExportConfig)
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate export request."""
        errors = []
        
        if not self.content:
            errors.append("Content is required")
        
        if not self.config.format:
            errors.append("Export format is required")
        
        if not self.config.document_type:
            errors.append("Document type is required")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if request is valid."""
        return len(self.validate()) == 0


@dataclass
class ExportTask:
    """Export task entity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request: ExportRequest
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional['ExportResult'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> None:
        """Start the export task."""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot start task in status: {self.status}")
        
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.now()
        self.progress = 0.0
    
    def complete(self, result: 'ExportResult') -> None:
        """Complete the export task."""
        if self.status != TaskStatus.PROCESSING:
            raise ValueError(f"Cannot complete task in status: {self.status}")
        
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 100.0
        self.result = result
    
    def fail(self, error_message: str) -> None:
        """Mark task as failed."""
        if self.status not in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
            raise ValueError(f"Cannot fail task in status: {self.status}")
        
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
    
    def cancel(self) -> None:
        """Cancel the export task."""
        if self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            raise ValueError(f"Cannot cancel task in status: {self.status}")
        
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
    
    def update_progress(self, progress: float) -> None:
        """Update task progress."""
        if not 0 <= progress <= 100:
            raise ValueError("Progress must be between 0 and 100")
        
        self.progress = progress
    
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.PROCESSING
    
    def is_finished(self) -> bool:
        """Check if task is finished (completed, failed, or cancelled)."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]


@dataclass
class ExportResult:
    """Export result entity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    success: bool
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    format: ExportFormat = ExportFormat.PDF
    quality_score: Optional[float] = None
    processing_time: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def get_file_extension(self) -> str:
        """Get file extension based on format."""
        extensions = {
            ExportFormat.PDF: ".pdf",
            ExportFormat.DOCX: ".docx",
            ExportFormat.HTML: ".html",
            ExportFormat.MARKDOWN: ".md",
            ExportFormat.RTF: ".rtf",
            ExportFormat.TXT: ".txt",
            ExportFormat.JSON: ".json",
            ExportFormat.XML: ".xml"
        }
        return extensions.get(self.format, ".pdf")
    
    def get_mime_type(self) -> str:
        """Get MIME type based on format."""
        mime_types = {
            ExportFormat.PDF: "application/pdf",
            ExportFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ExportFormat.HTML: "text/html",
            ExportFormat.MARKDOWN: "text/markdown",
            ExportFormat.RTF: "application/rtf",
            ExportFormat.TXT: "text/plain",
            ExportFormat.JSON: "application/json",
            ExportFormat.XML: "application/xml"
        }
        return mime_types.get(self.format, "application/pdf")


@dataclass
class ExportTemplate:
    """Export template entity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    format: ExportFormat
    document_type: DocumentType
    template_data: Dict[str, Any] = field(default_factory=dict)
    styles: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    
    def update_template(self, template_data: Dict[str, Any]) -> None:
        """Update template data."""
        self.template_data = template_data
        self.updated_at = datetime.now()
    
    def update_styles(self, styles: Dict[str, Any]) -> None:
        """Update template styles."""
        self.styles = styles
        self.updated_at = datetime.now()
    
    def activate(self) -> None:
        """Activate the template."""
        self.is_active = True
        self.updated_at = datetime.now()
    
    def deactivate(self) -> None:
        """Deactivate the template."""
        self.is_active = False
        self.updated_at = datetime.now()


@dataclass
class ExportBatch:
    """Export batch entity for bulk operations."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    requests: List[ExportRequest] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[ExportResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    
    def add_request(self, request: ExportRequest) -> None:
        """Add a request to the batch."""
        self.requests.append(request)
    
    def start(self) -> None:
        """Start batch processing."""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot start batch in status: {self.status}")
        
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.now()
        self.progress = 0.0
    
    def complete(self) -> None:
        """Complete batch processing."""
        if self.status != TaskStatus.PROCESSING:
            raise ValueError(f"Cannot complete batch in status: {self.status}")
        
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 100.0
    
    def fail(self) -> None:
        """Mark batch as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
    
    def update_progress(self, progress: float) -> None:
        """Update batch progress."""
        if not 0 <= progress <= 100:
            raise ValueError("Progress must be between 0 and 100")
        
        self.progress = progress
    
    def get_success_rate(self) -> float:
        """Get success rate of completed requests."""
        if not self.results:
            return 0.0
        
        successful = sum(1 for result in self.results if result.success)
        return (successful / len(self.results)) * 100




