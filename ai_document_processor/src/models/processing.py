"""
Processing Models - Processing Data Structures
============================================

Pydantic models for document processing operations and results.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ProcessingStage(str, Enum):
    """Processing stage enumeration."""
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    TRANSFORMATION = "transformation"
    FINALIZATION = "finalization"


class ProcessingPriority(str, Enum):
    """Processing priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".md", ".pdf", ".docx", ".doc", ".txt"],
        description="Allowed file extensions"
    )
    max_workers: int = Field(default=8, description="Maximum number of workers")
    chunk_size: int = Field(default=8192, description="Processing chunk size")
    enable_streaming: bool = Field(default=True, description="Enable streaming processing")
    enable_parallel: bool = Field(default=True, description="Enable parallel processing")
    enable_ai_classification: bool = Field(default=True, description="Enable AI classification")
    enable_ai_transformation: bool = Field(default=True, description="Enable AI transformation")
    timeout_seconds: int = Field(default=300, description="Processing timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL, description="Processing priority")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom processing settings")
    
    @validator('max_file_size_mb')
    def validate_max_file_size(cls, v):
        """Validate max file size."""
        if v <= 0:
            raise ValueError("max_file_size_mb must be positive")
        return v
    
    @validator('max_workers')
    def validate_max_workers(cls, v):
        """Validate max workers."""
        if v <= 0:
            raise ValueError("max_workers must be positive")
        return v
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        """Validate chunk size."""
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v


class ProcessingStageInfo(BaseModel):
    """Information about a processing stage."""
    stage: ProcessingStage = Field(..., description="Processing stage")
    status: ProcessingStatus = Field(..., description="Stage status")
    started_at: Optional[datetime] = Field(None, description="Stage start time")
    completed_at: Optional[datetime] = Field(None, description="Stage completion time")
    duration_seconds: Optional[float] = Field(None, description="Stage duration in seconds")
    progress_percentage: float = Field(default=0.0, description="Stage progress percentage")
    error_message: Optional[str] = Field(None, description="Stage error message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Stage metadata")
    
    def start(self):
        """Start the stage."""
        self.status = ProcessingStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
        self.progress_percentage = 0.0
    
    def complete(self, progress: float = 100.0):
        """Complete the stage."""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress_percentage = progress
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def fail(self, error_message: str):
        """Mark stage as failed."""
        self.status = ProcessingStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def update_progress(self, percentage: float):
        """Update stage progress."""
        self.progress_percentage = max(0.0, min(100.0, percentage))


class ProcessingResult(BaseModel):
    """Document processing result."""
    id: str = Field(..., description="Processing result ID")
    document_id: str = Field(..., description="Source document ID")
    status: ProcessingStatus = Field(..., description="Processing status")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    duration_seconds: Optional[float] = Field(None, description="Total processing duration")
    stages: List[ProcessingStageInfo] = Field(default_factory=list, description="Processing stages")
    extracted_text: Optional[str] = Field(None, description="Extracted text content")
    classified_type: Optional[str] = Field(None, description="Classified document type")
    transformed_content: Optional[str] = Field(None, description="Transformed content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    def add_stage(self, stage: ProcessingStageInfo):
        """Add a processing stage."""
        self.stages.append(stage)
    
    def get_stage(self, stage: ProcessingStage) -> Optional[ProcessingStageInfo]:
        """Get a specific processing stage."""
        for s in self.stages:
            if s.stage == stage:
                return s
        return None
    
    def get_current_stage(self) -> Optional[ProcessingStageInfo]:
        """Get the current processing stage."""
        for stage in reversed(self.stages):
            if stage.status == ProcessingStatus.IN_PROGRESS:
                return stage
        return None
    
    def get_completed_stages(self) -> List[ProcessingStageInfo]:
        """Get all completed stages."""
        return [stage for stage in self.stages if stage.status == ProcessingStatus.COMPLETED]
    
    def get_failed_stages(self) -> List[ProcessingStageInfo]:
        """Get all failed stages."""
        return [stage for stage in self.stages if stage.status == ProcessingStatus.FAILED]
    
    def get_overall_progress(self) -> float:
        """Get overall processing progress."""
        if not self.stages:
            return 0.0
        
        total_progress = sum(stage.progress_percentage for stage in self.stages)
        return total_progress / len(self.stages)
    
    def complete(self):
        """Mark processing as completed."""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def fail(self, error_message: str):
        """Mark processing as failed."""
        self.status = ProcessingStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def add_warning(self, warning: str):
        """Add a processing warning."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def add_metric(self, key: str, value: Any):
        """Add a performance metric."""
        self.performance_metrics[key] = value
    
    def get_metric(self, key: str, default: Any = None) -> Any:
        """Get a performance metric."""
        return self.performance_metrics.get(key, default)
    
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.status == ProcessingStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.status == ProcessingStatus.FAILED
    
    def is_in_progress(self) -> bool:
        """Check if processing is in progress."""
        return self.status == ProcessingStatus.IN_PROGRESS
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchProcessingResult(BaseModel):
    """Batch processing result."""
    id: str = Field(..., description="Batch processing ID")
    total_documents: int = Field(..., description="Total number of documents")
    processed_documents: int = Field(default=0, description="Number of processed documents")
    failed_documents: int = Field(default=0, description="Number of failed documents")
    pending_documents: int = Field(default=0, description="Number of pending documents")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Batch start time")
    completed_at: Optional[datetime] = Field(None, description="Batch completion time")
    duration_seconds: Optional[float] = Field(None, description="Total batch duration")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Batch status")
    results: List[ProcessingResult] = Field(default_factory=list, description="Individual results")
    error_message: Optional[str] = Field(None, description="Batch error message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")
    
    def add_result(self, result: ProcessingResult):
        """Add a processing result."""
        self.results.append(result)
        self._update_counts()
    
    def _update_counts(self):
        """Update document counts."""
        self.processed_documents = sum(1 for r in self.results if r.is_completed())
        self.failed_documents = sum(1 for r in self.results if r.is_failed())
        self.pending_documents = self.total_documents - len(self.results)
    
    def get_success_rate(self) -> float:
        """Get success rate percentage."""
        if not self.results:
            return 0.0
        return (self.processed_documents / len(self.results)) * 100
    
    def get_failure_rate(self) -> float:
        """Get failure rate percentage."""
        if not self.results:
            return 0.0
        return (self.failed_documents / len(self.results)) * 100
    
    def complete(self):
        """Mark batch as completed."""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self._update_counts()
    
    def fail(self, error_message: str):
        """Mark batch as failed."""
        self.status = ProcessingStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

















