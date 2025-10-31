"""
API models for the Export IA system.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from ..core.models import ExportFormat, DocumentType, QualityLevel


class ExportRequest(BaseModel):
    """Request model for export operations."""
    content: Dict[str, Any] = Field(..., description="Document content to export")
    format: ExportFormat = Field(..., description="Export format")
    document_type: DocumentType = Field(default=DocumentType.REPORT, description="Type of document")
    quality_level: QualityLevel = Field(default=QualityLevel.PROFESSIONAL, description="Quality level")
    template: Optional[str] = Field(None, description="Custom template name")
    custom_styles: Dict[str, Any] = Field(default_factory=dict, description="Custom styling options")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    branding: Dict[str, Any] = Field(default_factory=dict, description="Branding options")
    output_options: Dict[str, Any] = Field(default_factory=dict, description="Output-specific options")


class ExportResponse(BaseModel):
    """Response model for export operations."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")
    created_at: datetime = Field(..., description="Task creation timestamp")


class TaskStatusResponse(BaseModel):
    """Response model for task status queries."""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Current task status")
    progress: float = Field(..., description="Task progress (0.0 to 1.0)")
    created_at: Optional[datetime] = Field(None, description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    file_path: Optional[str] = Field(None, description="Output file path")
    file_size: Optional[int] = Field(None, description="Output file size in bytes")
    quality_score: Optional[float] = Field(None, description="Quality score (0.0 to 1.0)")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


class StatisticsResponse(BaseModel):
    """Response model for system statistics."""
    total_tasks: int = Field(..., description="Total number of tasks")
    active_tasks: int = Field(..., description="Number of active tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    format_distribution: Dict[str, int] = Field(..., description="Distribution of export formats")
    quality_distribution: Dict[str, int] = Field(..., description="Distribution of quality levels")
    average_quality_score: float = Field(..., description="Average quality score")
    average_processing_time: float = Field(..., description="Average processing time")
    total_processing_time: float = Field(..., description="Total processing time")


class FormatInfo(BaseModel):
    """Information about a supported export format."""
    format: str = Field(..., description="Format identifier")
    name: str = Field(..., description="Format display name")
    description: str = Field(..., description="Format description")
    professional_features: List[str] = Field(..., description="List of professional features")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")




