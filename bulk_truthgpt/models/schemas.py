"""
Pydantic Schemas for Bulk TruthGPT System
=========================================

Data models and validation schemas for the TruthGPT-based document generation system.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid

class OptimizationLevel(str, Enum):
    """Optimization levels for document generation."""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"

class DocumentStatus(str, Enum):
    """Document generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStatus(str, Enum):
    """Task status for bulk generation."""
    CREATED = "created"
    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class GenerationConfig(BaseModel):
    """Configuration for document generation."""
    
    # Basic settings
    max_tokens: int = Field(default=2000, ge=100, le=8000, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    model: str = Field(default="gpt-3.5-turbo", min_length=1, max_length=128, description="LLM model to use")
    
    # Generation settings
    max_documents: int = Field(default=10, ge=1, le=1000, description="Maximum documents to generate")
    estimated_duration: Optional[int] = Field(default=None, ge=1, description="Estimated duration in minutes")
    
    # Optimization settings
    @validator('model')
    def validate_model(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()
    
    @validator('max_documents')
    def validate_max_documents(cls, v):
        if v > 1000:
            raise ValueError("max_documents cannot exceed 1000 for resource constraints")
        return v
    optimization_level: OptimizationLevel = Field(default=OptimizationLevel.ADVANCED, description="Optimization level")
    force_regeneration: bool = Field(default=False, description="Force regeneration even if cached")
    
    # Output settings
    output_format: Optional[str] = Field(default=None, description="Output format (html, markdown, pdf, etc.)")
    template: Optional[str] = Field(default=None, description="Template to use for generation")
    
    # Quality settings
    min_quality_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum quality score")
    enable_validation: bool = Field(default=True, description="Enable content validation")
    
    # Advanced settings
    enable_learning: bool = Field(default=True, description="Enable learning from previous generations")
    enable_optimization: bool = Field(default=True, description="Enable content optimization")
    enable_analysis: bool = Field(default=True, description="Enable content analysis")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v < 100 or v > 8000:
            raise ValueError('Max tokens must be between 100 and 8000')
        return v

class TruthGPTConfig(BaseModel):
    """Configuration for TruthGPT engine."""
    
    # Knowledge base settings
    max_knowledge_entries: int = Field(default=10, ge=1, le=100, description="Maximum knowledge entries to use")
    knowledge_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Knowledge similarity threshold")
    
    # Optimization settings
    enable_prompt_optimization: bool = Field(default=True, description="Enable prompt optimization")
    enable_content_analysis: bool = Field(default=True, description="Enable content analysis")
    enable_truthfulness_check: bool = Field(default=True, description="Enable truthfulness checking")
    
    # Learning settings
    enable_continuous_learning: bool = Field(default=True, description="Enable continuous learning")
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Learning rate for optimization")
    
    # Performance settings
    max_concurrent_generations: int = Field(default=5, ge=1, le=20, description="Maximum concurrent generations")
    cache_ttl: int = Field(default=3600, ge=60, le=86400, description="Cache TTL in seconds")
    
    @validator('knowledge_threshold')
    def validate_knowledge_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Knowledge threshold must be between 0.0 and 1.0')
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Learning rate must be between 0.0 and 1.0')
        return v

class BulkGenerationRequest(BaseModel):
    """Request for bulk document generation."""
    
    # Required fields
    query: str = Field(..., min_length=1, max_length=1000, description="Generation query")
    config: GenerationConfig = Field(default_factory=GenerationConfig, description="Generation configuration")
    
    # Optional fields
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    priority: int = Field(default=5, ge=1, le=10, description="Generation priority (1=low, 10=high)")
    
    # Advanced settings
    enable_parallel_generation: bool = Field(default=True, description="Enable parallel generation")
    max_parallel_tasks: int = Field(default=3, ge=1, le=10, description="Maximum parallel tasks")
    
    # Monitoring settings
    enable_progress_tracking: bool = Field(default=True, description="Enable progress tracking")
    progress_callback_url: Optional[str] = Field(default=None, description="Progress callback URL")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('priority')
    def validate_priority(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Priority must be between 1 and 10')
        return v

class BulkGenerationResponse(BaseModel):
    """Response for bulk document generation."""
    
    # Task information
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    message: str = Field(..., description="Status message")
    
    # Generation information
    estimated_documents: int = Field(..., description="Estimated number of documents")
    estimated_duration: Optional[int] = Field(default=None, description="Estimated duration in minutes")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    # Progress information
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    documents_generated: int = Field(default=0, ge=0, description="Number of documents generated")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DocumentMetadata(BaseModel):
    """Metadata for generated documents."""
    
    # Document information
    document_id: str = Field(..., description="Unique document identifier")
    task_id: str = Field(..., description="Associated task identifier")
    
    # Content information
    title: Optional[str] = Field(default=None, description="Document title")
    content_type: str = Field(default="text", description="Content type")
    format: str = Field(default="markdown", description="Document format")
    
    # Quality metrics
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality score")
    readability_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Readability score")
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Coherence score")
    
    # Generation information
    generation_time: float = Field(default=0.0, ge=0.0, description="Generation time in seconds")
    tokens_used: int = Field(default=0, ge=0, description="Number of tokens used")
    model_used: str = Field(default="", description="Model used for generation")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    
    # Optimization information
    optimization_applied: List[str] = Field(default_factory=list, description="Optimizations applied")
    optimization_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Optimization score")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class GenerationMetrics(BaseModel):
    """Metrics for generation system."""
    
    # System metrics
    total_generations: int = Field(default=0, ge=0, description="Total generations")
    successful_generations: int = Field(default=0, ge=0, description="Successful generations")
    failed_generations: int = Field(default=0, ge=0, description="Failed generations")
    
    # Quality metrics
    average_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average quality score")
    average_generation_time: float = Field(default=0.0, ge=0.0, description="Average generation time")
    average_optimization_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average optimization score")
    
    # Performance metrics
    total_tokens_used: int = Field(default=0, ge=0, description="Total tokens used")
    average_tokens_per_document: float = Field(default=0.0, ge=0.0, description="Average tokens per document")
    
    # System health
    system_uptime: float = Field(default=0.0, ge=0.0, description="System uptime in seconds")
    active_tasks: int = Field(default=0, ge=0, description="Number of active tasks")
    queue_size: int = Field(default=0, ge=0, description="Queue size")
    
    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TaskInfo(BaseModel):
    """Information about a generation task."""
    
    # Task identification
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    
    # Task details
    query: str = Field(..., description="Generation query")
    config: GenerationConfig = Field(..., description="Generation configuration")
    
    # Progress information
    total_documents: int = Field(default=0, ge=0, description="Total documents to generate")
    completed_documents: int = Field(default=0, ge=0, description="Completed documents")
    failed_documents: int = Field(default=0, ge=0, description="Failed documents")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    # Performance metrics
    average_generation_time: float = Field(default=0.0, ge=0.0, description="Average generation time")
    total_generation_time: float = Field(default=0.0, ge=0.0, description="Total generation time")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="List of errors")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SystemStatus(BaseModel):
    """System status information."""
    
    # System health
    status: str = Field(..., description="System status")
    health_score: float = Field(default=0.0, ge=0.0, le=1.0, description="System health score")
    
    # Component status
    components: Dict[str, str] = Field(default_factory=dict, description="Component status")
    
    # Performance metrics
    cpu_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="Memory usage percentage")
    disk_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="Disk usage percentage")
    
    # Generation metrics
    active_tasks: int = Field(default=0, ge=0, description="Active tasks")
    completed_tasks: int = Field(default=0, ge=0, description="Completed tasks")
    failed_tasks: int = Field(default=0, ge=0, description="Failed tasks")
    
    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SuccessResponse(BaseModel):
    """Success response model."""
    
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }










