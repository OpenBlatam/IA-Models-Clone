from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import List, Optional, Dict, Any, Callable, Awaitable, TypeVar, Generic
from functools import wraps, partial
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from fastapi import (
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator
from pydantic.types import conint, constr
import structlog
from .sqlalchemy_2_implementation import (
    import hashlib
from typing import Any, List, Dict, Optional
"""
🚀 Functional FastAPI Components with Pydantic Models
====================================================

Functional approach to FastAPI using:
- Pure functions for business logic
- Pydantic models for validation and serialization
- Immutable data structures
- Declarative patterns
- Composition over inheritance
- Type-safe operations
"""


    FastAPI, Depends, HTTPException, status, Request, Response,
    BackgroundTasks, Query, Path, Body, Header
)

    DatabaseConfig, SQLAlchemy2Manager,
    TextAnalysisCreate, TextAnalysisUpdate, BatchAnalysisCreate,
    TextAnalysisResponse, BatchAnalysisResponse,
    AnalysisType, AnalysisStatus, OptimizationTier,
    DatabaseError, ValidationError
)

# ============================================================================
# Type Definitions and Generic Types
# ============================================================================

T = TypeVar('T')
R = TypeVar('R')

@dataclass(frozen=True)
class RequestContext:
    """Immutable request context."""
    request_id: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class ProcessingResult(Generic[T]):
    """Immutable processing result."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Pydantic Models for Input Validation
# ============================================================================

class AnalysisTypeEnum(str, Enum):
    """Analysis type enumeration."""
    SENTIMENT = "sentiment"
    QUALITY = "quality"
    EMOTION = "emotion"
    LANGUAGE = "language"
    KEYWORDS = "keywords"
    READABILITY = "readability"
    ENTITIES = "entities"
    TOPICS = "topics"

class OptimizationTierEnum(str, Enum):
    """Optimization tier enumeration."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ULTRA = "ultra"

class AnalysisStatusEnum(str, Enum):
    """Analysis status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CACHED = "cached"
    ERROR = "error"

class TextAnalysisRequest(BaseModel):
    """Input model for text analysis requests."""
    text_content: constr(min_length=1, max_length=10000) = Field(
        ..., 
        description="Text content to analyze",
        example="This is a sample text for sentiment analysis."
    )
    analysis_type: AnalysisTypeEnum = Field(
        ..., 
        description="Type of analysis to perform"
    )
    optimization_tier: OptimizationTierEnum = Field(
        default=OptimizationTierEnum.STANDARD,
        description="Performance optimization tier"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text_content": "This is a positive text for sentiment analysis.",
                "analysis_type": "sentiment",
                "optimization_tier": "standard",
                "metadata": {"source": "user_input", "priority": "high"}
            }
        }
    )
    
    @validator('text_content')
    def validate_text_content(cls, v: str) -> str:
        """Validate and clean text content."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Text content cannot be empty or whitespace only')
        return cleaned
    
    @root_validator
    def validate_metadata(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata constraints."""
        metadata = values.get('metadata', {})
        
        # Check metadata size
        if len(str(metadata)) > 1000:
            raise ValueError('Metadata too large (max 1000 characters)')
        
        # Validate metadata keys
        for key in metadata.keys():
            if not isinstance(key, str) or len(key) > 50:
                raise ValueError('Invalid metadata key')
        
        return values

class BatchAnalysisRequest(BaseModel):
    """Input model for batch analysis requests."""
    batch_name: constr(min_length=1, max_length=200) = Field(
        ..., 
        description="Name for the batch analysis"
    )
    texts: List[constr(min_length=1, max_length=10000)] = Field(
        ..., 
        min_items=1, 
        max_items=1000,
        description="List of texts to analyze"
    )
    analysis_type: AnalysisTypeEnum = Field(
        ..., 
        description="Type of analysis to perform"
    )
    optimization_tier: OptimizationTierEnum = Field(
        default=OptimizationTierEnum.STANDARD,
        description="Performance optimization tier"
    )
    priority: conint(ge=1, le=10) = Field(
        default=5,
        description="Processing priority (1-10)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "batch_name": "Customer Feedback Analysis",
                "texts": [
                    "Great product, highly recommended!",
                    "Disappointed with the service quality.",
                    "Average experience, could be better."
                ],
                "analysis_type": "sentiment",
                "optimization_tier": "standard",
                "priority": 7
            }
        }
    )
    
    @validator('texts')
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate and clean text list."""
        if not v:
            raise ValueError('At least one text must be provided')
        
        cleaned_texts = []
        for text in v:
            cleaned = text.strip()
            if not cleaned:
                raise ValueError('Text content cannot be empty or whitespace only')
            cleaned_texts.append(cleaned)
        
        return cleaned_texts
    
    @validator('batch_name')
    def validate_batch_name(cls, v: str) -> str:
        """Validate batch name."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Batch name cannot be empty')
        return cleaned

class AnalysisUpdateRequest(BaseModel):
    """Input model for analysis updates."""
    status: Optional[AnalysisStatusEnum] = Field(
        None, 
        description="Analysis status"
    )
    sentiment_score: Optional[float] = Field(
        None, 
        ge=-1.0, 
        le=1.0,
        description="Sentiment score (-1.0 to 1.0)"
    )
    quality_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Quality score (0.0 to 1.0)"
    )
    processing_time_ms: Optional[float] = Field(
        None, 
        ge=0.0,
        description="Processing time in milliseconds"
    )
    model_used: Optional[constr(max_length=100)] = Field(
        None,
        description="Model used for analysis"
    )
    confidence_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )
    error_message: Optional[constr(max_length=1000)] = Field(
        None,
        description="Error message if analysis failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "completed",
                "sentiment_score": 0.8,
                "quality_score": 0.95,
                "processing_time_ms": 150.5,
                "model_used": "distilbert-sentiment",
                "confidence_score": 0.92,
                "metadata": {"model_version": "1.2.0", "batch_id": "123"}
            }
        }
    )
    
    @root_validator
    def validate_status_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate status consistency with other fields."""
        status = values.get('status')
        error_message = values.get('error_message')
        sentiment_score = values.get('sentiment_score')
        
        if status == AnalysisStatusEnum.ERROR and not error_message:
            raise ValueError('Error message required when status is error')
        
        if status == AnalysisStatusEnum.COMPLETED and sentiment_score is None:
            raise ValueError('Sentiment score required when status is completed')
        
        return values

class PaginationRequest(BaseModel):
    """Input model for pagination parameters."""
    page: conint(ge=1) = Field(
        default=1,
        description="Page number (1-based)"
    )
    size: conint(ge=1, le=100) = Field(
        default=20,
        description="Page size (1-100)"
    )
    order_by: str = Field(
        default="created_at",
        description="Field to order by"
    )
    order_desc: bool = Field(
        default=True,
        description="Descending order"
    )
    
    @property
    def offset(self) -> int:
        """Calculate offset for pagination."""
        return (self.page - 1) * self.size

class AnalysisFilterRequest(BaseModel):
    """Input model for analysis filtering."""
    analysis_type: Optional[AnalysisTypeEnum] = Field(
        None,
        description="Filter by analysis type"
    )
    status: Optional[AnalysisStatusEnum] = Field(
        None,
        description="Filter by status"
    )
    optimization_tier: Optional[OptimizationTierEnum] = Field(
        None,
        description="Filter by optimization tier"
    )
    date_from: Optional[datetime] = Field(
        None,
        description="Filter from date"
    )
    date_to: Optional[datetime] = Field(
        None,
        description="Filter to date"
    )
    min_sentiment_score: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Minimum sentiment score"
    )
    max_sentiment_score: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Maximum sentiment score"
    )
    
    @root_validator
    def validate_date_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate date range consistency."""
        date_from = values.get('date_from')
        date_to = values.get('date_to')
        
        if date_from and date_to and date_from > date_to:
            raise ValueError('date_from must be before date_to')
        
        return values
    
    @root_validator
    def validate_sentiment_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sentiment score range consistency."""
        min_score = values.get('min_sentiment_score')
        max_score = values.get('max_sentiment_score')
        
        if min_score is not None and max_score is not None and min_score > max_score:
            raise ValueError('min_sentiment_score must be less than max_sentiment_score')
        
        return values

# ============================================================================
# Pydantic Models for Response Schemas
# ============================================================================

class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    id: int = Field(description="Analysis ID")
    text_content: str = Field(description="Analyzed text content")
    analysis_type: AnalysisTypeEnum = Field(description="Analysis type")
    status: AnalysisStatusEnum = Field(description="Analysis status")
    sentiment_score: Optional[float] = Field(description="Sentiment score")
    quality_score: Optional[float] = Field(description="Quality score")
    processing_time_ms: Optional[float] = Field(description="Processing time")
    model_used: Optional[str] = Field(description="Model used")
    confidence_score: Optional[float] = Field(description="Confidence score")
    optimization_tier: OptimizationTierEnum = Field(description="Optimization tier")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    processed_at: Optional[datetime] = Field(description="Processing completion timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @dataclass
class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "text_content": "This is a positive text for analysis.",
                "analysis_type": "sentiment",
                "status": "completed",
                "sentiment_score": 0.8,
                "quality_score": 0.95,
                "processing_time_ms": 150.5,
                "model_used": "distilbert-sentiment",
                "confidence_score": 0.92,
                "optimization_tier": "standard",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:05Z",
                "processed_at": "2024-01-15T10:30:05Z",
                "metadata": {"source": "user_input", "priority": "high"}
            }
        }

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    id: int = Field(description="Batch ID")
    batch_name: str = Field(description="Batch name")
    analysis_type: AnalysisTypeEnum = Field(description="Analysis type")
    status: AnalysisStatusEnum = Field(description="Batch status")
    total_texts: int = Field(description="Total number of texts")
    completed_count: int = Field(description="Completed analyses count")
    error_count: int = Field(description="Error count")
    progress_percentage: float = Field(description="Progress percentage")
    optimization_tier: OptimizationTierEnum = Field(description="Optimization tier")
    priority: int = Field(description="Processing priority")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    completed_at: Optional[datetime] = Field(description="Completion timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @dataclass
class Config:
        from_attributes = True
    
    @property
    def is_completed(self) -> bool:
        """Check if batch is completed."""
        return self.status == AnalysisStatusEnum.COMPLETED
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_texts == 0:
            return 0.0
        return (self.completed_count / self.total_texts) * 100

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    items: List[T] = Field(description="List of items")
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    size: int = Field(description="Page size")
    pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Has next page")
    has_prev: bool = Field(description="Has previous page")
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        size: int
    ) -> 'PaginatedResponse[T]':
        """Create paginated response."""
        pages = (total + size - 1) // size  # Ceiling division
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )

class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(description="Service status")
    timestamp: datetime = Field(description="Check timestamp")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    database: Dict[str, Any] = Field(description="Database health status")
    performance: Dict[str, Any] = Field(description="Performance metrics")
    errors: List[str] = Field(default_factory=list, description="Error messages")

class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(description="Error code")
    detail: Optional[str] = Field(description="Detailed error information")
    timestamp: datetime = Field(description="Error timestamp")
    request_id: Optional[str] = Field(description="Request ID for tracking")
    path: Optional[str] = Field(description="Request path")

class SuccessResponse(BaseModel):
    """Response model for successful operations."""
    message: str = Field(description="Success message")
    data: Optional[Dict[str, Any]] = Field(description="Response data")
    timestamp: datetime = Field(description="Response timestamp")
    request_id: Optional[str] = Field(description="Request ID")

# ============================================================================
# Pure Functions for Business Logic
# ============================================================================

def validate_text_content(text: str) -> ValidationResult:
    """Pure function to validate text content."""
    errors = []
    warnings = []
    
    if not text:
        errors.append("Text content cannot be empty")
        return ValidationResult(is_valid=False, errors=errors)
    
    cleaned = text.strip()
    if not cleaned:
        errors.append("Text content cannot be whitespace only")
        return ValidationResult(is_valid=False, errors=errors)
    
    if len(cleaned) > 10000:
        errors.append("Text content too long (max 10000 characters)")
        return ValidationResult(is_valid=False, errors=errors)
    
    if len(cleaned) < 10:
        warnings.append("Text content is very short, analysis may be less accurate")
    
    return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

def calculate_processing_priority(
    optimization_tier: OptimizationTierEnum,
    text_length: int,
    analysis_type: AnalysisTypeEnum
) -> int:
    """Pure function to calculate processing priority."""
    base_priority = 5
    
    # Adjust based on optimization tier
    tier_multipliers = {
        OptimizationTierEnum.BASIC: 0.8,
        OptimizationTierEnum.STANDARD: 1.0,
        OptimizationTierEnum.ADVANCED: 1.2,
        OptimizationTierEnum.ULTRA: 1.5
    }
    
    # Adjust based on text length
    length_factor = min(text_length / 1000, 2.0)
    
    # Adjust based on analysis type complexity
    complexity_factors = {
        AnalysisTypeEnum.SENTIMENT: 1.0,
        AnalysisTypeEnum.QUALITY: 1.2,
        AnalysisTypeEnum.EMOTION: 1.3,
        AnalysisTypeEnum.LANGUAGE: 0.8,
        AnalysisTypeEnum.KEYWORDS: 1.1,
        AnalysisTypeEnum.READABILITY: 1.0,
        AnalysisTypeEnum.ENTITIES: 1.4,
        AnalysisTypeEnum.TOPICS: 1.5
    }
    
    priority = int(
        base_priority * 
        tier_multipliers[optimization_tier] * 
        length_factor * 
        complexity_factors[analysis_type]
    )
    
    return max(1, min(10, priority))

def estimate_processing_time(
    text_length: int,
    analysis_type: AnalysisTypeEnum,
    optimization_tier: OptimizationTierEnum
) -> float:
    """Pure function to estimate processing time."""
    base_time_ms = 100.0
    
    # Adjust based on text length
    length_factor = text_length / 1000
    
    # Adjust based on analysis type
    type_factors = {
        AnalysisTypeEnum.SENTIMENT: 1.0,
        AnalysisTypeEnum.QUALITY: 1.5,
        AnalysisTypeEnum.EMOTION: 1.8,
        AnalysisTypeEnum.LANGUAGE: 0.5,
        AnalysisTypeEnum.KEYWORDS: 1.2,
        AnalysisTypeEnum.READABILITY: 1.3,
        AnalysisTypeEnum.ENTITIES: 2.0,
        AnalysisTypeEnum.TOPICS: 2.5
    }
    
    # Adjust based on optimization tier
    tier_factors = {
        OptimizationTierEnum.BASIC: 1.5,
        OptimizationTierEnum.STANDARD: 1.0,
        OptimizationTierEnum.ADVANCED: 0.8,
        OptimizationTierEnum.ULTRA: 0.6
    }
    
    estimated_time = (
        base_time_ms * 
        length_factor * 
        type_factors[analysis_type] * 
        tier_factors[optimization_tier]
    )
    
    return max(50.0, estimated_time)

def calculate_batch_progress(
    completed_count: int,
    error_count: int,
    total_count: int
) -> Dict[str, Any]:
    """Pure function to calculate batch progress."""
    if total_count == 0:
        return {
            "progress_percentage": 0.0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "remaining_count": 0
        }
    
    progress_percentage = ((completed_count + error_count) / total_count) * 100
    success_rate = (completed_count / total_count) * 100
    error_rate = (error_count / total_count) * 100
    remaining_count = total_count - completed_count - error_count
    
    return {
        "progress_percentage": round(progress_percentage, 2),
        "success_rate": round(success_rate, 2),
        "error_rate": round(error_rate, 2),
        "remaining_count": remaining_count
    }

def generate_cache_key(
    text_content: str,
    analysis_type: AnalysisTypeEnum,
    optimization_tier: OptimizationTierEnum
) -> str:
    """Pure function to generate cache key."""
    
    # Create a deterministic string for hashing
    content = f"{text_content}:{analysis_type}:{optimization_tier}"
    
    # Generate hash
    hash_object = hashlib.sha256(content.encode())
    return f"analysis:{hash_object.hexdigest()}"

def transform_analysis_to_response(
    analysis: Any,
    include_metadata: bool = True
) -> AnalysisResponse:
    """Pure function to transform database model to response."""
    response_data = {
        "id": analysis.id,
        "text_content": analysis.text_content,
        "analysis_type": analysis.analysis_type,
        "status": analysis.status,
        "sentiment_score": analysis.sentiment_score,
        "quality_score": analysis.quality_score,
        "processing_time_ms": analysis.processing_time_ms,
        "model_used": analysis.model_used,
        "confidence_score": analysis.confidence_score,
        "optimization_tier": analysis.optimization_tier,
        "created_at": analysis.created_at,
        "updated_at": analysis.updated_at,
        "processed_at": analysis.processed_at
    }
    
    if include_metadata:
        response_data["metadata"] = analysis.metadata or {}
    
    return AnalysisResponse(**response_data)

# ============================================================================
# Functional Decorators and Utilities
# ============================================================================

def with_error_handling(
    error_handler: Optional[Callable[[Exception], Dict[str, Any]]] = None
):
    """Functional decorator for error handling."""
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[ProcessingResult[R]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ProcessingResult[R]:
            start_time = datetime.now()
            
            try:
                result = await func(*args, **kwargs)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return ProcessingResult(
                    success=True,
                    data=result,
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if error_handler:
                    error_data = error_handler(e)
                    return ProcessingResult(
                        success=False,
                        error=str(e),
                        processing_time_ms=processing_time,
                        metadata=error_data
                    )
                
                return ProcessingResult(
                    success=False,
                    error=str(e),
                    processing_time_ms=processing_time
                )
        
        return wrapper
    return decorator

def with_validation(validator_func: Callable[[Any], ValidationResult]):
    """Functional decorator for input validation."""
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[ProcessingResult[R]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ProcessingResult[R]:
            # Validate first argument (assuming it's the input)
            if args:
                validation_result = validator_func(args[0])
                if not validation_result.is_valid:
                    return ProcessingResult(
                        success=False,
                        error="Validation failed",
                        metadata={"validation_errors": validation_result.errors}
                    )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def with_caching(cache_key_func: Callable[..., str], ttl_seconds: int = 3600):
    """Functional decorator for caching."""
    cache = {}  # In production, use Redis or similar
    
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> R:
            cache_key = cache_key_func(*args, **kwargs)
            
            # Check cache
            if cache_key in cache:
                cached_data, timestamp = cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < ttl_seconds:
                    return cached_data
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[cache_key] = (result, datetime.now())
            
            return result
        
        return wrapper
    return decorator

def with_logging(logger_name: str):
    """Functional decorator for logging."""
    logger = structlog.get_logger(logger_name)
    
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> R:
            start_time = datetime.now()
            
            logger.info(
                "Function started",
                function=func.__name__,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            try:
                result = await func(*args, **kwargs)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "Function completed",
                    function=func.__name__,
                    processing_time=processing_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    processing_time=processing_time,
                    error=str(e),
                    success=False
                )
                raise
        
        return wrapper
    return decorator

# ============================================================================
# Functional Service Layer
# ============================================================================

@with_logging("analysis_service")
@with_error_handling()
async def create_analysis_service(
    request: TextAnalysisRequest,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    """Pure function to create analysis."""
    # Validate input
    validation_result = validate_text_content(request.text_content)
    if not validation_result.is_valid:
        raise ValidationError("Invalid text content")
    
    # Calculate processing priority
    priority = calculate_processing_priority(
        request.optimization_tier,
        len(request.text_content),
        request.analysis_type
    )
    
    # Create analysis data
    analysis_data = TextAnalysisCreate(
        text_content=request.text_content,
        analysis_type=request.analysis_type,
        optimization_tier=request.optimization_tier
    )
    
    # Save to database
    analysis = await db_manager.create_text_analysis(analysis_data)
    
    # Transform to response
    return transform_analysis_to_response(analysis)

@with_logging("analysis_service")
@with_error_handling()
async def get_analysis_service(
    analysis_id: int,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    """Pure function to get analysis."""
    analysis = await db_manager.get_text_analysis(analysis_id)
    if not analysis:
        raise ValueError(f"Analysis with id {analysis_id} not found")
    
    return transform_analysis_to_response(analysis)

@with_logging("analysis_service")
@with_error_handling()
async def update_analysis_service(
    analysis_id: int,
    update_request: AnalysisUpdateRequest,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    """Pure function to update analysis."""
    # Convert to database update model
    update_data = TextAnalysisUpdate(**update_request.model_dump(exclude_unset=True))
    
    # Update in database
    analysis = await db_manager.update_text_analysis(analysis_id, update_data)
    if not analysis:
        raise ValueError(f"Analysis with id {analysis_id} not found")
    
    return transform_analysis_to_response(analysis)

@with_logging("analysis_service")
@with_error_handling()
async def list_analyses_service(
    pagination: PaginationRequest,
    filters: AnalysisFilterRequest,
    db_manager: SQLAlchemy2Manager
) -> PaginatedResponse[AnalysisResponse]:
    """Pure function to list analyses."""
    # Build filter parameters
    filter_params = {}
    if filters.analysis_type:
        filter_params["analysis_type"] = filters.analysis_type
    if filters.status:
        filter_params["status"] = filters.status
    if filters.optimization_tier:
        filter_params["optimization_tier"] = filters.optimization_tier
    
    # Get analyses from database
    analyses = await db_manager.list_text_analyses(
        limit=pagination.size,
        offset=pagination.offset,
        order_by=pagination.order_by,
        order_desc=pagination.order_desc,
        **filter_params
    )
    
    # Transform to responses
    analysis_responses = [
        transform_analysis_to_response(analysis) 
        for analysis in analyses
    ]
    
    # For simplicity, assume total count
    total = len(analyses) + pagination.offset
    
    return PaginatedResponse.create(
        items=analysis_responses,
        total=total,
        page=pagination.page,
        size=pagination.size
    )

@with_logging("batch_service")
@with_error_handling()
async def create_batch_service(
    request: BatchAnalysisRequest,
    db_manager: SQLAlchemy2Manager
) -> BatchAnalysisResponse:
    """Pure function to create batch analysis."""
    # Create batch
    batch_data = BatchAnalysisCreate(
        batch_name=request.batch_name,
        analysis_type=request.analysis_type,
        optimization_tier=request.optimization_tier
    )
    
    batch = await db_manager.create_batch_analysis(batch_data)
    
    # Calculate progress
    progress = calculate_batch_progress(0, 0, len(request.texts))
    
    # Create response
    return BatchAnalysisResponse(
        id=batch.id,
        batch_name=batch.batch_name,
        analysis_type=batch.analysis_type,
        status=batch.status,
        total_texts=len(request.texts),
        completed_count=0,
        error_count=0,
        progress_percentage=progress["progress_percentage"],
        optimization_tier=batch.optimization_tier,
        priority=request.priority,
        created_at=batch.created_at,
        updated_at=batch.updated_at,
        completed_at=batch.completed_at,
        metadata={"priority": request.priority}
    )

# ============================================================================
# Functional API Handlers
# ============================================================================

async def create_analysis_handler(
    request: TextAnalysisRequest,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    """Functional API handler for creating analysis."""
    result = await create_analysis_service(request, db_manager)
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error
        )
    
    return result.data

async def get_analysis_handler(
    analysis_id: int,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    """Functional API handler for getting analysis."""
    result = await get_analysis_service(analysis_id, db_manager)
    
    if not result.success:
        if "not found" in result.error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error
        )
    
    return result.data

async def update_analysis_handler(
    analysis_id: int,
    update_request: AnalysisUpdateRequest,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    """Functional API handler for updating analysis."""
    result = await update_analysis_service(analysis_id, update_request, db_manager)
    
    if not result.success:
        if "not found" in result.error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error
        )
    
    return result.data

async def list_analyses_handler(
    pagination: PaginationRequest = Depends(),
    filters: AnalysisFilterRequest = Depends(),
    db_manager: SQLAlchemy2Manager = Depends()
) -> PaginatedResponse[AnalysisResponse]:
    """Functional API handler for listing analyses."""
    result = await list_analyses_service(pagination, filters, db_manager)
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error
        )
    
    return result.data

async def create_batch_handler(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    db_manager: SQLAlchemy2Manager
) -> BatchAnalysisResponse:
    """Functional API handler for creating batch."""
    result = await create_batch_service(request, db_manager)
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error
        )
    
    # Add background task for processing
    background_tasks.add_task(
        process_batch_texts_functional,
        result.data.id,
        request.texts,
        request.analysis_type,
        db_manager
    )
    
    return result.data

# ============================================================================
# Functional Background Tasks
# ============================================================================

async def process_batch_texts_functional(
    batch_id: int,
    texts: List[str],
    analysis_type: AnalysisTypeEnum,
    db_manager: SQLAlchemy2Manager
):
    """Functional background task for processing batch texts."""
    logger = structlog.get_logger("batch_processor")
    
    completed_count = 0
    error_count = 0
    
    logger.info(f"Starting batch processing for batch {batch_id}")
    
    for i, text in enumerate(texts):
        try:
            # Create analysis request
            analysis_request = TextAnalysisRequest(
                text_content=text,
                analysis_type=analysis_type
            )
            
            # Process analysis
            analysis_result = await create_analysis_service(analysis_request, db_manager)
            
            if analysis_result.success:
                # Update with simulated results
                update_request = AnalysisUpdateRequest(
                    status=AnalysisStatusEnum.COMPLETED,
                    sentiment_score=0.5 + (i * 0.1),
                    processing_time_ms=100.0 + i,
                    model_used="functional-model"
                )
                
                await update_analysis_service(analysis_result.data.id, update_request, db_manager)
                completed_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing text {i+1} in batch {batch_id}: {e}")
    
    # Update batch progress
    await db_manager.update_batch_progress(batch_id, completed_count, error_count)
    
    logger.info(
        f"Completed batch {batch_id}: {completed_count} successful, {error_count} errors"
    )

# ============================================================================
# Example Usage
# ============================================================================

async def example_functional_usage():
    """Example usage of functional components."""
    
    # Create analysis request
    request = TextAnalysisRequest(
        text_content="This is a sample text for functional analysis.",
        analysis_type=AnalysisTypeEnum.SENTIMENT,
        optimization_tier=OptimizationTierEnum.STANDARD
    )
    
    # Validate request
    validation_result = validate_text_content(request.text_content)
    print(f"Validation result: {validation_result}")
    
    # Calculate priority
    priority = calculate_processing_priority(
        request.optimization_tier,
        len(request.text_content),
        request.analysis_type
    )
    print(f"Processing priority: {priority}")
    
    # Estimate processing time
    estimated_time = estimate_processing_time(
        len(request.text_content),
        request.analysis_type,
        request.optimization_tier
    )
    print(f"Estimated processing time: {estimated_time}ms")
    
    # Generate cache key
    cache_key = generate_cache_key(
        request.text_content,
        request.analysis_type,
        request.optimization_tier
    )
    print(f"Cache key: {cache_key}")

match __name__:
    case "__main__":
    asyncio.run(example_functional_usage()) 