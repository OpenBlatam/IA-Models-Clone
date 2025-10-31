from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import logging
from typing import Dict, Any, Optional, List, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid
from pydantic import (
        import re
        import re
    import re
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Pydantic Schemas
Product Descriptions Feature - Comprehensive Input/Output Validation and Response Schemas
"""


    BaseModel, 
    Field, 
    validator, 
    root_validator,
    ConfigDict,
    EmailStr,
    HttpUrl,
    IPvAnyAddress,
    conint,
    confloat,
    constr,
    AnyUrl
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for type safety
class GitStatus(str, Enum):
    """Git status enumeration"""
    UNTRACKED = "untracked"
    MODIFIED = "modified"
    STAGED = "staged"
    COMMITTED = "committed"
    IGNORED = "ignored"

class ModelStatus(str, Enum):
    """Model status enumeration"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class OperationType(str, Enum):
    """Operation type enumeration"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    VALIDATE = "validate"

class SeverityLevel(str, Enum):
    """Severity level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CacheStrategy(str, Enum):
    """Cache strategy enumeration"""
    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"

# Base Models with common configurations
class BaseRequestModel(BaseModel):
    """Base request model with common configurations"""
    model_config = ConfigDict(
        extra="forbid",  # Reject extra fields
        validate_assignment=True,  # Validate on assignment
        str_strip_whitespace=True,  # Strip whitespace from strings
        str_min_length=1,  # Minimum string length
        use_enum_values=True,  # Use enum values
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }
    )
    
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Request timestamp"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for distributed tracing"
    )

class BaseResponseModel(BaseModel):
    """Base response model with common configurations"""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }
    )
    
    success: bool = Field(description="Operation success status")
    request_id: str = Field(description="Request ID for tracking")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )
    duration_ms: float = Field(description="Request duration in milliseconds")
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for distributed tracing"
    )

class BaseErrorModel(BaseModel):
    """Base error model with common configurations"""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True
    )
    
    error_code: str = Field(description="Error code identifier")
    message: str = Field(description="Human-readable error message")
    details: Optional[str] = Field(default=None, description="Detailed error description")
    severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM, description="Error severity level")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for debugging")

# Git-related schemas
class GitFileInfo(BaseModel):
    """Git file information"""
    path: str = Field(description="File path")
    status: GitStatus = Field(description="File status")
    size: Optional[int] = Field(default=None, description="File size in bytes")
    last_modified: Optional[datetime] = Field(default=None, description="Last modification time")
    staged: bool = Field(default=False, description="Whether file is staged")
    
    @validator('path')
    def validate_path(cls, v) -> bool:
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()

class GitStatusRequest(BaseRequestModel):
    """Request model for git status"""
    include_untracked: bool = Field(
        default=True, 
        description="Include untracked files"
    )
    include_ignored: bool = Field(
        default=False, 
        description="Include ignored files"
    )
    include_staged: bool = Field(
        default=True, 
        description="Include staged files"
    )
    max_files: Optional[int] = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of files to return"
    )
    
    @validator('max_files')
    def validate_max_files(cls, v) -> bool:
        if v is not None and (v < 1 or v > 1000):
            raise ValueError("max_files must be between 1 and 1000")
        return v

class GitStatusResponse(BaseResponseModel):
    """Response model for git status"""
    data: Dict[str, Any] = Field(description="Git status data")
    files: List[GitFileInfo] = Field(
        default_factory=list,
        description="List of file information"
    )
    branch: Optional[str] = Field(default=None, description="Current branch name")
    commit_hash: Optional[str] = Field(default=None, description="Current commit hash")
    is_clean: bool = Field(description="Whether working directory is clean")
    total_files: int = Field(description="Total number of files")
    staged_files: int = Field(description="Number of staged files")
    modified_files: int = Field(description="Number of modified files")
    untracked_files: int = Field(description="Number of untracked files")

class CreateBranchRequest(BaseRequestModel):
    """Request model for creating branch"""
    branch_name: constr(min_length=1, max_length=100) = Field(
        description="Name of the new branch"
    )
    base_branch: constr(min_length=1, max_length=100) = Field(
        default="main",
        description="Base branch to create from"
    )
    checkout: bool = Field(
        default=True,
        description="Checkout the new branch after creation"
    )
    push_remote: bool = Field(
        default=False,
        description="Push branch to remote repository"
    )
    
    @validator('branch_name')
    def validate_branch_name(cls, v) -> bool:
        # Git branch naming rules
        if not v or not v.strip():
            raise ValueError("Branch name cannot be empty")
        
        invalid_chars = ['..', '~', '^', ':', '?', '*', '[', '\\', ' ', '\t', '\n', '\r']
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Branch name cannot contain '{char}'")
        
        if v.startswith('-'):
            raise ValueError("Branch name cannot start with '-'")
        
        if v.endswith('.'):
            raise ValueError("Branch name cannot end with '.'")
        
        return v.strip()
    
    @validator('base_branch')
    def validate_base_branch(cls, v) -> bool:
        if not v or not v.strip():
            raise ValueError("Base branch cannot be empty")
        return v.strip()

class CreateBranchResponse(BaseResponseModel):
    """Response model for creating branch"""
    data: Dict[str, Any] = Field(description="Branch creation data")
    branch_name: str = Field(description="Created branch name")
    base_branch: str = Field(description="Base branch used")
    checkout_performed: bool = Field(description="Whether checkout was performed")
    push_performed: bool = Field(description="Whether push was performed")
    commit_hash: Optional[str] = Field(default=None, description="Commit hash at branch creation")

class CommitChangesRequest(BaseRequestModel):
    """Request model for committing changes"""
    message: constr(min_length=1, max_length=500) = Field(
        description="Commit message"
    )
    files: Optional[List[constr(min_length=1)]] = Field(
        default=None,
        description="Specific files to commit"
    )
    include_untracked: bool = Field(
        default=True,
        description="Include untracked files"
    )
    amend: bool = Field(
        default=False,
        description="Amend previous commit"
    )
    author_name: Optional[str] = Field(
        default=None,
        description="Author name for commit"
    )
    author_email: Optional[EmailStr] = Field(
        default=None,
        description="Author email for commit"
    )
    
    @validator('message')
    def validate_message(cls, v) -> bool:
        if not v or not v.strip():
            raise ValueError("Commit message cannot be empty")
        
        # Check for common commit message patterns
        lines = v.strip().split('\n')
        if len(lines) > 0 and len(lines[0]) > 72:
            raise ValueError("First line of commit message should be 72 characters or less")
        
        return v.strip()
    
    @validator('files')
    def validate_files(cls, v) -> bool:
        if v is not None:
            if not v:
                raise ValueError("Files list cannot be empty if specified")
            # Remove duplicates and empty strings
            unique_files = list(set(f.strip() for f in v if f.strip()))
            if not unique_files:
                raise ValueError("Files list contains only empty strings")
            return unique_files
        return v

class CommitChangesResponse(BaseResponseModel):
    """Response model for committing changes"""
    data: Dict[str, Any] = Field(description="Commit data")
    commit_hash: str = Field(description="Commit hash")
    message: str = Field(description="Commit message")
    author: Optional[str] = Field(default=None, description="Commit author")
    timestamp: datetime = Field(description="Commit timestamp")
    files_committed: List[str] = Field(
        default_factory=list,
        description="List of committed files"
    )
    total_files: int = Field(description="Total number of files committed")

# Model versioning schemas
class ModelVersion(BaseModel):
    """Model version information"""
    version: constr(min_length=1, max_length=50) = Field(description="Version identifier")
    description: Optional[str] = Field(default=None, description="Version description")
    tags: List[str] = Field(
        default_factory=list,
        description="Version tags"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Version creation time"
    )
    status: ModelStatus = Field(
        default=ModelStatus.DRAFT,
        description="Version status"
    )
    file_size: Optional[int] = Field(default=None, description="Model file size in bytes")
    checksum: Optional[str] = Field(default=None, description="Model file checksum")
    
    @validator('version')
    def validate_version(cls, v) -> bool:
        if not v or not v.strip():
            raise ValueError("Version cannot be empty")
        
        # Semantic versioning pattern
        semver_pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        if not re.match(semver_pattern, v):
            raise ValueError("Version should follow semantic versioning (e.g., 1.0.0)")
        
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        if v:
            # Remove duplicates and empty strings
            unique_tags = list(set(tag.strip() for tag in v if tag.strip()))
            return unique_tags
        return v

class ModelVersionRequest(BaseRequestModel):
    """Request model for model versioning"""
    model_name: constr(min_length=1, max_length=100) = Field(
        description="Name of the model"
    )
    version: constr(min_length=1, max_length=50) = Field(
        description="Version identifier"
    )
    description: Optional[str] = Field(
        default=None,
        description="Version description"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Version tags"
    )
    status: ModelStatus = Field(
        default=ModelStatus.DRAFT,
        description="Initial version status"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional model metadata"
    )
    
    @validator('model_name')
    def validate_model_name(cls, v) -> bool:
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        
        # Model naming rules
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Model name can only contain letters, numbers, underscores, and hyphens")
        
        return v.strip()

class ModelVersionResponse(BaseResponseModel):
    """Response model for model versioning"""
    data: Dict[str, Any] = Field(description="Model version data")
    model_name: str = Field(description="Model name")
    version_info: ModelVersion = Field(description="Version information")
    model_path: Optional[Path] = Field(default=None, description="Model file path")
    download_url: Optional[HttpUrl] = Field(default=None, description="Model download URL")
    dependencies: Optional[List[str]] = Field(
        default=None,
        description="Model dependencies"
    )

# Performance and optimization schemas
class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    response_time_ms: float = Field(description="Response time in milliseconds")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(default=None, description="CPU usage percentage")
    cache_hit_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cache hit rate (0.0 to 1.0)"
    )
    throughput_requests_per_second: Optional[float] = Field(
        default=None,
        description="Throughput in requests per second"
    )

class CacheInfo(BaseModel):
    """Cache information"""
    cache_type: CacheStrategy = Field(description="Cache strategy type")
    cache_size: Optional[int] = Field(default=None, description="Cache size")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    cache_evictions: int = Field(default=0, description="Number of cache evictions")
    ttl_seconds: Optional[int] = Field(default=None, description="Time to live in seconds")

class BatchProcessRequest(BaseRequestModel):
    """Request model for batch processing"""
    items: List[Any] = Field(description="Items to process")
    operation: OperationType = Field(description="Operation to perform")
    batch_size: conint(ge=1, le=1000) = Field(
        default=10,
        description="Batch size"
    )
    max_concurrent: conint(ge=1, le=50) = Field(
        default=5,
        description="Maximum concurrent operations"
    )
    timeout_seconds: conint(ge=1, le=3600) = Field(
        default=300,
        description="Operation timeout in seconds"
    )
    retry_attempts: conint(ge=0, le=5) = Field(
        default=3,
        description="Number of retry attempts"
    )
    
    @validator('items')
    def validate_items(cls, v) -> bool:
        if not v:
            raise ValueError("Items list cannot be empty")
        if len(v) > 10000:
            raise ValueError("Items list cannot exceed 10,000 items")
        return v

class BatchProcessResponse(BaseResponseModel):
    """Response model for batch processing"""
    data: Dict[str, Any] = Field(description="Batch processing results")
    operation: OperationType = Field(description="Operation performed")
    total_items: int = Field(description="Total number of items")
    processed_items: int = Field(description="Number of processed items")
    failed_items: int = Field(description="Number of failed items")
    results: List[Any] = Field(
        default_factory=list,
        description="Processing results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Processing errors"
    )
    performance_metrics: Optional[PerformanceMetrics] = Field(
        default=None,
        description="Performance metrics"
    )

# Error and monitoring schemas
class ErrorContext(BaseModel):
    """Error context information"""
    field: Optional[str] = Field(default=None, description="Field that caused the error")
    value: Optional[Any] = Field(default=None, description="Value that caused the error")
    expected: Optional[Any] = Field(default=None, description="Expected value or format")
    suggestion: Optional[str] = Field(default=None, description="Suggestion to fix the error")
    documentation_url: Optional[HttpUrl] = Field(default=None, description="Link to relevant documentation")

class ValidationError(BaseErrorModel):
    """Validation error model"""
    validation_errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Field validation errors"
    )
    context: Optional[ErrorContext] = Field(
        default=None,
        description="Additional error context"
    )

class ErrorStats(BaseModel):
    """Error statistics"""
    total_errors: int = Field(description="Total number of errors")
    errors_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Errors grouped by type"
    )
    errors_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Errors grouped by severity"
    )
    errors_by_status_code: Dict[int, int] = Field(
        default_factory=dict,
        description="Errors grouped by HTTP status code"
    )
    errors_by_path: Dict[str, int] = Field(
        default_factory=dict,
        description="Errors grouped by request path"
    )
    recent_errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent error records"
    )
    error_rate: float = Field(description="Error rate per minute")
    avg_response_time: float = Field(description="Average response time")
    uptime: float = Field(description="Application uptime in seconds")

class MonitoringData(BaseModel):
    """Monitoring data"""
    error_stats: ErrorStats = Field(description="Error statistics")
    performance_metrics: PerformanceMetrics = Field(description="Performance metrics")
    cache_info: CacheInfo = Field(description="Cache information")
    system_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="System information"
    )
    active_requests: int = Field(description="Number of active requests")
    memory_usage: Optional[float] = Field(default=None, description="Memory usage percentage")
    cpu_usage: Optional[float] = Field(default=None, description="CPU usage percentage")

# Health and status schemas
class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ServiceHealth(BaseModel):
    """Service health information"""
    service_name: str = Field(description="Service name")
    status: HealthStatus = Field(description="Service status")
    last_check: datetime = Field(description="Last health check time")
    response_time_ms: Optional[float] = Field(default=None, description="Response time")
    error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")

class AppStatusResponse(BaseResponseModel):
    """Application status response"""
    status: HealthStatus = Field(description="Application status")
    version: str = Field(description="API version")
    uptime: float = Field(description="Application uptime in seconds")
    services: List[ServiceHealth] = Field(
        default_factory=list,
        description="Service health information"
    )
    environment: str = Field(description="Environment (development, staging, production)")
    build_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Build information"
    )

# Configuration schemas
class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = Field(description="Database host")
    port: conint(ge=1, le=65535) = Field(description="Database port")
    database: str = Field(description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password")
    pool_size: conint(ge=1, le=100) = Field(default=10, description="Connection pool size")
    max_overflow: conint(ge=0, le=50) = Field(default=20, description="Maximum overflow connections")
    
    @property
    def connection_string(self) -> str:
        """Get database connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class CacheConfig(BaseModel):
    """Cache configuration"""
    strategy: CacheStrategy = Field(description="Cache strategy")
    host: Optional[str] = Field(default=None, description="Cache host (for Redis)")
    port: Optional[conint(ge=1, le=65535)] = Field(default=None, description="Cache port")
    password: Optional[str] = Field(default=None, description="Cache password")
    database: Optional[conint(ge=0, le=15)] = Field(default=0, description="Cache database number")
    ttl_seconds: conint(ge=1, le=86400) = Field(default=3600, description="Default TTL in seconds")
    max_size: conint(ge=1, le=100000) = Field(default=10000, description="Maximum cache size")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[Path] = Field(default=None, description="Log file path")
    max_size_mb: conint(ge=1, le=1000) = Field(default=100, description="Maximum log file size in MB")
    backup_count: conint(ge=0, le=10) = Field(default=5, description="Number of backup files")

class AppConfig(BaseModel):
    """Application configuration"""
    app_name: str = Field(description="Application name")
    version: str = Field(description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        description="Environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    host: IPvAnyAddress = Field(default="0.0.0.0", description="Application host")
    port: conint(ge=1, le=65535) = Field(default=8000, description="Application port")
    database: DatabaseConfig = Field(description="Database configuration")
    cache: CacheConfig = Field(description="Cache configuration")
    logging: LoggingConfig = Field(description="Logging configuration")
    cors_origins: List[str] = Field(
        default_factory=list,
        description="CORS allowed origins"
    )
    rate_limit_requests: conint(ge=1, le=10000) = Field(
        default=100,
        description="Rate limit requests per minute"
    )
    rate_limit_window: conint(ge=1, le=3600) = Field(
        default=60,
        description="Rate limit window in seconds"
    )

# Utility functions for schema validation
def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Validate that all required fields are present"""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

def validate_field_length(value: str, field_name: str, min_length: int = 1, max_length: int = 1000) -> str:
    """Validate field length"""
    if not value or len(value.strip()) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters long")
    if len(value) > max_length:
        raise ValueError(f"{field_name} cannot exceed {max_length} characters")
    return value.strip()

def validate_email_format(email: str) -> str:
    """Validate email format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValueError("Invalid email format")
    return email.lower()

def validate_url_format(url: str) -> str:
    """Validate URL format"""
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")
    return url

# Schema factory functions
async def create_git_status_request(
    include_untracked: bool = True,
    include_ignored: bool = False,
    include_staged: bool = True,
    max_files: Optional[int] = 100
) -> GitStatusRequest:
    """Create a git status request"""
    return GitStatusRequest(
        include_untracked=include_untracked,
        include_ignored=include_ignored,
        include_staged=include_staged,
        max_files=max_files
    )

async def create_branch_request(
    branch_name: str,
    base_branch: str = "main",
    checkout: bool = True,
    push_remote: bool = False
) -> CreateBranchRequest:
    """Create a branch creation request"""
    return CreateBranchRequest(
        branch_name=branch_name,
        base_branch=base_branch,
        checkout=checkout,
        push_remote=push_remote
    )

async def create_commit_request(
    message: str,
    files: Optional[List[str]] = None,
    include_untracked: bool = True,
    amend: bool = False
) -> CommitChangesRequest:
    """Create a commit request"""
    return CommitChangesRequest(
        message=message,
        files=files,
        include_untracked=include_untracked,
        amend=amend
    )

async def create_model_version_request(
    model_name: str,
    version: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    status: ModelStatus = ModelStatus.DRAFT
) -> ModelVersionRequest:
    """Create a model version request"""
    return ModelVersionRequest(
        model_name=model_name,
        version=version,
        description=description,
        tags=tags,
        status=status
    )

async def create_batch_process_request(
    items: List[Any],
    operation: OperationType,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> BatchProcessRequest:
    """Create a batch process request"""
    return BatchProcessRequest(
        items=items,
        operation=operation,
        batch_size=batch_size,
        max_concurrent=max_concurrent
    )

# Response factory functions
def create_success_response(
    data: Dict[str, Any],
    request_id: Optional[str] = None,
    duration_ms: float = 0.0,
    correlation_id: Optional[str] = None
) -> BaseResponseModel:
    """Create a success response"""
    return BaseResponseModel(
        success=True,
        data=data,
        request_id=request_id or str(uuid.uuid4()),
        duration_ms=duration_ms,
        correlation_id=correlation_id
    )

def create_error_response(
    error_code: str,
    message: str,
    details: Optional[str] = None,
    severity: SeverityLevel = SeverityLevel.MEDIUM,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> BaseErrorModel:
    """Create an error response"""
    return BaseErrorModel(
        error_code=error_code,
        message=message,
        details=details,
        severity=severity,
        request_id=request_id,
        correlation_id=correlation_id
    )

def create_validation_error_response(
    message: str,
    validation_errors: List[Dict[str, Any]],
    field: Optional[str] = None,
    value: Optional[Any] = None,
    expected: Optional[Any] = None,
    suggestion: Optional[str] = None,
    request_id: Optional[str] = None
) -> ValidationError:
    """Create a validation error response"""
    context = ErrorContext(
        field=field,
        value=value,
        expected=expected,
        suggestion=suggestion
    )
    
    return ValidationError(
        error_code="VALIDATION_ERROR",
        message=message,
        severity=SeverityLevel.LOW,
        validation_errors=validation_errors,
        context=context,
        request_id=request_id
    ) 