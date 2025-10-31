"""
Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


# Enums
class PluginStatus(str, Enum):
    """Plugin status enumeration."""
    UNINSTALLED = "uninstalled"
    INSTALLED = "installed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class ExtensionPointType(str, Enum):
    """Extension point type enumeration."""
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    FILTER = "filter"
    AGGREGATION = "aggregation"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class MiddlewareType(str, Enum):
    """Middleware type enumeration."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    LOGGING = "logging"
    METRICS = "metrics"
    CACHING = "caching"
    RATE_LIMITING = "rate_limiting"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CUSTOM = "custom"


class ComponentScope(str, Enum):
    """Component scope enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class EventType(str, Enum):
    """Event type enumeration."""
    DOMAIN = "domain"
    INTEGRATION = "integration"
    SYSTEM = "system"
    USER = "user"
    AUDIT = "audit"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class EventPriority(str, Enum):
    """Event priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# Base schemas
class BaseResponse(BaseModel):
    """Base response schema."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response schema."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Plugin schemas
class PluginInfo(BaseModel):
    """Plugin information schema."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = Field(default_factory=list)
    status: PluginStatus = PluginStatus.UNINSTALLED
    installed_at: Optional[datetime] = None


class PluginInstallRequest(BaseModel):
    """Plugin installation request schema."""
    plugin_name: str = Field(..., min_length=1, max_length=100)
    auto_install_dependencies: bool = True


class PluginActivationRequest(BaseModel):
    """Plugin activation request schema."""
    plugin_name: str = Field(..., min_length=1, max_length=100)


# Extension schemas
class ExtensionPointInfo(BaseModel):
    """Extension point information schema."""
    name: str
    type: ExtensionPointType
    description: str = ""
    priority: int = Field(default=0, ge=0, le=1000)
    extension_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExtensionExecutionRequest(BaseModel):
    """Extension execution request schema."""
    point_name: str = Field(..., min_length=1, max_length=100)
    data: Any
    metadata: Optional[Dict[str, Any]] = None


class ExtensionExecutionResponse(BaseModel):
    """Extension execution response schema."""
    point_name: str
    original_data: Any
    processed_data: Any
    modified: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Middleware schemas
class MiddlewareInfo(BaseModel):
    """Middleware information schema."""
    name: str
    type: MiddlewareType
    priority: int = Field(default=0, ge=0, le=1000)
    enabled: bool = True
    execution_count: int = 0
    last_execution: Optional[datetime] = None
    average_execution_time: float = 0.0


class MiddlewareExecutionRequest(BaseModel):
    """Middleware execution request schema."""
    request_data: Any
    pipeline_name: str = "default"


class MiddlewareExecutionResponse(BaseModel):
    """Middleware execution response schema."""
    original_request: Any
    processed_request: Any
    response: Optional[Any] = None
    modified: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Component schemas
class ComponentInfo(BaseModel):
    """Component information schema."""
    name: str
    type: str
    scope: ComponentScope
    dependencies: List[str] = Field(default_factory=list)
    instance_count: int = 0
    registered_at: datetime = Field(default_factory=datetime.utcnow)


class ComponentRegistrationRequest(BaseModel):
    """Component registration request schema."""
    name: str = Field(..., min_length=1, max_length=100)
    component_type: str = Field(..., min_length=1, max_length=100)
    scope: ComponentScope = ComponentScope.TRANSIENT
    dependencies: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


# Event schemas
class EventPublishRequest(BaseModel):
    """Event publish request schema."""
    event_type: str = Field(..., min_length=1, max_length=100)
    event_data: Any
    priority: EventPriority = EventPriority.NORMAL
    metadata: Optional[Dict[str, Any]] = None


class EventSubscriptionRequest(BaseModel):
    """Event subscription request schema."""
    event_type: str = Field(..., min_length=1, max_length=100)
    filters: Optional[Dict[str, Any]] = None


# Workflow schemas
class WorkflowStepInfo(BaseModel):
    """Workflow step information schema."""
    step_id: str
    name: str
    type: str
    status: str
    dependencies: List[str] = Field(default_factory=list)
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


class WorkflowInfo(BaseModel):
    """Workflow information schema."""
    workflow_id: str
    name: str
    description: str = ""
    status: WorkflowStatus
    step_count: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WorkflowStartRequest(BaseModel):
    """Workflow start request schema."""
    workflow_id: str = Field(..., min_length=1, max_length=100)
    initial_data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


# Analysis schemas
class ContentAnalysisRequest(BaseModel):
    """Content analysis request schema."""
    content: str = Field(..., min_length=1, max_length=10000)
    model_version: str = Field(default="1.0.0", min_length=1, max_length=50)
    analysis_type: str = Field(default="comprehensive", min_length=1, max_length=50)
    metadata: Optional[Dict[str, Any]] = None
    
    @validator("content")
    def validate_content(cls, v):
        """Validate content is not empty after stripping."""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()


class ContentAnalysisResponse(BaseModel):
    """Content analysis response schema."""
    content: str
    model_version: str
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    word_count: int
    character_count: int
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    systems_used: Dict[str, bool] = Field(default_factory=dict)
    processing_time: Optional[float] = None


# Statistics schemas
class SystemStats(BaseModel):
    """System statistics schema."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    systems: Dict[str, Any] = Field(default_factory=dict)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    systems: Dict[str, bool] = Field(default_factory=dict)
    version: str = "8.0.0"
    uptime: Optional[float] = None




