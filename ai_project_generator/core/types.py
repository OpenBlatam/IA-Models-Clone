"""
Type Definitions - Definiciones de tipos
========================================

Definiciones de tipos comunes para todo el sistema.
"""

from typing import (
    Dict,
    Any,
    Optional,
    List,
    Union,
    Callable,
    Awaitable,
    Protocol,
    TypedDict,
    Literal,
    TypeVar,
    Generic,
    Sequence,
    Mapping,
    Iterator,
    AsyncIterator,
)
from datetime import datetime
from enum import Enum
from pathlib import Path

# Type Variables
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# Common Types
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONDict = Dict[str, JSONValue]
JSONList = List[JSONValue]

# Request/Response Types
RequestData = Dict[str, Any]
ResponseData = Dict[str, Any]
Headers = Dict[str, str]
QueryParams = Dict[str, Union[str, List[str]]]

# Service Types
ServiceName = str
ServiceURL = str
ServiceConfig = Dict[str, Any]

# Database Types
DatabaseKey = Union[str, int]
DatabaseValue = Dict[str, Any]
DatabaseQuery = Dict[str, Any]

# Cache Types
CacheKey = str
CacheValue = Any
CacheTTL = int

# Event Types
EventName = str
EventData = Dict[str, Any]
EventHandler = Callable[[EventData], Awaitable[None]]

# Worker Types
TaskID = str
TaskResult = Dict[str, Any]
TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# API Gateway Types
RoutePath = str
RouteMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
RouteConfig = Dict[str, Any]

# Security Types
IPAddress = str
UserID = str
Token = str
Scope = str

# Health Check Types
HealthStatus = Literal["healthy", "degraded", "unhealthy", "unknown"]
ComponentHealth = Dict[str, Any]

# Serverless Types
HandlerEvent = Dict[str, Any]
HandlerContext = Dict[str, Any]
HandlerResponse = Dict[str, Any]

# Protocol Definitions
class AsyncCallable(Protocol[T]):
    """Protocol for async callable"""
    async def __call__(self, *args: Any, **kwargs: Any) -> T: ...


class Repository(Protocol[T]):
    """Protocol for repository"""
    async def get(self, key: str) -> Optional[T]: ...
    async def put(self, key: str, value: T) -> bool: ...
    async def delete(self, key: str) -> bool: ...
    async def list(self, filters: Optional[Dict[str, Any]] = None) -> List[T]: ...


class CacheService(Protocol):
    """Protocol for cache service"""
    async def get(self, key: CacheKey) -> Optional[CacheValue]: ...
    async def set(self, key: CacheKey, value: CacheValue, ttl: Optional[CacheTTL] = None) -> bool: ...
    async def delete(self, key: CacheKey) -> bool: ...
    async def exists(self, key: CacheKey) -> bool: ...


class EventPublisher(Protocol):
    """Protocol for event publisher"""
    async def publish(self, event: EventName, data: EventData) -> bool: ...
    async def subscribe(self, event: EventName, handler: EventHandler) -> bool: ...


# TypedDict Definitions
class ProjectData(TypedDict, total=False):
    """Project data structure"""
    id: str
    description: str
    project_name: Optional[str]
    author: str
    version: str
    priority: int
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: Optional[str]
    status: TaskStatus


class ServiceConfigDict(TypedDict, total=False):
    """Service configuration"""
    name: ServiceName
    url: ServiceURL
    timeout: int
    retries: int
    circuit_breaker: bool
    rate_limit: Dict[str, Any]


class HealthCheckResult(TypedDict):
    """Health check result"""
    status: HealthStatus
    timestamp: str
    message: Optional[str]
    details: Dict[str, Any]


class RateLimitConfig(TypedDict, total=False):
    """Rate limit configuration"""
    strategy: str
    limit: int
    window: int
    per_consumer: bool
    per_ip: bool


class SecurityPolicy(TypedDict, total=False):
    """Security policy"""
    cors: Dict[str, Any]
    ip_restriction: Dict[str, Any]
    oauth2: Dict[str, Any]
    request_size_limit: int


# Enum Types
class HTTPMethod(str, Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


# Utility Types
MaybeAsync = Union[T, Awaitable[T]]
MaybeNone = Optional[T]
MaybeList = Union[T, List[T]]

# Function Types
AsyncFunction = Callable[..., Awaitable[T]]
SyncFunction = Callable[..., T]
MiddlewareFunction = Callable[[Any, Callable], Awaitable[Any]]

# Path Types
FilePath = Union[str, Path]
DirectoryPath = Union[str, Path]

# Time Types
Timestamp = Union[int, float, datetime]
Duration = Union[int, float]  # seconds

# Error Types
ErrorCode = str
ErrorMessage = str
ErrorDetails = Dict[str, Any]















