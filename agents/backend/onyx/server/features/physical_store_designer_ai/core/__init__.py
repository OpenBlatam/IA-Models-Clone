"""
Core models and business logic for Physical Store Designer AI
"""

from .models import (
    StoreType,
    DesignStyle,
    StoreDesignRequest,
    StoreDesign,
    MarketingPlan,
    DecorationPlan,
    StoreLayout,
    StoreVisualization,
    ChatMessage,
    ChatSession,
)
from .exceptions import (
    PhysicalStoreDesignerError,
    ValidationError,
    NotFoundError,
    StorageError,
    ServiceError,
    ExternalAPIError,
    RateLimitError,
    AuthenticationError,
    AuthorizationError,
    TimeoutError,
    ConflictError,
    TooManyRequestsError,
)
from .response_models import (
    SuccessResponse,
    ErrorResponse,
    PaginatedResponse,
    create_success_response,
    create_error_response,
    create_paginated_response,
)
from .decorators import (
    log_execution_time,
    retry_on_failure,
    validate_input,
)
from .dependencies import (
    verify_api_key,
    get_pagination_params,
    get_sort_params,
)
from .interfaces import (
    IStorageService,
    IChatService,
    IDesignService,
    IAnalysisService,
    IExportService,
)
from .factories import (
    ServiceFactory,
    ConfigFactory,
)
from .validators import (
    Validator,
    validate_and_raise,
)
from .metrics import (
    MetricsCollector,
    MetricsContext,
    get_metrics_collector,
    track_metric,
    increment_counter,
    time_operation,
)
from .route_helpers import (
    handle_route_errors,
    track_route_metrics,
)
from .service_base import (
    BaseService,
    TimestampedService,
)
from .route_utils import (
    get_client_info,
    track_request_metrics,
    log_request_start,
    log_request_end,
    get_query_params,
    get_path_params,
    get_request_body_size,
    is_json_request,
    get_request_id,
    extract_user_agent_info,
    build_pagination_response,
)
from .service_registry import (
    ServiceRegistry,
)
from .cache import (
    CacheManager,
    LRUCache,
    CacheEntry,
    get_cache_manager,
    cached,
    generate_cache_key,
)
from .serializers import (
    JSONEncoder,
    serialize_to_json,
    deserialize_from_json,
    serialize_to_base64,
    deserialize_from_base64,
    serialize_dict_keys,
    flatten_dict,
    unflatten_dict,
    sanitize_for_json,
    normalize_data_types,
    extract_nested_value,
    set_nested_value,
    merge_dicts,
    deep_merge_dicts,
)
from .background_tasks import (
    TaskStatus,
    BackgroundTask,
    TaskQueue,
    AsyncTaskExecutor,
    get_task_queue,
    submit_background_task,
    get_task_status,
)
from .compression import (
    compress_gzip,
    decompress_gzip,
    compress_deflate,
    decompress_deflate,
    compress_data,
    decompress_data,
    get_compression_ratio,
    should_compress,
)
from .rate_limiting import (
    EndpointRateLimiter,
    get_endpoint_rate_limiter,
    rate_limit_endpoint,
)
from .timeout import (
    timeout,
    timeout_context,
    with_timeout,
)
from .circuit_breaker import (
    CircuitState,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerEvent,
    CircuitBreakerEventType,
    CircuitBreakerGroup,
    CircuitBreakerChain,
    CircuitBreakerStateStore,
    InMemoryStateStore,
    circuit_breaker,
    get_circuit_breaker,
    get_circuit_breaker_sync,
    get_all_circuit_breakers,
    reset_all_circuit_breakers,
    get_trace_context,
    add_tracing_to_circuit_breaker,
    create_circuit_breaker_with_persistence,
)
from .utils import (
    generate_id,
    deep_merge,
    get_nested_value,
    set_nested_value,
    flatten_dict,
    unflatten_dict,
    safe_json_loads,
    safe_json_dumps,
    slugify,
    is_valid_email,
    is_valid_url,
    normalize_whitespace,
    time_ago,
    chunk_list,
    filter_dict,
    exclude_dict_keys,
)

__all__ = [
    # Models
    "StoreType",
    "DesignStyle",
    "StoreDesignRequest",
    "StoreDesign",
    "MarketingPlan",
    "DecorationPlan",
    "StoreLayout",
    "StoreVisualization",
    "ChatMessage",
    "ChatSession",
    # Exceptions
    "PhysicalStoreDesignerError",
    "ValidationError",
    "NotFoundError",
    "StorageError",
    "ServiceError",
    "ExternalAPIError",
    "RateLimitError",
    "AuthenticationError",
    "AuthorizationError",
    "TimeoutError",
    "ConflictError",
    "TooManyRequestsError",
    # Response Models
    "SuccessResponse",
    "ErrorResponse",
    "PaginatedResponse",
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
    # Decorators
    "log_execution_time",
    "retry_on_failure",
    "validate_input",
    # Dependencies
    "verify_api_key",
    "get_pagination_params",
    "get_sort_params",
    # Interfaces
    "IStorageService",
    "IChatService",
    "IDesignService",
    "IAnalysisService",
    "IExportService",
    # Factories
    "ServiceFactory",
    "ConfigFactory",
    # Validators
    "Validator",
    "validate_and_raise",
    # Metrics
    "MetricsCollector",
    "MetricsContext",
    "get_metrics_collector",
    "track_metric",
    "increment_counter",
    "time_operation",
    # Route Helpers
    "handle_route_errors",
    "track_route_metrics",
    # Service Base
    "BaseService",
    "TimestampedService",
    # Route Utils
    "get_client_info",
    "track_request_metrics",
    "log_request_start",
    "log_request_end",
    "get_query_params",
    "get_path_params",
    "get_request_body_size",
    "is_json_request",
    "get_request_id",
    "extract_user_agent_info",
    "build_pagination_response",
    # Service Registry
    "ServiceRegistry",
    # Cache
    "CacheManager",
    "LRUCache",
    "CacheEntry",
    "get_cache_manager",
    "cached",
    "generate_cache_key",
    # Serializers
    "JSONEncoder",
    "serialize_to_json",
    "deserialize_from_json",
    "serialize_to_base64",
    "deserialize_from_base64",
    "serialize_dict_keys",
    "flatten_dict",
    "unflatten_dict",
    "sanitize_for_json",
    "normalize_data_types",
    "extract_nested_value",
    "set_nested_value",
    "merge_dicts",
    "deep_merge_dicts",
    # Background Tasks
    "TaskStatus",
    "BackgroundTask",
    "TaskQueue",
    "AsyncTaskExecutor",
    "get_task_queue",
    "submit_background_task",
    "get_task_status",
    # Compression
    "compress_gzip",
    "decompress_gzip",
    "compress_deflate",
    "decompress_deflate",
    "compress_data",
    "decompress_data",
    "get_compression_ratio",
    "should_compress",
    # Rate Limiting
    "EndpointRateLimiter",
    "get_endpoint_rate_limiter",
    "rate_limit_endpoint",
    # Timeout
    "timeout",
    "timeout_context",
    "with_timeout",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerEvent",
    "CircuitBreakerEventType",
    "CircuitBreakerGroup",
    "CircuitBreakerChain",
    "CircuitBreakerStateStore",
    "InMemoryStateStore",
    "circuit_breaker",
    "get_circuit_breaker",
    "get_circuit_breaker_sync",
    "get_all_circuit_breakers",
    "reset_all_circuit_breakers",
    "get_trace_context",
    "add_tracing_to_circuit_breaker",
    "create_circuit_breaker_with_persistence",
    # Utils
    "generate_id",
    "deep_merge",
    "get_nested_value",
    "set_nested_value",
    "flatten_dict",
    "unflatten_dict",
    "safe_json_loads",
    "safe_json_dumps",
    "slugify",
    "is_valid_email",
    "is_valid_url",
    "normalize_whitespace",
    "time_ago",
    "chunk_list",
    "filter_dict",
    "exclude_dict_keys",
]




