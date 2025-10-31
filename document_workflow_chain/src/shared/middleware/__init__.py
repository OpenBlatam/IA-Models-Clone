"""
Middleware Package
==================

Middleware components for the application.
"""

from .auth import (
    AuthMiddleware,
    JWTBearer,
    APIKeyBearer,
    get_current_user,
    get_current_user_optional,
    require_permissions,
    require_roles
)

from .cache import (
    CacheMiddleware,
    cache_response,
    invalidate_cache,
    get_cache_key
)

from .logging import (
    LoggingMiddleware,
    StructuredLogger,
    get_request_logger
)

from .monitoring import (
    MonitoringMiddleware,
    PrometheusMiddleware,
    get_metrics_endpoint
)

from .rate_limiter import (
    RateLimiterMiddleware,
    rate_limit,
    get_rate_limit_info
)

from .metrics_middleware import (
    MetricsMiddleware,
    CustomMetricsMiddleware,
    record_workflow_metrics,
    record_node_metrics,
    record_ai_metrics,
    record_cache_metrics
)

from .security_middleware import (
    SecurityMiddleware,
    JWTBearer,
    APIKeyBearer,
    get_current_user,
    get_current_user_optional,
    require_authentication,
    require_admin,
    require_api_access
)

from .error_handler import (
    ErrorHandlerMiddleware,
    ErrorResponse,
    BusinessLogicError,
    ValidationError,
    ResourceNotFoundError,
    PermissionDeniedError,
    RateLimitExceededError,
    handle_business_logic_error,
    handle_validation_error,
    handle_resource_not_found_error,
    handle_permission_denied_error,
    handle_rate_limit_exceeded_error,
    create_error_response,
    log_error
)

__all__ = [
    # Auth Middleware
    "AuthMiddleware",
    "JWTBearer",
    "APIKeyBearer",
    "get_current_user",
    "get_current_user_optional",
    "require_permissions",
    "require_roles",
    
    # Cache Middleware
    "CacheMiddleware",
    "cache_response",
    "invalidate_cache",
    "get_cache_key",
    
    # Logging Middleware
    "LoggingMiddleware",
    "StructuredLogger",
    "get_request_logger",
    
    # Monitoring Middleware
    "MonitoringMiddleware",
    "PrometheusMiddleware",
    "get_metrics_endpoint",
    
    # Rate Limiter Middleware
    "RateLimiterMiddleware",
    "rate_limit",
    "get_rate_limit_info",
    
    # Metrics Middleware
    "MetricsMiddleware",
    "CustomMetricsMiddleware",
    "record_workflow_metrics",
    "record_node_metrics",
    "record_ai_metrics",
    "record_cache_metrics",
    
    # Security Middleware
    "SecurityMiddleware",
    "JWTBearer",
    "APIKeyBearer",
    "get_current_user",
    "get_current_user_optional",
    "require_authentication",
    "require_admin",
    "require_api_access",
    
    # Error Handler Middleware
    "ErrorHandlerMiddleware",
    "ErrorResponse",
    "BusinessLogicError",
    "ValidationError",
    "ResourceNotFoundError",
    "PermissionDeniedError",
    "RateLimitExceededError",
    "handle_business_logic_error",
    "handle_validation_error",
    "handle_resource_not_found_error",
    "handle_permission_denied_error",
    "handle_rate_limit_exceeded_error",
    "create_error_response",
    "log_error"
]