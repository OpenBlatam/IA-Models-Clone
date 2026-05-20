"""
Utils Module
General utilities and helpers
"""

from .base import (
    ValidationError,
    ConfigurationError,
    RetryConfig,
    CacheConfig,
    UtilityBase
)
from .service import (
    UtilityService,
    retry_on_failure,
    log_execution_time,
    validate_input
)
from .exceptions import (
    BaseAppException,
    ValidationError as ValidationException,
    ConfigurationError as ConfigException,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConflictError,
    ServiceUnavailableError,
    RateLimitError,
    DatabaseError,
    ExternalServiceError
)
from .health import (
    HealthStatus,
    HealthCheck,
    SystemHealth,
    HealthChecker
)
from .circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerManager
)
from .rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitExceededError
)
from .validators import (
    BaseValidator,
    EmailValidator,
    URLValidator,
    PasswordValidator,
    DateTimeRangeValidator,
    PaginationValidator,
    SortValidator,
    SearchQueryValidator,
    IDValidator,
    ListValidator,
    validate_uuid,
    validate_phone,
    validate_slug,
    validate_json_string,
    validate_enum
)

__all__ = [
    # Base classes
    "ValidationError",
    "ConfigurationError",
    "RetryConfig",
    "CacheConfig",
    "UtilityBase",
    # Service
    "UtilityService",
    # Decorators
    "retry_on_failure",
    "log_execution_time",
    "validate_input",
    # Exceptions
    "BaseAppException",
    "ValidationException",
    "ConfigException",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ConflictError",
    "ServiceUnavailableError",
    "RateLimitError",
    "DatabaseError",
    "ExternalServiceError",
    # Health checks
    "HealthStatus",
    "HealthCheck",
    "SystemHealth",
    "HealthChecker",
    # Circuit breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitBreakerManager",
    # Rate limiting
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitExceededError",
    # Validators
    "BaseValidator",
    "EmailValidator",
    "URLValidator",
    "PasswordValidator",
    "DateTimeRangeValidator",
    "PaginationValidator",
    "SortValidator",
    "SearchQueryValidator",
    "IDValidator",
    "ListValidator",
    "validate_uuid",
    "validate_phone",
    "validate_slug",
    "validate_json_string",
    "validate_enum",
]
