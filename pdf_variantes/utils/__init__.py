"""
PDF Variantes Utils
Utility functions and helpers
"""

from .config import Settings, get_settings, get_settings_by_env, FeatureFlags, AIConfig, SecurityConfig, PerformanceConfig, ExportConfig
from .ai_helpers import AIProcessor, ContentAnalyzer
from .file_helpers import FileProcessor, PDFProcessor, FileStorageManager, FileValidator
from .cache_helpers import CacheManager, PerformanceOptimizer, CacheService
from .auth import SecurityService, AuthenticationService, get_current_user, require_permissions
from .logging_config import setup_logging, get_logger
from .structured_logging import (
    setup_structured_logging,
    get_logger as get_structured_logger,
    set_request_context,
    clear_request_context,
    log_with_context,
    log_performance
)
from .validation import (
    Validator,
    validate_filename,
    validate_file_extension,
    validate_integer_range,
    validate_string_length,
    validate_email,
    validate_uuid,
    validate_enum_value,
    validate_list_not_empty,
    ValidationError
)
from .validators import validate_file_upload, validate_content_type
from .text_utils import TextProcessor, TextAnalyzer
from .validation_utils import ValidationResult, InputValidator, DataValidator, SecurityValidator, BusinessLogicValidator
from .response_helpers import (
    create_response,
    create_error_response,
    create_success_response,
    create_paginated_response,
    create_validation_error_response,
    create_not_found_response,
    create_unauthorized_response,
    create_forbidden_response,
    create_rate_limit_response,
    get_request_id,
    set_request_id
)
from .optimization import (
    cached_async,
    batch_process,
    async_batch_process,
    measure_time,
    ConnectionPool,
    LazyLoader,
    chunk_iterable
)
from .real_world import (
    ErrorCode,
    RetryStrategy,
    retry_with_backoff,
    CircuitBreaker,
    with_timeout,
    HealthCheck,
    RateLimiter,
    validate_pdf_file,
    format_error_response,
    safe_divide
)
from .resilience import (
    FallbackStrategy,
    with_fallback,
    degrade_gracefully,
    Bulkhead,
    RequestQueue,
    batched_process,
    async_batched_process
)

__all__ = [
    # Configuration
    "Settings", "get_settings", "get_settings_by_env", "FeatureFlags", 
    "AIConfig", "SecurityConfig", "PerformanceConfig", "ExportConfig",
    
    # AI Helpers
    "AIProcessor", "ContentAnalyzer",
    
    # File Helpers
    "FileProcessor", "PDFProcessor", "FileStorageManager", "FileValidator",
    
    # Cache Helpers
    "CacheManager", "PerformanceOptimizer", "CacheService",
    
    # Authentication
    "SecurityService", "AuthenticationService", "get_current_user", "require_permissions",
    
    # Logging
    "setup_logging", "get_logger",
    "setup_structured_logging",
    "get_structured_logger",
    "set_request_context",
    "clear_request_context",
    "log_with_context",
    "log_performance",
    
    # Validation
    "Validator",
    "validate_filename",
    "validate_file_extension",
    "validate_integer_range",
    "validate_string_length",
    "validate_email",
    "validate_uuid",
    "validate_enum_value",
    "validate_list_not_empty",
    "ValidationError",
    
    # Validators
    "validate_file_upload", "validate_content_type",
    
    # Text Utils
    "TextProcessor", "TextAnalyzer",
    
    # Validation Utils
    "ValidationResult", "InputValidator", "DataValidator", "SecurityValidator", "BusinessLogicValidator",
    
    # Response Helpers
    "create_response",
    "create_error_response",
    "create_success_response",
    "create_paginated_response",
    "create_validation_error_response",
    "create_not_found_response",
    "create_unauthorized_response",
    "create_forbidden_response",
    "create_rate_limit_response",
    "get_request_id",
    "set_request_id",
    
    # Optimization
    "cached_async",
    "batch_process",
    "async_batch_process",
    "measure_time",
    "ConnectionPool",
    "LazyLoader",
    "chunk_iterable",
    
    # Real-World Utilities
    "ErrorCode",
    "RetryStrategy",
    "retry_with_backoff",
    "CircuitBreaker",
    "with_timeout",
    "HealthCheck",
    "RateLimiter",
    "validate_pdf_file",
    "format_error_response",
    "safe_divide",
    
    # Resilience
    "FallbackStrategy",
    "with_fallback",
    "degrade_gracefully",
    "Bulkhead",
    "RequestQueue",
    "batched_process",
    "async_batched_process"
]