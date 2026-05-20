"""
Utils Package - Common utilities for Content Redundancy Detector
"""

# Validation utilities
from .validation import (
    ValidationError,
    validate_content_length,
    validate_text_input,
    validate_similarity_threshold,
    validate_batch_size,
    validate_list_not_empty,
    validate_uuid,
    validate_positive_number,
    ContentValidator
)

# Error codes and response helpers
from .error_codes import (
    ErrorCode,
    format_error_response,
    get_status_code_for_error
)

from .response_helpers import (
    get_request_id,
    set_request_id,
    create_success_response,
    create_error_response,
    create_paginated_response,
    json_response
)

# Structured logging
from .structured_logging import (
    setup_structured_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    log_with_context,
    log_performance
)

# Health checks
from .health_checks import (
    HealthCheck,
    health_check,
    setup_default_health_checks,
    check_webhook_health,
    check_cache_health,
    check_database_health,
    check_ai_ml_health
)

__all__ = [
    # Validation
    "ValidationError",
    "validate_content_length",
    "validate_text_input",
    "validate_similarity_threshold",
    "validate_batch_size",
    "validate_list_not_empty",
    "validate_uuid",
    "validate_positive_number",
    "ContentValidator",
    
    # Error codes
    "ErrorCode",
    "format_error_response",
    "get_status_code_for_error",
    
    # Response helpers
    "get_request_id",
    "set_request_id",
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
    "json_response",
    
    # Logging
    "setup_structured_logging",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "log_with_context",
    "log_performance",
    
    # Health checks
    "HealthCheck",
    "health_check",
    "setup_default_health_checks",
    "check_webhook_health",
    "check_cache_health",
    "check_database_health",
    "check_ai_ml_health",
]
