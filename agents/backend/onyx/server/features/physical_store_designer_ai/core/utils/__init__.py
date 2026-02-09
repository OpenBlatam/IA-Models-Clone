"""
Utilities Module

Refactored utilities organized by category.
"""

# Import all utilities from submodules
from .logging_utils import setup_logging, JsonFormatter
from .validation_utils import validate_store_type, validate_design_style
from .file_utils import sanitize_filename, ensure_directory
from .format_utils import format_error_response, format_success_response, format_bytes
from .data_utils import generate_id, truncate_text, batch_process, chunk_list, retry_with_backoff
from .math_utils import safe_divide, parse_bool, clamp, calculate_percentage, normalize_percentage
from .dict_utils import (
    deep_merge,
    get_nested_value,
    set_nested_value,
    normalize_dict_keys,
    filter_dict,
    exclude_dict_keys,
    flatten_dict,
    unflatten_dict,
)
from .json_utils import safe_json_loads, safe_json_dumps
from .response_utils import (
    create_success_response,
    create_error_response,
    create_paginated_response
)
from .error_utils import (
    get_error_response,
    is_client_error,
    is_server_error,
    should_retry,
    get_retryable_status_codes
)

__all__ = [
    # Logging
    "setup_logging",
    "JsonFormatter",
    # Validation
    "validate_store_type",
    "validate_design_style",
    # File operations
    "sanitize_filename",
    "ensure_directory",
    # Formatting
    "format_error_response",
    "format_success_response",
    "format_bytes",
    # Data utilities
    "generate_id",
    "truncate_text",
    "batch_process",
    "chunk_list",
    "retry_with_backoff",
    # Math utilities
    "safe_divide",
    "parse_bool",
    "clamp",
    "calculate_percentage",
    "normalize_percentage",
    # Dictionary utilities
    "deep_merge",
    "get_nested_value",
    "set_nested_value",
    "normalize_dict_keys",
    "filter_dict",
    "exclude_dict_keys",
    "flatten_dict",
    "unflatten_dict",
    # JSON utilities
    "safe_json_loads",
    "safe_json_dumps",
    # Response utilities
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
    # Error utilities
    "get_error_response",
    "is_client_error",
    "is_server_error",
    "should_retry",
    "get_retryable_status_codes",
]




