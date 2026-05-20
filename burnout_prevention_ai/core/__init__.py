"""
Core Module
===========

Core utilities, constants, and shared functionality.

Includes:
- Constants and configuration values
- Utility functions (JSON parsing, content extraction)
- Validators and sanitizers
- Security utilities
- Custom exceptions
- Cache implementation
- Metrics and observability
- Prompt building helpers
- Fallback data
- DateTime utilities
- Type aliases
- Hashing utilities
- Logging helpers
- Pydantic utilities
"""

from .constants import SYSTEM_PROMPT, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_CHAT_TEMPERATURE
from .utils import (
    extract_json_from_text,
    extract_content_from_response,
    extract_suggestions,
    extract_resources
)
from .validators import (
    validate_assessment_data,
    validate_conversation_history,
    validate_api_parameters,
    sanitize_assessment_data,
    validate_non_empty_list,
    validate_non_empty_dict,
    validate_positive_number
)
from .security import (
    sanitize_string,
    sanitize_list,
    validate_numeric_range,
    get_security_headers
)
from .exceptions import (
    BurnoutPreventionError,
    ValidationError,
    APIError,
    CacheError,
    ConfigurationError,
    ProcessingError
)
from .cache import get_cache, set_cache, clear_cache, make_cache_key, make_messages_cache_key, get_cache_stats, clear_expired
from .metrics import (
    record_api_request,
    record_cache_hit,
    record_cache_miss,
    update_cache_size,
    record_openrouter_request
)
from .prompt_builder import (
    build_system_user_messages,
    format_list_items,
    format_optional_field,
    build_assessment_prompt,
    build_wellness_check_prompt,
    build_coping_strategy_prompt,
    build_progress_tracking_prompt,
    build_trend_analysis_prompt,
    build_resource_prompt,
    build_personalized_plan_prompt
)
from .fallbacks import (
    get_assessment_fallback,
    get_wellness_fallback,
    get_coping_fallback,
    get_progress_fallback,
    get_trend_fallback,
    get_resource_fallback,
    get_plan_fallback
)
from .datetime_utils import (
    get_utc_now,
    get_iso_timestamp,
    get_utc_iso_timestamp
)
from .types import (
    JSONDict,
    MessageDict,
    MessageList,
    CacheEntry,
    APIResponse,
    ModelList
)
from .hashing import (
    hash_string,
    hash_data
)
from .logging_helpers import (
    log_error,
    log_warning,
    log_info,
    log_debug,
    truncate_error_message
)
from .pydantic_utils import (
    model_to_dict
)
from .data_extraction import (
    safe_get,
    safe_get_list,
    safe_get_float,
    safe_get_int,
    safe_get_str
)

__all__ = [
    "SYSTEM_PROMPT",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_CHAT_TEMPERATURE",
    "extract_json_from_text",
    "extract_content_from_response",
    "extract_suggestions",
    "extract_resources",
    "validate_assessment_data",
    "validate_conversation_history",
    "validate_api_parameters",
    "sanitize_assessment_data",
    "validate_non_empty_list",
    "validate_non_empty_dict",
    "validate_positive_number",
    "sanitize_string",
    "sanitize_list",
    "validate_numeric_range",
    "get_security_headers",
    "BurnoutPreventionError",
    "ValidationError",
    "APIError",
    "CacheError",
    "ConfigurationError",
    "ProcessingError",
    "get_cache",
    "set_cache",
    "clear_cache",
    "make_cache_key",
    "make_messages_cache_key",
    "get_cache_stats",
    "clear_expired",
    "record_api_request",
    "record_cache_hit",
    "record_cache_miss",
    "update_cache_size",
    "record_openrouter_request",
    "build_system_user_messages",
    "format_list_items",
    "format_optional_field",
    "build_assessment_prompt",
    "build_wellness_check_prompt",
    "build_coping_strategy_prompt",
    "build_progress_tracking_prompt",
    "build_trend_analysis_prompt",
    "build_resource_prompt",
    "build_personalized_plan_prompt",
    "get_assessment_fallback",
    "get_wellness_fallback",
    "get_coping_fallback",
    "get_progress_fallback",
    "get_trend_fallback",
    "get_resource_fallback",
    "get_plan_fallback",
    "get_utc_now",
    "get_iso_timestamp",
    "get_utc_iso_timestamp",
    "JSONDict",
    "MessageDict",
    "MessageList",
    "CacheEntry",
    "APIResponse",
    "ModelList",
    "hash_string",
    "hash_data",
    "log_error",
    "log_warning",
    "log_info",
    "log_debug",
    "truncate_error_message",
    "model_to_dict",
    "safe_get",
    "safe_get_list",
    "safe_get_float",
    "safe_get_int",
    "safe_get_str"
]

