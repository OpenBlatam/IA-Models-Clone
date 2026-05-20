"""
Utilities module for Lovable Community SAM3.
"""

# Import all utility functions
from .decorators import log_execution_time, handle_errors, validate_inputs
from .pagination import calculate_pagination_metadata
from .validators import validate_pagination, validate_limit, validate_sort
from .response_builder import build_success_response, build_error_response, build_paginated_response
from .cache_helpers import cache_key, cached_result
from .query_helpers import (
    apply_pagination,
    apply_sorting,
    safe_query_execute,
    build_filter_conditions
)
from .performance import (
    memoize_with_ttl,
    batch_fetch,
    optimize_query,
    chunk_list
)
from .serializers import serialize_model, serialize_list
from .transformers import (
    rename_keys,
    normalize_types,
    flatten_dict,
    unflatten_dict
)
from .api_docs import generate_endpoint_docs
from .security import (
    sanitize_input,
    validate_email,
    validate_url,
    sanitize_filename,
    generate_secure_token,
    hash_data
)
from .formatters import (
    format_datetime,
    format_number,
    format_file_size,
    format_percentage,
    truncate_text
)
from .async_helpers import (
    run_parallel,
    retry_async,
    timeout_async
)
from .repository_helpers import (
    build_query_filters,
    apply_date_range_filter,
    apply_text_search_filter,
    get_aggregate_stats,
    batch_update,
    batch_delete
)
from .statistics_helpers import (
    calculate_basic_stats,
    calculate_field_stats,
    count_by_condition,
    calculate_percentage,
    calculate_query_stats,
    group_and_count,
    calculate_trend
)
from .service_helpers import (
    build_filter_dict,
    apply_common_filters,
    safe_service_call,
    batch_process
)
from .request_context import (
    get_request_id,
    set_request_id,
    generate_request_id
)

__all__ = [
    # Decorators
    "log_execution_time",
    "handle_errors",
    "validate_inputs",
    # Pagination
    "calculate_pagination_metadata",
    # Validators
    "validate_pagination",
    "validate_limit",
    "validate_sort",
    "validate_title",
    "validate_description",
    "validate_tags",
    "validate_user_id_validator",
    "validate_chat_id_validator",
    "validate_vote_type",
    "validate_category",
    "validate_comment",
    "validate_string_length",
    # Response builders
    "build_success_response",
    "build_error_response",
    "build_paginated_response",
    # Cache helpers
    "cache_key",
    "cached_result",
    # Query helpers
    "apply_pagination",
    "apply_sorting",
    "safe_query_execute",
    "build_filter_conditions",
    # Performance
    "memoize_with_ttl",
    "batch_fetch",
    "optimize_query",
    "chunk_list",
    # Serializers
    "serialize_model",
    "serialize_list",
    # Transformers
    "rename_keys",
    "normalize_types",
    "flatten_dict",
    "unflatten_dict",
    # API docs
    "generate_endpoint_docs",
    # Security
    "sanitize_input",
    "validate_email",
    "validate_url",
    "sanitize_filename",
    "generate_secure_token",
    "hash_data",
    # Formatters
    "format_datetime",
    "format_number",
    "format_file_size",
    "format_percentage",
    "truncate_text",
    # Async helpers
    "run_parallel",
    "retry_async",
    "timeout_async",
    # Repository helpers
    "build_query_filters",
    "apply_date_range_filter",
    "apply_text_search_filter",
    "get_aggregate_stats",
    "batch_update",
    "batch_delete",
    # Statistics helpers
    "calculate_basic_stats",
    "calculate_field_stats",
    "count_by_condition",
    "calculate_percentage",
    "calculate_query_stats",
    "group_and_count",
    "calculate_trend",
    # Service helpers
    "build_filter_dict",
    "apply_common_filters",
    "safe_service_call",
    "batch_process",
    # Request context
    "get_request_id",
    "set_request_id",
    "generate_request_id",
]




