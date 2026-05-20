"""
API utilities
"""

from .response_formatters import (
    format_track_response,
    format_tracks_response,
    format_paginated_response
)
from .error_handlers import (
    create_error_response,
    global_exception_handler,
    http_exception_handler
)
from .service_helpers import (
    get_track_or_search,
    get_track_full_data,
    format_artists,
    safe_get_nested
)
from .performance import (
    measure_time,
    PerformanceMonitor,
    performance_monitor
)
from .cache_helpers import (
    generate_cache_key,
    cache_key_from_params,
    cached_call
)
from .batch_helpers import (
    process_batch,
    chunk_list,
    process_parallel
)
from .async_helpers import (
    async_retry,
    timeout_after,
    run_with_timeout,
    gather_with_errors
)
from .router_helpers import (
    validate_track_ids_count,
    extract_track_ids_from_request,
    build_pagination_params,
    format_error_message
)
from .service_cache import (
    ServiceCache,
    get_cached_service,
    clear_service_cache
)
from .response_builders import (
    build_success_response,
    build_error_response,
    build_list_response
)
from .analysis_helpers import (
    perform_track_analysis,
    add_coaching_to_analysis,
    trigger_webhook_safe,
    save_analysis_to_history
)
from .export_helpers import (
    get_export_method,
    export_analysis
)
from .service_result_helpers import (
    require_success,
    require_not_none,
    extract_bearer_token,
    build_list_response_data,
    check_service_error
)

__all__ = [
    "format_track_response",
    "format_tracks_response",
    "format_paginated_response",
    "create_error_response",
    "global_exception_handler",
    "http_exception_handler",
    "get_track_or_search",
    "get_track_full_data",
    "format_artists",
    "safe_get_nested",
    "measure_time",
    "PerformanceMonitor",
    "performance_monitor",
    "generate_cache_key",
    "cache_key_from_params",
    "cached_call",
    "process_batch",
    "chunk_list",
    "process_parallel",
    "async_retry",
    "timeout_after",
    "run_with_timeout",
    "gather_with_errors",
    "validate_track_ids_count",
    "extract_track_ids_from_request",
    "build_pagination_params",
    "format_error_message",
    "ServiceCache",
    "get_cached_service",
    "clear_service_cache",
    "build_success_response",
    "build_error_response",
    "build_list_response",
    "perform_track_analysis",
    "add_coaching_to_analysis",
    "trigger_webhook_safe",
    "save_analysis_to_history",
    "get_export_method",
    "export_analysis",
    "require_success",
    "require_not_none",
    "extract_bearer_token",
    "build_list_response_data",
    "check_service_error"
]
