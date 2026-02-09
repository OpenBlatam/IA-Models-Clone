"""
Utilidades para la API

Incluye:
- response_cache: Caching de respuestas HTTP
- query_optimizer: Optimización de queries y búsquedas
- batch_processor: Procesamiento en batch optimizado
- performance_optimizations: Optimizaciones de rendimiento
"""

from .response_cache import cache_response, clear_response_cache, get_cache_stats
from .query_optimizer import (
    optimize_search_query,
    filter_songs_efficiently,
    paginate_results
)
from .batch_processor import (
    process_batch_async,
    process_batch_sync,
    batch_get_songs
)
from .validation_helpers import (
    validate_uuid_list,
    parse_comma_separated_ids,
    validate_prompt_length,
    sanitize_string,
    validate_genre,
    validate_duration
)
from .performance_monitor import (
    measure_time,
    performance_monitor,
    get_performance_stats,
    clear_performance_stats
)
from .error_handlers import (
    handle_service_error,
    safe_execute,
    safe_execute_async
)
from .async_helpers import (
    retry_async,
    gather_with_limit,
    timeout_async
)
from .rate_limit_helpers import (
    check_rate_limit,
    get_rate_limit_info,
    clear_rate_limit
)
from .compression import (
    compress_gzip,
    compress_brotli,
    get_best_compression,
    should_compress
)
from .request_helpers import (
    get_client_ip,
    get_user_agent,
    get_accept_encoding,
    add_cache_headers,
    add_cors_headers,
    get_request_metadata
)
from .decorators import (
    log_request,
    rate_limit_decorator,
    validate_request,
    cache_control,
    retry_on_failure,
    measure_performance,
    require_auth
)

__all__ = [
    "cache_response",
    "clear_response_cache",
    "get_cache_stats",
    "optimize_search_query",
    "filter_songs_efficiently",
    "paginate_results",
    "process_batch_async",
    "process_batch_sync",
    "batch_get_songs",
    "validate_uuid_list",
    "parse_comma_separated_ids",
    "validate_prompt_length",
    "sanitize_string",
    "validate_genre",
    "validate_duration",
    "measure_time",
    "performance_monitor",
    "get_performance_stats",
    "clear_performance_stats",
    "handle_service_error",
    "safe_execute",
    "safe_execute_async",
    "retry_async",
    "gather_with_limit",
    "timeout_async",
    "check_rate_limit",
    "get_rate_limit_info",
    "clear_rate_limit",
    "compress_gzip",
    "compress_brotli",
    "get_best_compression",
    "should_compress",
    "get_client_ip",
    "get_user_agent",
    "get_accept_encoding",
    "add_cache_headers",
    "add_cors_headers",
    "get_request_metadata",
    "log_request",
    "rate_limit_decorator",
    "validate_request",
    "cache_control",
    "retry_on_failure",
    "measure_performance",
    "require_auth"
]
