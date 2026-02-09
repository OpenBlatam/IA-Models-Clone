"""Utils module for Logistics AI Platform"""

from .exceptions import (
    LogisticsException,
    NotFoundError,
    ValidationError,
    BusinessLogicError,
    ConflictError,
    RateLimitError,
)
from .cache import CacheService, cache_service
from .cache.helpers import (
    get_cached_or_fetch,
    cache_entity,
    invalidate_cache,
    invalidate_cache_pattern,
    invalidate_related_caches,
)
from .dependencies import get_services
from .geospatial import calculate_distance_km, geocode_location
from .json_serializer import json_dumps, json_loads
from .response import success_response, error_response, paginated_response
from .performance import batch_process, parallel_execute, lazy_load, memoize_async
from .async_utils import (
    run_in_background,
    gather_with_errors,
    timeout_after,
    retry_async,
    chunked_process,
)

__all__ = [
    "LogisticsException",
    "NotFoundError",
    "ValidationError",
    "BusinessLogicError",
    "ConflictError",
    "RateLimitError",
    "CacheService",
    "cache_service",
    "get_cached_or_fetch",
    "cache_entity",
    "invalidate_cache",
    "invalidate_cache_pattern",
    "invalidate_related_caches",
    "get_services",
    "calculate_distance_km",
    "geocode_location",
    "json_dumps",
    "json_loads",
    "success_response",
    "error_response",
    "paginated_response",
    "batch_process",
    "parallel_execute",
    "lazy_load",
    "memoize_async",
    "run_in_background",
    "gather_with_errors",
    "timeout_after",
    "retry_async",
    "chunked_process",
]
