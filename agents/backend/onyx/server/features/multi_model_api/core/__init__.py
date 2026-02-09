"""Core services for multi-model feature"""

from .models import ModelRegistry, ModelMetadata, get_registry
from .cache import (
    EnhancedCache,
    CacheStats,
    CacheEntry,
    TagManager,
    CacheContext,
    get_cache,
    close_cache,
    cached
)
from .health import HealthMonitor, HealthMetrics, get_health_monitor
from .rate_limiter import RateLimiter, RateLimitInfo, get_rate_limiter
from .consensus import (
    apply_consensus,
    simple_majority_vote,
    weighted_vote,
    similarity_clustering,
    average_consensus,
    best_performer_consensus
)
from .config import MultiModelConfig, get_config
from .utils import (
    generate_request_id,
    format_latency_ms,
    format_bytes,
    safe_json_dumps,
    safe_json_loads,
    timer,
    calculate_success_rate
)
from .context import (
    RequestContext,
    get_request_context,
    set_request_context,
    create_request_context,
    clear_request_context
)

__all__ = [
    "ModelRegistry",
    "ModelMetadata",
    "get_registry",
    "EnhancedCache",
    "CacheStats",
    "CacheEntry",
    "TagManager",
    "CacheContext",
    "get_cache",
    "close_cache",
    "cached",
    "HealthMonitor",
    "HealthMetrics",
    "get_health_monitor",
    "RateLimiter",
    "RateLimitInfo",
    "get_rate_limiter",
    "apply_consensus",
    "simple_majority_vote",
    "weighted_vote",
    "similarity_clustering",
    "average_consensus",
    "best_performer_consensus",
    "MultiModelConfig",
    "get_config",
    # Utils
    "generate_request_id",
    "format_latency_ms",
    "format_bytes",
    "safe_json_dumps",
    "safe_json_loads",
    "timer",
    "calculate_success_rate",
    # Context
    "RequestContext",
    "get_request_context",
    "set_request_context",
    "create_request_context",
    "clear_request_context"
]

