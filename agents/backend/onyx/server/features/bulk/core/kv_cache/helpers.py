"""
Helper functions for KV Cache.

Utility functions for common operations.
"""
from __future__ import annotations

import logging
from typing import Callable, Any, TypeVar

from kv_cache.types import StatsDict
from kv_cache.constants import (
    DEFAULT_LOW_HIT_RATE_THRESHOLD,
    DEFAULT_HIGH_MEMORY_MB_THRESHOLD,
    HIGH_HIT_RATE_RELAX,
    LOW_HIT_RATE_WARNING
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
CacheType = TypeVar("CacheType")


def create_cache_from_config(
    config_dict: dict[str, Any],
    cache_class: type[CacheType] | None = None
) -> CacheType:
    """
    Create cache instance from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        cache_class: Optional cache class (defaults to BaseKVCache)
        
    Returns:
        Cache instance
    """
    from kv_cache import KVCacheConfig, BaseKVCache
    
    if cache_class is None:
        cache_class = BaseKVCache
    
    config = KVCacheConfig(**config_dict)
    return cache_class(config)


def batch_process_cache_operations(
    cache: Any,
    operations: list[Callable[[Any], Any]],
    batch_size: int = 32
) -> list[Any]:
    """
    Process cache operations in batches.
    
    Args:
        cache: Cache instance
        operations: List of operation functions
        batch_size: Batch size for processing
        
    Returns:
        List of operation results
    """
    results = []
    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]
        batch_results = [op(cache) for op in batch]
        results.extend(batch_results)
    return results


def estimate_cache_memory(
    num_entries: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16
) -> float:
    """
    Estimate memory usage for cache.
    
    Args:
        num_entries: Number of cache entries
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        seq_len: Sequence length
        dtype: Data type
        
    Returns:
        Estimated memory in MB
    """
    from kv_cache.constants import BYTES_TO_MB
    
    element_size = torch.tensor(0, dtype=dtype).element_size()
    elements_per_entry = 2 * num_heads * seq_len * head_dim  # key + value
    total_elements = num_entries * elements_per_entry
    total_bytes = total_elements * element_size
    
    return total_bytes * BYTES_TO_MB


def validate_cache_config(config: Any) -> tuple[bool, str | None]:
    """
    Validate cache configuration.
    
    Args:
        config: Cache configuration
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if hasattr(config, 'validate'):
            config.validate()
            return True, None
        
        # Basic validation
        if hasattr(config, 'max_tokens') and config.max_tokens <= 0:
            return False, "max_tokens must be positive"
        
        if hasattr(config, 'head_dim') and config.head_dim <= 0:
            return False, "head_dim must be positive"
        
        return True, None
    except Exception as e:
        return False, str(e)


def get_cache_recommendations(stats: StatsDict) -> list[str]:
    """
    Get recommendations based on cache statistics.
    
    Args:
        stats: Cache statistics dictionary
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    hit_rate = stats.get("hit_rate", 0.0)
    num_entries = stats.get("num_entries", 0)
    max_tokens = stats.get("max_tokens", 0)
    memory_mb = stats.get("storage_memory_mb", 0.0)
    
    # Low hit rate recommendations
    if hit_rate < LOW_HIT_RATE_WARNING:
        recommendations.append(
            f"Low hit rate ({hit_rate:.2%}): Consider increasing cache size "
            f"or adjusting eviction strategy"
        )
    
    # Memory recommendations
    if memory_mb > DEFAULT_HIGH_MEMORY_MB_THRESHOLD:
        recommendations.append(
            f"High memory usage ({memory_mb:.2f} MB): Consider enabling "
            f"compression or quantization"
        )
    
    # Capacity recommendations
    usage_ratio = num_entries / max_tokens if max_tokens > 0 else 0.0
    if usage_ratio > 0.9:
        recommendations.append(
            f"Cache nearly full ({usage_ratio:.2%}): Consider increasing "
            f"max_tokens or adjusting eviction strategy"
        )
    
    return recommendations


def format_cache_info(stats: StatsDict) -> str:
    """
    Format cache statistics as a readable string.
    
    Args:
        stats: Cache statistics dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("Cache Statistics:")
    lines.append(f"  Hit Rate: {stats.get('hit_rate', 0.0):.2%}")
    lines.append(f"  Entries: {stats.get('num_entries', 0)}/{stats.get('max_tokens', 0)}")
    lines.append(f"  Memory: {stats.get('storage_memory_mb', 0.0):.2f} MB")
    lines.append(f"  Hits: {stats.get('hits', 0)}")
    lines.append(f"  Misses: {stats.get('misses', 0)}")
    lines.append(f"  Evictions: {stats.get('evictions', 0)}")
    
    return "\n".join(lines)

