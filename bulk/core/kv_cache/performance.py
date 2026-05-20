"""
Performance optimization utilities for KV Cache.

Provides tools for analyzing and optimizing cache performance.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar, Any

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)

T = TypeVar("T")


def measure_latency(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """
    Measure latency of a function call.
    
    Args:
        func: Function to measure
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Tuple of (result, latency_ms)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    return result, elapsed


def calculate_throughput(num_operations: int, total_time_seconds: float) -> float:
    """
    Calculate operations per second.
    
    Args:
        num_operations: Number of operations performed
        total_time_seconds: Total time in seconds
        
    Returns:
        Operations per second
    """
    if total_time_seconds <= 0:
        return 0.0
    return num_operations / total_time_seconds


def estimate_cache_efficiency(stats: StatsDict) -> dict[str, float]:
    """
    Estimate cache efficiency metrics.
    
    Args:
        stats: Cache statistics
        
    Returns:
        Dictionary with efficiency metrics
    """
    hit_rate = stats.get("hit_rate", 0.0)
    num_entries = stats.get("num_entries", 0)
    max_tokens = stats.get("max_tokens", 0)
    evictions = stats.get("evictions", 0)
    
    # Cache utilization
    utilization = num_entries / max_tokens if max_tokens > 0 else 0.0
    
    # Eviction rate
    total_ops = stats.get("hits", 0) + stats.get("misses", 0)
    eviction_rate = evictions / total_ops if total_ops > 0 else 0.0
    
    # Overall efficiency score (weighted combination)
    efficiency_score = (
        0.5 * hit_rate +  # Hit rate is most important
        0.3 * (1.0 - eviction_rate) +  # Lower eviction is better
        0.2 * utilization  # Good utilization is desirable
    )
    
    return {
        "hit_rate": hit_rate,
        "utilization": utilization,
        "eviction_rate": eviction_rate,
        "efficiency_score": efficiency_score,
    }


def optimize_cache_size(
    current_stats: StatsDict,
    target_hit_rate: float = 0.8,
    max_growth_factor: float = 2.0
) -> int | None:
    """
    Suggest optimal cache size based on current statistics.
    
    Args:
        current_stats: Current cache statistics
        target_hit_rate: Target hit rate (0.0-1.0)
        max_growth_factor: Maximum growth factor
        
    Returns:
        Suggested cache size or None if no recommendation
    """
    hit_rate = current_stats.get("hit_rate", 0.0)
    current_max = current_stats.get("max_tokens", 0)
    num_entries = current_stats.get("num_entries", 0)
    
    if current_max == 0:
        return None
    
    # If hit rate is below target and cache is full, suggest increase
    if hit_rate < target_hit_rate and num_entries >= current_max * 0.9:
        # Estimate required size based on hit rate gap
        growth_factor = min(
            max_growth_factor,
            1.0 + (target_hit_rate - hit_rate)  # Linear scaling
        )
        suggested_size = int(current_max * growth_factor)
        
        # Round to nearest power of 2 for efficiency
        import math
        suggested_size = 2 ** math.ceil(math.log2(suggested_size))
        
        return suggested_size
    
    return None


def analyze_bottlenecks(stats: StatsDict, profiler_stats: StatsDict | None = None) -> list[str]:
    """
    Analyze performance bottlenecks.
    
    Args:
        stats: Cache statistics
        profiler_stats: Optional profiler statistics
        
    Returns:
        List of bottleneck descriptions
    """
    bottlenecks = []
    
    # Low hit rate
    hit_rate = stats.get("hit_rate", 0.0)
    if hit_rate < 0.3:
        bottlenecks.append(
            f"Low hit rate ({hit_rate:.2%}): Consider increasing cache size "
            "or improving access patterns"
        )
    
    # High eviction rate
    evictions = stats.get("evictions", 0)
    total_ops = stats.get("hits", 0) + stats.get("misses", 0)
    if total_ops > 0:
        eviction_rate = evictions / total_ops
        if eviction_rate > 0.1:
            bottlenecks.append(
                f"High eviction rate ({eviction_rate:.2%}): Cache may be too small "
                "for workload"
            )
    
    # Profiler-based bottlenecks
    if profiler_stats:
        avg_put_time = profiler_stats.get("avg_time", {}).get("put", 0)
        avg_get_time = profiler_stats.get("avg_time", {}).get("get", 0)
        
        if avg_put_time > 100:  # ms
            bottlenecks.append(
                f"Slow put operations ({avg_put_time:.2f}ms): Consider "
                "disabling compression/quantization"
            )
        
        if avg_get_time > 10:  # ms
            bottlenecks.append(
                f"Slow get operations ({avg_get_time:.2f}ms): Consider "
                "optimizing storage access"
            )
    
    return bottlenecks


def benchmark_cache_operations(
    cache: Any,
    num_operations: int = 1000,
    warmup_operations: int = 100
) -> StatsDict:
    """
    Benchmark cache operations.
    
    Args:
        cache: Cache instance
        num_operations: Number of benchmark operations
        warmup_operations: Number of warmup operations
        
    Returns:
        Dictionary with benchmark results
    """
    import torch
    from kv_cache.testing import create_test_tensors
    
    device = cache.device
    num_heads = cache.config.num_heads
    head_dim = cache.config.head_dim
    
    # Warmup
    for i in range(warmup_operations):
        key, value = create_test_tensors(1, num_heads, 128, head_dim, device)
        cache.put(i % cache.config.max_tokens, key, value)
    
    # Benchmark
    put_times = []
    get_times = []
    
    start_time = time.perf_counter()
    
    for i in range(num_operations):
        # Put benchmark
        key, value = create_test_tensors(1, num_heads, 128, head_dim, device)
        put_start = time.perf_counter()
        cache.put(i % cache.config.max_tokens, key, value)
        put_times.append((time.perf_counter() - put_start) * 1000)
        
        # Get benchmark
        get_start = time.perf_counter()
        cache.get(i % cache.config.max_tokens)
        get_times.append((time.perf_counter() - get_start) * 1000)
    
    total_time = time.perf_counter() - start_time
    
    return {
        "num_operations": num_operations,
        "total_time_seconds": total_time,
        "throughput_ops_per_sec": calculate_throughput(num_operations * 2, total_time),
        "avg_put_time_ms": sum(put_times) / len(put_times) if put_times else 0,
        "avg_get_time_ms": sum(get_times) / len(get_times) if get_times else 0,
        "min_put_time_ms": min(put_times) if put_times else 0,
        "max_put_time_ms": max(put_times) if put_times else 0,
        "min_get_time_ms": min(get_times) if get_times else 0,
        "max_get_time_ms": max(get_times) if get_times else 0,
    }

