"""
Example usage patterns for KV Cache.

Demonstrates common use cases and best practices.
"""
from __future__ import annotations

import time

import torch

from kv_cache import (
    KVCacheConfig, CacheStrategy, CacheMode,
    BaseKVCache, AdaptiveKVCache, PagedKVCache,
    create_cache_from_config
)
from kv_cache.constants import DEFAULT_MAX_TOKENS
from kv_cache.helpers import format_cache_info, get_cache_recommendations
from kv_cache.monitoring import CacheMonitor
from kv_cache.persistence import CachePersistence


def example_basic_usage() -> None:
    """
    Basic cache usage example.
    
    Demonstrates basic KV cache operations.
    """
    print("=== Basic Cache Usage ===")
    
    # Create configuration
    config = KVCacheConfig(
        max_tokens=DEFAULT_MAX_TOKENS,
        cache_strategy=CacheStrategy.ADAPTIVE,
        use_quantization=True,
        use_compression=True,
    )
    
    # Create cache
    cache = BaseKVCache(config)
    
    # Create sample data
    device = cache.device
    key = torch.randn(1, 8, 128, 64, device=device)
    value = torch.randn(1, 8, 128, 64, device=device)
    
    # Use cache
    cached_key, cached_value, info = cache.forward(key, value, cache_position=0)
    
    print(f"Cached: {info.get('cached', False)}")
    stats = cache.get_stats()
    print(f"Stats: {format_cache_info(stats)}")
    
    # Get recommendations
    recommendations = get_cache_recommendations(stats)
    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")


def example_adaptive_cache() -> None:
    """
    Adaptive cache usage example.
    
    Demonstrates adaptive cache behavior with automatic optimization.
    """
    print("\n=== Adaptive Cache ===")
    
    config = KVCacheConfig(
        max_tokens=2048,
        cache_strategy=CacheStrategy.ADAPTIVE,
        adaptive_compression=True,
        adaptive_quantization=True,
    )
    
    cache = AdaptiveKVCache(config)
    device = cache.device
    
    # Use cache (will auto-adapt)
    for i in range(100):
        key = torch.randn(1, 8, 64, 64, device=device)
        value = torch.randn(1, 8, 64, 64, device=device)
        cache.forward(key, value, cache_position=i)
    
    # Manual adaptation
    cache.adapt({"hit_rate": 0.7, "memory_usage": 0.85})
    
    stats = cache.get_stats()
    print(f"Adaptive stats: {format_cache_info(stats)}")


def example_paged_cache() -> None:
    """
    Paged cache usage example.
    
    Demonstrates paged cache for efficient memory management.
    """
    print("\n=== Paged Cache ===")
    
    config = KVCacheConfig(
        max_tokens=8192,
        cache_strategy=CacheStrategy.PAGED,
        block_size=256,
    )
    
    cache = PagedKVCache(config)
    device = cache.device
    
    # Store entries
    for i in range(100):
        key = torch.randn(1, 8, 128, 64, device=device)
        value = torch.randn(1, 8, 128, 64, device=device)
        cache.put(i, key, value)
    
    # Get page
    page = cache.get_page(page_id=0)
    page_stats = cache.get_page_stats()
    
    print(f"Page stats: {format_cache_info(page_stats)}")


def example_with_profiling() -> None:
    """
    Cache usage with profiling enabled.
    
    Demonstrates performance profiling capabilities.
    """
    print("\n=== Cache with Profiling ===")
    
    config = KVCacheConfig(
        max_tokens=2048,
        enable_profiling=True,  # Enable profiling
    )
    
    cache = BaseKVCache(config)
    device = cache.device
    
    # Run operations
    for i in range(50):
        key = torch.randn(1, 8, 128, 64, device=device)
        value = torch.randn(1, 8, 128, 64, device=device)
        cache.put(i, key, value)
        cache.get(i)
    
    # View profiling stats
    cache.profiler.print_stats()


def example_monitoring() -> None:
    """
    Cache usage with monitoring.
    
    Demonstrates real-time monitoring and metrics collection.
    """
    print("\n=== Cache with Monitoring ===")
    
    config = KVCacheConfig(max_tokens=2048)
    cache = BaseKVCache(config)
    monitor = CacheMonitor()
    device = cache.device
    
    # Simulate operations
    for i in range(100):
        start = time.time()
        key = torch.randn(1, 8, 64, 64, device=device)
        value = torch.randn(1, 8, 64, 64, device=device)
        cache.put(i, key, value)
        operation_time = time.time() - start
        
        monitor.record_operation(operation_time)
        
        if i % 10 == 0:
            stats = cache.get_stats()
            metrics = monitor.update_metrics(stats)
            print(
                f"Hit rate: {metrics.hit_rate:.2%}, "
                f"Throughput: {metrics.throughput_ops_per_sec:.2f} ops/s"
            )


def example_persistence() -> None:
    """
    Cache persistence example.
    
    Demonstrates saving and loading cache state.
    """
    print("\n=== Cache Persistence ===")
    
    config = KVCacheConfig(
        max_tokens=2048,
        enable_persistence=True,
        persistence_path="./cache_checkpoints"
    )
    
    cache = BaseKVCache(config)
    device = cache.device
    
    # Store some data
    for i in range(10):
        key = torch.randn(1, 8, 128, 64, device=device)
        value = torch.randn(1, 8, 128, 64, device=device)
        cache.put(i, key, value)
    
    # Save cache
    persistence = CachePersistence("./cache_checkpoints")
    persistence.save_cache(cache, "cache_state.pkl")
    
    # Create new cache and load
    new_cache = BaseKVCache(config)
    persistence.load_cache(new_cache, "cache_state.pkl")
    
    print("Cache saved and loaded successfully")


def example_from_config_dict() -> None:
    """
    Example creating cache from configuration dictionary.
    
    Demonstrates the helper function for easy cache creation.
    """
    print("\n=== Cache from Config Dict ===")
    
    config_dict = {
        "max_tokens": 4096,
        "cache_strategy": "adaptive",
        "use_compression": True,
        "use_quantization": True,
    }
    
    cache = create_cache_from_config(config_dict)
    print(f"Created cache with {cache.config.max_tokens} max tokens")
    print(f"Strategy: {cache.config.cache_strategy.value}")


def run_all_examples() -> None:
    """Run all examples if CUDA is available."""
    if torch.cuda.is_available():
        example_basic_usage()
        example_adaptive_cache()
        example_paged_cache()
        example_with_profiling()
        example_monitoring()
        example_persistence()
        example_from_config_dict()
    else:
        print("CUDA not available - examples require GPU")
        print("Note: Some examples can be adapted to run on CPU")


if __name__ == "__main__":
    run_all_examples()

