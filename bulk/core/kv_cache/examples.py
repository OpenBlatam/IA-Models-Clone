"""
Example usage patterns for KV Cache.

Demonstrates common use cases and best practices.
"""
import torch
from kv_cache import (
    KVCacheConfig, CacheStrategy, CacheMode,
    BaseKVCache, AdaptiveKVCache, PagedKVCache
)


def example_basic_usage():
    """Basic cache usage example."""
    print("=== Basic Cache Usage ===")
    
    # Create configuration
    config = KVCacheConfig(
        max_tokens=4096,
        cache_strategy=CacheStrategy.ADAPTIVE,
        use_quantization=True,
        use_compression=True,
    )
    
    # Create cache
    cache = BaseKVCache(config)
    
    # Create sample data
    key = torch.randn(1, 8, 128, 64).cuda()
    value = torch.randn(1, 8, 128, 64).cuda()
    
    # Use cache
    cached_key, cached_value, info = cache.forward(key, value, cache_position=0)
    
    print(f"Cached: {info['cached']}")
    print(f"Stats: {cache.get_stats()}")


def example_adaptive_cache():
    """Adaptive cache usage example."""
    print("\n=== Adaptive Cache ===")
    
    config = KVCacheConfig(
        max_tokens=2048,
        cache_strategy=CacheStrategy.ADAPTIVE,
        adaptive_compression=True,
        adaptive_quantization=True,
    )
    
    cache = AdaptiveKVCache(config)
    
    # Use cache (will auto-adapt)
    for i in range(100):
        key = torch.randn(1, 8, 64, 64).cuda()
        value = torch.randn(1, 8, 64, 64).cuda()
        cache.forward(key, value, cache_position=i)
    
    # Manual adaptation
    cache.adapt({"hit_rate": 0.7, "memory_usage": 0.85})
    
    print(f"Adaptive stats: {cache.get_stats()}")


def example_paged_cache():
    """Paged cache usage example."""
    print("\n=== Paged Cache ===")
    
    config = KVCacheConfig(
        max_tokens=8192,
        cache_strategy=CacheStrategy.PAGED,
        block_size=256,
    )
    
    cache = PagedKVCache(config)
    
    # Store entries
    for i in range(100):
        key = torch.randn(1, 8, 128, 64).cuda()
        value = torch.randn(1, 8, 128, 64).cuda()
        cache.put(i, key, value)
    
    # Get page
    page = cache.get_page(page_id=0)
    page_stats = cache.get_page_stats()
    
    print(f"Page stats: {page_stats}")


def example_with_profiling():
    """Cache usage with profiling enabled."""
    print("\n=== Cache with Profiling ===")
    
    config = KVCacheConfig(
        max_tokens=2048,
        enable_profiling=True,  # Enable profiling
    )
    
    cache = BaseKVCache(config)
    
    # Run operations
    for i in range(50):
        key = torch.randn(1, 8, 128, 64).cuda()
        value = torch.randn(1, 8, 128, 64).cuda()
        cache.put(i, key, value)
        cache.get(i)
    
    # View profiling stats
    cache.profiler.print_stats()


def example_monitoring():
    """Cache usage with monitoring."""
    print("\n=== Cache with Monitoring ===")
    
    from kv_cache.monitoring import CacheMonitor
    
    config = KVCacheConfig(max_tokens=2048)
    cache = BaseKVCache(config)
    monitor = CacheMonitor()
    
    # Simulate operations
    import time
    for i in range(100):
        start = time.time()
        key = torch.randn(1, 8, 64, 64).cuda()
        value = torch.randn(1, 8, 64, 64).cuda()
        cache.put(i, key, value)
        operation_time = time.time() - start
        
        monitor.record_operation(operation_time)
        
        if i % 10 == 0:
            stats = cache.get_stats()
            metrics = monitor.update_metrics(stats)
            print(f"Hit rate: {metrics.hit_rate:.2%}, Throughput: {metrics.throughput_ops_per_sec:.2f} ops/s")


def example_persistence():
    """Cache persistence example."""
    print("\n=== Cache Persistence ===")
    
    from kv_cache.persistence import CachePersistence
    
    config = KVCacheConfig(
        max_tokens=2048,
        enable_persistence=True,
        persistence_path="./cache_checkpoints"
    )
    
    cache = BaseKVCache(config)
    
    # Store some data
    for i in range(10):
        key = torch.randn(1, 8, 128, 64).cuda()
        value = torch.randn(1, 8, 128, 64).cuda()
        cache.put(i, key, value)
    
    # Save cache
    persistence = CachePersistence("./cache_checkpoints")
    persistence.save_cache(cache, "cache_state.pkl")
    
    # Create new cache and load
    new_cache = BaseKVCache(config)
    persistence.load_cache(new_cache, "cache_state.pkl")
    
    print("Cache saved and loaded successfully")


if __name__ == "__main__":
    # Run examples
    if torch.cuda.is_available():
        example_basic_usage()
        example_adaptive_cache()
        example_paged_cache()
        example_with_profiling()
        example_monitoring()
        example_persistence()
    else:
        print("CUDA not available - examples require GPU")

