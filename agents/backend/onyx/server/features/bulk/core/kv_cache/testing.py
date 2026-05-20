"""
Testing utilities for KV Cache.

Provides helpers for testing and validation.
"""
from __future__ import annotations

import logging
from typing import Callable

import torch

from kv_cache import BaseKVCache, KVCacheConfig, CacheStrategy
from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


def create_test_cache(
    max_tokens: int = 1024,
    num_heads: int = 8,
    head_dim: int = 64,
    strategy: CacheStrategy = CacheStrategy.LRU
) -> BaseKVCache:
    """
    Create a test cache with default settings.
    
    Args:
        max_tokens: Maximum cache entries
        num_heads: Number of attention heads
        head_dim: Head dimension
        strategy: Cache eviction strategy
        
    Returns:
        Configured BaseKVCache instance
    """
    config = KVCacheConfig(
        max_tokens=max_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        cache_strategy=strategy,
        use_compression=False,  # Disable for testing
        use_quantization=False,  # Disable for testing
    )
    return BaseKVCache(config)


def create_test_tensors(
    batch_size: int = 1,
    num_heads: int = 8,
    seq_len: int = 128,
    head_dim: int = 64,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float16
) -> TensorPair:
    """
    Create test key-value tensors.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        device: Target device (None = CPU)
        dtype: Tensor dtype
        
    Returns:
        Tuple of (key, value) tensors
    """
    if device is None:
        device = torch.device("cpu")
    
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    return key, value


def benchmark_cache_operation(
    cache: BaseKVCache,
    operation: Callable[[int, torch.Tensor, torch.Tensor], None],
    num_operations: int = 100,
    batch_size: int = 1,
    seq_len: int = 128
) -> dict[str, float]:
    """
    Benchmark a cache operation.
    
    Args:
        cache: Cache instance
        operation: Operation to benchmark (put, get, etc.)
        num_operations: Number of operations to perform
        batch_size: Batch size for tensors
        seq_len: Sequence length
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    device = cache.device
    num_heads = cache.config.num_heads
    head_dim = cache.config.head_dim
    
    # Warmup
    key, value = create_test_tensors(
        batch_size, num_heads, seq_len, head_dim, device
    )
    for i in range(10):
        operation(i, key, value)
    
    # Benchmark
    start_time = time.time()
    for i in range(num_operations):
        key, value = create_test_tensors(
            batch_size, num_heads, seq_len, head_dim, device
        )
        operation(i % cache.config.max_tokens, key, value)
    
    elapsed = time.time() - start_time
    ops_per_sec = num_operations / elapsed
    
    return {
        "total_time": elapsed,
        "ops_per_sec": ops_per_sec,
        "avg_time_per_op": elapsed / num_operations,
    }


def validate_cache_integrity(cache: BaseKVCache) -> tuple[bool, str | None]:
    """
    Validate cache integrity.
    
    Args:
        cache: Cache instance to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    stats = cache.get_stats()
    
    # Check basic stats
    num_entries = stats.get("num_entries", 0)
    max_tokens = stats.get("max_tokens", 0)
    
    if num_entries < 0:
        return False, f"Invalid num_entries: {num_entries}"
    
    if num_entries > max_tokens:
        return False, f"Cache overflow: {num_entries} > {max_tokens}"
    
    # Check hit rate is valid
    hit_rate = stats.get("hit_rate", 0.0)
    if not 0.0 <= hit_rate <= 1.0:
        return False, f"Invalid hit_rate: {hit_rate}"
    
    return True, None


def test_cache_basic_operations(cache: BaseKVCache) -> dict[str, bool]:
    """
    Test basic cache operations.
    
    Args:
        cache: Cache instance to test
        
    Returns:
        Dictionary with test results
    """
    results = {}
    device = cache.device
    num_heads = cache.config.num_heads
    head_dim = cache.config.head_dim
    
    # Test put
    try:
        key, value = create_test_tensors(1, num_heads, 128, head_dim, device)
        cache.put(0, key, value)
        results["put"] = True
    except Exception as e:
        logger.error(f"Put test failed: {e}")
        results["put"] = False
    
    # Test get
    try:
        retrieved = cache.get(0)
        results["get"] = retrieved is not None
    except Exception as e:
        logger.error(f"Get test failed: {e}")
        results["get"] = False
    
    # Test forward
    try:
        key, value = create_test_tensors(1, num_heads, 128, head_dim, device)
        _, _, info = cache.forward(key, value, cache_position=1)
        results["forward"] = "cached" in info or "hit" in info
    except Exception as e:
        logger.error(f"Forward test failed: {e}")
        results["forward"] = False
    
    # Test clear
    try:
        cache.clear()
        stats = cache.get_stats()
        results["clear"] = stats.get("num_entries", 0) == 0
    except Exception as e:
        logger.error(f"Clear test failed: {e}")
        results["clear"] = False
    
    return results


def compare_cache_strategies(
    strategies: list[CacheStrategy],
    num_operations: int = 1000,
    access_pattern: str = "random"
) -> dict[str, dict[str, float]]:
    """
    Compare different cache strategies.
    
    Args:
        strategies: List of strategies to compare
        num_operations: Number of operations
        access_pattern: Access pattern ("random", "sequential", "locality")
        
    Returns:
        Dictionary with results per strategy
    """
    results = {}
    
    for strategy in strategies:
        cache = create_test_cache(strategy=strategy)
        
        device = cache.device
        num_heads = cache.config.num_heads
        head_dim = cache.config.head_dim
        
        # Generate access pattern
        if access_pattern == "random":
            import random
            positions = [random.randint(0, cache.config.max_tokens - 1) 
                        for _ in range(num_operations)]
        elif access_pattern == "sequential":
            positions = list(range(num_operations))
        else:  # locality
            import random
            base = random.randint(0, cache.config.max_tokens - 100)
            positions = [base + random.randint(0, 99) 
                        for _ in range(num_operations)]
        
        # Run operations
        hits = 0
        for pos in positions:
            key, value = create_test_tensors(1, num_heads, 128, head_dim, device)
            cached_key, cached_value, info = cache.forward(key, value, cache_position=pos)
            if info.get("cached", False) or info.get("hit", False):
                hits += 1
        
        stats = cache.get_stats()
        results[strategy.value] = {
            "hit_rate": hits / num_operations,
            "num_entries": stats.get("num_entries", 0),
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
        }
    
    return results



