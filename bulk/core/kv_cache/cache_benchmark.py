"""
Cache benchmarking utilities.

Provides benchmarking tools for cache performance evaluation.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Callable, Optional
import torch

from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


class CacheBenchmark:
    """
    Cache benchmark suite.
    
    Provides comprehensive benchmarking capabilities.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache benchmark.
        
        Args:
            cache: Cache instance to benchmark
        """
        self.cache = cache
        self.results: List[Dict[str, Any]] = []
    
    def benchmark_get(
        self,
        num_operations: int = 1000,
        positions: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark get operations.
        
        Args:
            num_operations: Number of operations to benchmark
            positions: Optional list of positions (None = random)
            
        Returns:
            Dictionary with benchmark results
        """
        import random
        
        if positions is None:
            positions = list(range(self.cache.config.max_tokens))
        
        times = []
        hits = 0
        misses = 0
        
        start = time.time()
        
        for _ in range(num_operations):
            pos = random.choice(positions)
            
            op_start = time.time()
            result = self.cache.get(pos)
            op_time = time.time() - op_start
            
            times.append(op_time)
            
            if result is not None:
                hits += 1
            else:
                misses += 1
        
        total_time = time.time() - start
        
        return {
            "operation": "get",
            "num_operations": num_operations,
            "total_time_s": total_time,
            "avg_time_ms": (total_time / num_operations) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "throughput_ops_per_sec": num_operations / total_time,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / num_operations
        }
    
    def benchmark_put(
        self,
        num_operations: int = 1000,
        key_shape: tuple = (32, 128),
        value_shape: tuple = (32, 128),
        dtype: torch.dtype = torch.float16
    ) -> Dict[str, Any]:
        """
        Benchmark put operations.
        
        Args:
            num_operations: Number of operations to benchmark
            key_shape: Shape of key tensors
            value_shape: Shape of value tensors
            dtype: Tensor dtype
            
        Returns:
            Dictionary with benchmark results
        """
        device = self.cache.device
        times = []
        
        start = time.time()
        
        for i in range(num_operations):
            key = torch.randn(key_shape, dtype=dtype, device=device)
            value = torch.randn(value_shape, dtype=dtype, device=device)
            
            op_start = time.time()
            self.cache.put(i, key, value)
            op_time = time.time() - op_start
            
            times.append(op_time)
        
        total_time = time.time() - start
        
        return {
            "operation": "put",
            "num_operations": num_operations,
            "total_time_s": total_time,
            "avg_time_ms": (total_time / num_operations) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "throughput_ops_per_sec": num_operations / total_time,
            "key_shape": key_shape,
            "value_shape": value_shape,
            "dtype": str(dtype)
        }
    
    def benchmark_forward(
        self,
        num_operations: int = 1000,
        key_shape: tuple = (32, 128),
        value_shape: tuple = (32, 128),
        dtype: torch.dtype = torch.float16
    ) -> Dict[str, Any]:
        """
        Benchmark forward operations (get or put).
        
        Args:
            num_operations: Number of operations to benchmark
            key_shape: Shape of key tensors
            value_shape: Shape of value tensors
            dtype: Tensor dtype
            
        Returns:
            Dictionary with benchmark results
        """
        device = self.cache.device
        times = []
        hits = 0
        misses = 0
        
        start = time.time()
        
        for i in range(num_operations):
            key = torch.randn(key_shape, dtype=dtype, device=device)
            value = torch.randn(value_shape, dtype=dtype, device=device)
            
            op_start = time.time()
            _, _, info = self.cache.forward(key, value, cache_position=i)
            op_time = time.time() - op_start
            
            times.append(op_time)
            
            if info.get("cached", False):
                hits += 1
            else:
                misses += 1
        
        total_time = time.time() - start
        
        return {
            "operation": "forward",
            "num_operations": num_operations,
            "total_time_s": total_time,
            "avg_time_ms": (total_time / num_operations) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "throughput_ops_per_sec": num_operations / total_time,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / num_operations
        }
    
    def run_full_benchmark(
        self,
        num_operations: int = 1000
    ) -> Dict[str, Any]:
        """
        Run full benchmark suite.
        
        Args:
            num_operations: Number of operations per benchmark
            
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Running full benchmark suite...")
        
        results = {
            "timestamp": time.time(),
            "cache_config": {
                "max_tokens": self.cache.config.max_tokens,
                "strategy": self.cache.config.cache_strategy.value,
                "use_quantization": self.cache.config.use_quantization,
                "use_compression": self.cache.config.use_compression
            }
        }
        
        # Benchmark get
        logger.info("Benchmarking get operations...")
        results["get"] = self.benchmark_get(num_operations)
        
        # Benchmark put
        logger.info("Benchmarking put operations...")
        results["put"] = self.benchmark_put(num_operations)
        
        # Benchmark forward
        logger.info("Benchmarking forward operations...")
        results["forward"] = self.benchmark_forward(num_operations)
        
        # Cache stats
        results["final_stats"] = self.cache.get_stats()
        
        self.results.append(results)
        return results
    
    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        num_operations: int = 500
    ) -> Dict[str, Any]:
        """
        Compare different cache configurations.
        
        Args:
            configs: List of configuration dictionaries
            num_operations: Number of operations per config
            
        Returns:
            Dictionary with comparison results
        """
        from kv_cache import KVCacheConfig, BaseKVCache
        
        comparison = {
            "timestamp": time.time(),
            "configs": [],
            "results": []
        }
        
        for i, config_dict in enumerate(configs):
            logger.info(f"Testing configuration {i+1}/{len(configs)}...")
            
            # Create cache with this config
            config = KVCacheConfig(**config_dict)
            test_cache = BaseKVCache(config)
            
            # Benchmark
            benchmark = CacheBenchmark(test_cache)
            result = benchmark.run_full_benchmark(num_operations)
            
            comparison["configs"].append(config_dict)
            comparison["results"].append(result)
        
        return comparison

