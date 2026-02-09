"""
Cache performance tuning utilities.

Provides utilities for tuning cache performance.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TuningRecommendation:
    """Tuning recommendation."""
    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    impact: str  # "high", "medium", "low"


class CacheTuner:
    """
    Cache performance tuner.
    
    Provides automatic tuning recommendations.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize tuner.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.performance_history: List[Dict[str, Any]] = []
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze cache performance.
        
        Returns:
            Performance analysis
        """
        stats = self.cache.get_stats()
        
        analysis = {
            "hit_rate": stats.get("hit_rate", 0.0),
            "avg_latency_ms": stats.get("avg_latency_ms", 0.0),
            "memory_mb": stats.get("memory_mb", 0.0),
            "cache_size": stats.get("cache_size", 0),
            "eviction_count": stats.get("eviction_count", 0)
        }
        
        # Store in history
        analysis["timestamp"] = time.time()
        self.performance_history.append(analysis)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        return analysis
    
    def get_recommendations(self) -> List[TuningRecommendation]:
        """
        Get tuning recommendations.
        
        Returns:
            List of recommendations
        """
        stats = self.cache.get_stats()
        config = self.cache.config
        recommendations = []
        
        hit_rate = stats.get("hit_rate", 0.0)
        eviction_count = stats.get("eviction_count", 0)
        memory_mb = stats.get("memory_mb", 0.0)
        
        # Low hit rate
        if hit_rate < 0.7:
            recommendations.append(TuningRecommendation(
                parameter="max_tokens",
                current_value=config.max_tokens,
                recommended_value=config.max_tokens * 2,
                reason="Low hit rate indicates cache is too small",
                impact="high"
            ))
        
        # High eviction rate
        if eviction_count > 1000:
            recommendations.append(TuningRecommendation(
                parameter="cache_strategy",
                current_value=config.cache_strategy.value,
                recommended_value="ADAPTIVE",
                reason="High eviction rate suggests better strategy needed",
                impact="medium"
            ))
        
        # Memory pressure
        if memory_mb > 5000:
            recommendations.append(TuningRecommendation(
                parameter="use_compression",
                current_value=config.use_compression,
                recommended_value=True,
                reason="High memory usage, compression can help",
                impact="high"
            ))
        
        return recommendations
    
    def auto_tune(self, apply: bool = False) -> Dict[str, Any]:
        """
        Automatically tune cache.
        
        Args:
            apply: Whether to apply recommendations
            
        Returns:
            Tuning results
        """
        recommendations = self.get_recommendations()
        
        results = {
            "recommendations": recommendations,
            "applied": []
        }
        
        if apply:
            for rec in recommendations:
                if rec.impact == "high":
                    # Apply high-impact recommendations
                    try:
                        if rec.parameter == "max_tokens":
                            self.cache.config.max_tokens = rec.recommended_value
                        elif rec.parameter == "use_compression":
                            self.cache.config.use_compression = rec.recommended_value
                        
                        results["applied"].append(rec.parameter)
                        logger.info(f"Applied tuning: {rec.parameter} = {rec.recommended_value}")
                    except Exception as e:
                        logger.warning(f"Failed to apply tuning {rec.parameter}: {e}")
        
        return results
    
    def benchmark_configurations(
        self,
        configurations: List[Dict[str, Any]],
        test_fn: Optional[Callable] = None,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark different configurations.
        
        Args:
            configurations: List of configuration dicts
            test_fn: Optional test function
            iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for i, config_dict in enumerate(configurations):
            config_name = config_dict.get("name", f"config_{i}")
            
            # Create cache with config
            from kv_cache import KVCacheConfig, BaseKVCache
            config = KVCacheConfig(**{k: v for k, v in config_dict.items() if k != "name"})
            test_cache = BaseKVCache(config)
            
            # Run benchmark
            if test_fn:
                result = test_fn(test_cache)
            else:
                # Default benchmark
                start_time = time.time()
                for _ in range(iterations):
                    test_cache.get(0)
                    test_cache.put(0, (None, None))
                end_time = time.time()
                
                result = {
                    "iterations": iterations,
                    "time_seconds": end_time - start_time,
                    "ops_per_sec": iterations / (end_time - start_time)
                }
            
            results[config_name] = result
        
        return results


class CacheProfiler:
    """
    Advanced cache profiler.
    
    Provides detailed profiling of cache operations.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize profiler.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
    
    def profile_operation(self, operation_name: str):
        """
        Profile decorator for operations.
        
        Args:
            operation_name: Name of operation
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    
                    if operation_name not in self.operation_times:
                        self.operation_times[operation_name] = []
                        self.operation_counts[operation_name] = 0
                    
                    self.operation_times[operation_name].append(elapsed)
                    self.operation_counts[operation_name] += 1
                    
                    # Keep only recent times
                    if len(self.operation_times[operation_name]) > 1000:
                        self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
            
            return wrapper
        return decorator
    
    def get_profile_report(self) -> Dict[str, Any]:
        """
        Get profiling report.
        
        Returns:
            Profile report
        """
        report = {}
        
        for op_name, times in self.operation_times.items():
            if times:
                report[op_name] = {
                    "count": self.operation_counts[op_name],
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "p50": sorted(times)[len(times) // 2] if times else 0,
                    "p95": sorted(times)[int(len(times) * 0.95)] if times else 0,
                    "p99": sorted(times)[int(len(times) * 0.99)] if times else 0
                }
        
        return report

