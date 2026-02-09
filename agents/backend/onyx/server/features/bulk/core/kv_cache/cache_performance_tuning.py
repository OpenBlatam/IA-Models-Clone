"""
Performance tuning system for KV cache.

This module provides automatic and manual performance tuning capabilities.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class TuningTarget(Enum):
    """Performance tuning targets."""
    HIT_RATE = "hit_rate"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ALL = "all"


@dataclass
class TuningParameter:
    """A tunable parameter."""
    name: str
    current_value: Any
    min_value: Any
    max_value: Any
    step: Any
    description: str


@dataclass
class TuningRecommendation:
    """Performance tuning recommendation."""
    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    expected_improvement: float  # Percentage
    confidence: float


@dataclass
class TuningResult:
    """Result of tuning operation."""
    parameter: str
    old_value: Any
    new_value: Any
    actual_improvement: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class CachePerformanceTuner:
    """Performance tuner for cache."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._tunable_parameters: Dict[str, TuningParameter] = {}
        self._tuning_history: List[TuningResult] = []
        self._lock = threading.Lock()
        
    def register_parameter(
        self,
        name: str,
        current_value: Any,
        min_value: Any,
        max_value: Any,
        step: Any,
        description: str
    ) -> None:
        """Register a tunable parameter."""
        param = TuningParameter(
            name=name,
            current_value=current_value,
            min_value=min_value,
            max_value=max_value,
            step=step,
            description=description
        )
        
        with self._lock:
            self._tunable_parameters[name] = param
            
    def get_parameters(self) -> List[TuningParameter]:
        """Get all tunable parameters."""
        return list(self._tunable_parameters.values())
        
    def analyze_performance(self, target: TuningTarget = TuningTarget.ALL) -> List[TuningRecommendation]:
        """Analyze performance and generate recommendations."""
        recommendations = []
        
        # Analyze hit rate
        if target in [TuningTarget.HIT_RATE, TuningTarget.ALL]:
            if hasattr(self.cache, 'stats'):
                hit_rate = getattr(self.cache.stats, 'hit_rate', 0.0)
                
                if hit_rate < 0.7:
                    # Recommend increasing cache size
                    if 'max_size' in self._tunable_parameters:
                        param = self._tunable_parameters['max_size']
                        current = param.current_value
                        recommended = min(
                            param.max_value,
                            int(current * 1.5)  # Increase by 50%
                        )
                        
                        recommendations.append(TuningRecommendation(
                            parameter='max_size',
                            current_value=current,
                            recommended_value=recommended,
                            reason=f'Low hit rate ({hit_rate:.2%}), increase cache size',
                            expected_improvement=15.0,
                            confidence=0.8
                        ))
                        
        # Analyze memory usage
        if target in [TuningTarget.MEMORY_USAGE, TuningTarget.ALL]:
            if hasattr(self.cache, '_cache'):
                cache_size = len(self.cache._cache)
                if hasattr(self.cache, 'max_size'):
                    max_size = self.cache.max_size
                    usage_ratio = cache_size / max_size if max_size > 0 else 0.0
                    
                    if usage_ratio > 0.9:
                        # Recommend enabling compression or eviction
                        recommendations.append(TuningRecommendation(
                            parameter='enable_compression',
                            current_value=False,
                            recommended_value=True,
                            reason=f'High memory usage ({usage_ratio:.2%}), enable compression',
                            expected_improvement=30.0,
                            confidence=0.7
                        ))
                        
        return recommendations
        
    def apply_recommendation(self, recommendation: TuningRecommendation) -> TuningResult:
        """Apply a tuning recommendation."""
        param_name = recommendation.parameter
        
        if param_name not in self._tunable_parameters:
            raise ValueError(f"Parameter '{param_name}' not registered")
            
        param = self._tunable_parameters[param_name]
        old_value = param.current_value
        new_value = recommendation.recommended_value
        
        # Apply the change
        if hasattr(self.cache, param_name):
            setattr(self.cache, param_name, new_value)
            
        # Update parameter
        param.current_value = new_value
        
        result = TuningResult(
            parameter=param_name,
            old_value=old_value,
            new_value=new_value
        )
        
        with self._lock:
            self._tuning_history.append(result)
            
        return result
        
    def auto_tune(self, target: TuningTarget = TuningTarget.ALL) -> List[TuningResult]:
        """Automatically tune cache performance."""
        recommendations = self.analyze_performance(target)
        results = []
        
        for rec in recommendations:
            if rec.confidence > 0.7:  # Only apply high-confidence recommendations
                try:
                    result = self.apply_recommendation(rec)
                    results.append(result)
                except Exception as e:
                    print(f"Error applying recommendation: {e}")
                    
        return results
        
    def get_tuning_history(self) -> List[TuningResult]:
        """Get tuning history."""
        return self._tuning_history.copy()
        
    def benchmark(self, operations: int = 1000) -> Dict[str, Any]:
        """Benchmark cache performance."""
        import random
        
        # Warm up
        for i in range(100):
            self.cache.put(f"bench_{i}", f"value_{i}")
            
        # Benchmark gets
        start_time = time.time()
        for _ in range(operations):
            key = f"bench_{random.randint(0, 99)}"
            self.cache.get(key)
        get_time = time.time() - start_time
        
        # Benchmark puts
        start_time = time.time()
        for i in range(operations):
            self.cache.put(f"bench_{i}", f"value_{i}")
        put_time = time.time() - start_time
        
        return {
            'get_ops_per_sec': operations / get_time if get_time > 0 else 0,
            'put_ops_per_sec': operations / put_time if put_time > 0 else 0,
            'avg_get_latency_ms': (get_time / operations) * 1000,
            'avg_put_latency_ms': (put_time / operations) * 1000,
            'total_operations': operations * 2
        }














