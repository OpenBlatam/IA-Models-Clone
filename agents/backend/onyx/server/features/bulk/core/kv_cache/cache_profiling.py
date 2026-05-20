"""
Advanced profiling and performance analysis for KV cache.

This module provides deep profiling capabilities, performance analysis,
and optimization recommendations based on runtime behavior.
"""

import time
import cProfile
import pstats
import io
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import tracemalloc
import sys

from .constants import DEFAULT_STATS_INTERVAL


class ProfilingLevel(Enum):
    """Profiling granularity levels."""
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"
    DEEP = "deep"


class PerformanceMetric(Enum):
    """Performance metrics to track."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    OPERATION_LATENCY = "operation_latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CONCURRENCY_LEVEL = "concurrency_level"


@dataclass
class ProfilingResult:
    """Results from profiling analysis."""
    timestamp: float
    level: ProfilingLevel
    metrics: Dict[str, Any]
    hotspots: List[Dict[str, Any]]
    recommendations: List[str]
    memory_profile: Optional[Dict[str, Any]] = None
    cpu_profile: Optional[Dict[str, Any]] = None
    duration: float = 0.0


@dataclass
class FunctionProfile:
    """Profile information for a function."""
    name: str
    call_count: int
    total_time: float
    cumulative_time: float
    per_call_time: float
    memory_usage: int = 0


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    current: int
    peak: int
    diff: int = 0
    top_allocations: List[Tuple[str, int]] = field(default_factory=list)


class CacheProfiler:
    """Advanced profiler for cache operations."""
    
    def __init__(
        self,
        profiling_level: ProfilingLevel = ProfilingLevel.BASIC,
        enable_memory_profiling: bool = True,
        enable_cpu_profiling: bool = True,
        sample_rate: float = 1.0
    ):
        self.profiling_level = profiling_level
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_cpu_profiling = enable_cpu_profiling
        self.sample_rate = sample_rate
        
        self._cpu_profiler: Optional[cProfile.Profile] = None
        self._memory_tracemalloc: bool = False
        self._profiling_active: bool = False
        self._lock = threading.Lock()
        
        self._function_profiles: Dict[str, FunctionProfile] = {}
        self._operation_timings: Dict[str, List[float]] = defaultdict(list)
        self._memory_snapshots: List[MemorySnapshot] = []
        
    def start_profiling(self) -> None:
        """Start profiling session."""
        with self._lock:
            if self._profiling_active:
                return
                
            self._profiling_active = True
            
            if self.enable_cpu_profiling:
                self._cpu_profiler = cProfile.Profile()
                self._cpu_profiler.enable()
                
            if self.enable_memory_profiling:
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                    self._memory_tracemalloc = True
                    
    def stop_profiling(self) -> ProfilingResult:
        """Stop profiling and return results."""
        with self._lock:
            if not self._profiling_active:
                return ProfilingResult(
                    timestamp=time.time(),
                    level=self.profiling_level,
                    metrics={},
                    hotspots=[],
                    recommendations=[]
                )
                
            start_time = time.time()
            metrics = {}
            hotspots = []
            recommendations = []
            memory_profile = None
            cpu_profile = None
            
            # Stop CPU profiling
            if self._cpu_profiler:
                self._cpu_profiler.disable()
                cpu_profile = self._get_cpu_profile()
                
            # Stop memory profiling
            if self._memory_tracemalloc and tracemalloc.is_tracing():
                memory_profile = self._get_memory_profile()
                tracemalloc.stop()
                self._memory_tracemalloc = False
                
            # Collect metrics
            metrics = self._collect_metrics()
            
            # Find hotspots
            hotspots = self._find_hotspots()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, hotspots)
            
            duration = time.time() - start_time
            
            self._profiling_active = False
            
            return ProfilingResult(
                timestamp=time.time(),
                level=self.profiling_level,
                metrics=metrics,
                hotspots=hotspots,
                recommendations=recommendations,
                memory_profile=memory_profile,
                cpu_profile=cpu_profile,
                duration=duration
            )
            
    def _get_cpu_profile(self) -> Dict[str, Any]:
        """Extract CPU profiling data."""
        if not self._cpu_profiler:
            return {}
            
        stream = io.StringIO()
        stats = pstats.Stats(self._cpu_profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profile_data = stream.getvalue()
        
        # Parse profile data
        functions = []
        for line in profile_data.split('\n')[5:25]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        functions.append({
                            'ncalls': parts[0],
                            'tottime': float(parts[1]),
                            'cumtime': float(parts[2]),
                            'function': ' '.join(parts[4:])
                        })
                    except (ValueError, IndexError):
                        continue
                        
        return {
            'total_calls': stats.total_calls,
            'total_time': stats.total_tt,
            'functions': functions,
            'raw': profile_data
        }
        
    def _get_memory_profile(self) -> Dict[str, Any]:
        """Extract memory profiling data."""
        if not tracemalloc.is_tracing():
            return {}
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        top_allocations = []
        for stat in top_stats[:20]:  # Top 20 allocations
            top_allocations.append({
                'filename': stat.traceback[0].filename if stat.traceback else 'unknown',
                'size': stat.size,
                'count': stat.count
            })
            
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            'current_memory': current,
            'peak_memory': peak,
            'top_allocations': top_allocations,
            'total_traced': len(snapshot.traces)
        }
        
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {}
        
        # Calculate average operation latencies
        for op, timings in self._operation_timings.items():
            if timings:
                metrics[f'{op}_avg_latency'] = sum(timings) / len(timings)
                metrics[f'{op}_max_latency'] = max(timings)
                metrics[f'{op}_min_latency'] = min(timings)
                
        # Function call statistics
        if self._function_profiles:
            total_calls = sum(fp.call_count for fp in self._function_profiles.values())
            total_time = sum(fp.total_time for fp in self._function_profiles.values())
            
            metrics['total_function_calls'] = total_calls
            metrics['total_function_time'] = total_time
            
        return metrics
        
    def _find_hotspots(self) -> List[Dict[str, Any]]:
        """Identify performance hotspots."""
        hotspots = []
        
        # Find slow operations
        for op, timings in self._operation_timings.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                if avg_time > 0.1:  # Threshold: 100ms
                    hotspots.append({
                        'type': 'slow_operation',
                        'operation': op,
                        'avg_time': avg_time,
                        'max_time': max(timings),
                        'call_count': len(timings)
                    })
                    
        # Find frequently called functions
        for name, profile in self._function_profiles.items():
            if profile.call_count > 1000 or profile.total_time > 1.0:
                hotspots.append({
                    'type': 'hot_function',
                    'function': name,
                    'call_count': profile.call_count,
                    'total_time': profile.total_time,
                    'avg_time': profile.per_call_time
                })
                
        return sorted(hotspots, key=lambda x: x.get('total_time', x.get('avg_time', 0)), reverse=True)
        
    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        hotspots: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check for slow operations
        for hotspot in hotspots:
            if hotspot['type'] == 'slow_operation':
                recommendations.append(
                    f"Operation '{hotspot['operation']}' is slow (avg: {hotspot['avg_time']:.3f}s). "
                    f"Consider optimization or caching."
                )
                
        # Check for memory issues
        if self.enable_memory_profiling and self._memory_snapshots:
            recent_snapshot = self._memory_snapshots[-1]
            if recent_snapshot.peak > 1024 * 1024 * 1024:  # > 1GB
                recommendations.append(
                    f"High memory usage detected (peak: {recent_snapshot.peak / 1024 / 1024:.2f}MB). "
                    f"Consider implementing eviction strategies or compression."
                )
                
        # Check for high function call counts
        for hotspot in hotspots:
            if hotspot['type'] == 'hot_function' and hotspot['call_count'] > 10000:
                recommendations.append(
                    f"Function '{hotspot['function']}' is called frequently ({hotspot['call_count']} times). "
                    f"Consider memoization or caching."
                )
                
        return recommendations
        
    def profile_operation(self, operation_name: str) -> Callable:
        """Decorator to profile a specific operation."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                if not self._profiling_active or not self._should_sample():
                    return func(*args, **kwargs)
                    
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._operation_timings[operation_name].append(duration)
                    
            return wrapper
        return decorator
        
    def _should_sample(self) -> bool:
        """Determine if current operation should be sampled."""
        import random
        return random.random() < self.sample_rate
        
    def take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        if not tracemalloc.is_tracing():
            return MemorySnapshot(
                timestamp=time.time(),
                current=0,
                peak=0
            )
            
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        top_allocations = [
            (stat.traceback[0].filename if stat.traceback else 'unknown', stat.size)
            for stat in top_stats[:10]
        ]
        
        prev_snapshot = self._memory_snapshots[-1] if self._memory_snapshots else None
        diff = current - prev_snapshot.current if prev_snapshot else 0
        
        memory_snapshot = MemorySnapshot(
            timestamp=time.time(),
            current=current,
            peak=peak,
            diff=diff,
            top_allocations=top_allocations
        )
        
        self._memory_snapshots.append(memory_snapshot)
        return memory_snapshot
        
    def get_function_profile(self, function_name: str) -> Optional[FunctionProfile]:
        """Get profiling information for a specific function."""
        return self._function_profiles.get(function_name)
        
    def reset(self) -> None:
        """Reset profiling state."""
        with self._lock:
            self._function_profiles.clear()
            self._operation_timings.clear()
            self._memory_snapshots.clear()
            if self._cpu_profiler:
                self._cpu_profiler = None


class CachePerformanceAnalyzer:
    """Analyzes cache performance and provides insights."""
    
    def __init__(self, profiler: CacheProfiler):
        self.profiler = profiler
        self._analysis_history: List[ProfilingResult] = []
        
    def analyze(self, duration: float = 60.0) -> ProfilingResult:
        """Run performance analysis for specified duration."""
        self.profiler.start_profiling()
        time.sleep(duration)
        result = self.profiler.stop_profiling()
        self._analysis_history.append(result)
        return result
        
    def compare_analyses(
        self,
        baseline_index: int = -2,
        comparison_index: int = -1
    ) -> Dict[str, Any]:
        """Compare two profiling results."""
        if len(self._analysis_history) < 2:
            return {}
            
        baseline = self._analysis_history[baseline_index]
        comparison = self._analysis_history[comparison_index]
        
        comparison_results = {}
        
        # Compare metrics
        for metric in set(baseline.metrics.keys()) | set(comparison.metrics.keys()):
            baseline_val = baseline.metrics.get(metric, 0)
            comparison_val = comparison.metrics.get(metric, 0)
            
            if baseline_val > 0:
                change = ((comparison_val - baseline_val) / baseline_val) * 100
                comparison_results[metric] = {
                    'baseline': baseline_val,
                    'comparison': comparison_val,
                    'change_percent': change
                }
                
        return comparison_results
        
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self._analysis_history) < 3:
            return {}
            
        trends = {}
        
        # Analyze metric trends
        all_metrics = set()
        for result in self._analysis_history:
            all_metrics.update(result.metrics.keys())
            
        for metric in all_metrics:
            values = [r.metrics.get(metric, 0) for r in self._analysis_history]
            if values:
                trends[metric] = {
                    'first': values[0],
                    'last': values[-1],
                    'avg': sum(values) / len(values),
                    'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
                }
                
        return trends
















