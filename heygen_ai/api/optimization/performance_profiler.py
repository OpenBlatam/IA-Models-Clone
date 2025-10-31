from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import cProfile
import pstats
import io
import time
import threading
import traceback
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, NamedTuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from functools import wraps
import structlog
import json
import numpy as np
from contextlib import contextmanager
    import line_profiler
    import memory_profiler
    import py_spy
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Advanced Performance Profiler for HeyGen AI FastAPI
Real-time performance profiling with bottleneck detection and optimization recommendations.
"""


try:
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False

try:
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    HAS_PY_SPY = True
except ImportError:
    HAS_PY_SPY = False

logger = structlog.get_logger()

# =============================================================================
# Performance Profiling Types
# =============================================================================

class ProfilingLevel(Enum):
    """Performance profiling levels."""
    DISABLED = auto()
    BASIC = auto()
    DETAILED = auto()
    COMPREHENSIVE = auto()
    ULTRA_DETAILED = auto()

class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_BOUND = auto()
    IO_BOUND = auto()
    MEMORY_BOUND = auto()
    GPU_BOUND = auto()
    NETWORK_BOUND = auto()
    DATABASE_BOUND = auto()
    CACHE_MISS = auto()
    LOCK_CONTENTION = auto()
    GC_PRESSURE = auto()

class PerformanceCategory(Enum):
    """Performance measurement categories."""
    REQUEST_PROCESSING = auto()
    DATABASE_OPERATIONS = auto()
    AI_INFERENCE = auto()
    FILE_OPERATIONS = auto()
    NETWORK_REQUESTS = auto()
    CACHE_OPERATIONS = auto()
    BACKGROUND_TASKS = auto()

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    category: PerformanceCategory
    operation_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_time_ms: float
    io_operations: int
    cache_hits: int
    cache_misses: int
    error_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class BottleneckDetection:
    """Bottleneck detection result."""
    bottleneck_type: BottleneckType
    severity: float  # 0.0 to 1.0
    affected_operations: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ProfilerStatistics(NamedTuple):
    """Profiler statistics summary."""
    total_calls: int
    total_time: float
    avg_time_per_call: float
    slowest_functions: List[tuple]
    memory_peak_mb: float
    gc_collections: int

# =============================================================================
# Advanced Performance Profiler
# =============================================================================

class AdvancedPerformanceProfiler:
    """Advanced performance profiler with bottleneck detection."""
    
    def __init__(self, profiling_level: ProfilingLevel = ProfilingLevel.DETAILED):
        
    """__init__ function."""
self.profiling_level = profiling_level
        self.profiling_active = False
        self.profiler_stack: List[cProfile.Profile] = []
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=10000)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.bottlenecks: List[BottleneckDetection] = []
        
        # Profiling configuration
        self.profiling_config = self._get_profiling_config()
        
        # Real-time monitoring
        self.monitoring_interval = 1.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance thresholds
        self.performance_thresholds = {
            "slow_request_ms": 1000,
            "high_memory_mb": 500,
            "high_cpu_percent": 80,
            "cache_miss_rate": 0.3,
            "error_rate": 0.05
        }
        
        # Function call tracking
        self.function_calls: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "max_time": 0.0,
            "min_time": float('inf')
        })
        
        logger.info(f"Performance profiler initialized with level: {profiling_level.name}")
    
    def _get_profiling_config(self) -> Dict[str, Any]:
        """Get profiling configuration based on level."""
        configs = {
            ProfilingLevel.DISABLED: {
                "enable_cprofile": False,
                "enable_line_profiler": False,
                "enable_memory_profiler": False,
                "sampling_rate": 0.0
            },
            ProfilingLevel.BASIC: {
                "enable_cprofile": True,
                "enable_line_profiler": False,
                "enable_memory_profiler": False,
                "sampling_rate": 0.1
            },
            ProfilingLevel.DETAILED: {
                "enable_cprofile": True,
                "enable_line_profiler": HAS_LINE_PROFILER,
                "enable_memory_profiler": False,
                "sampling_rate": 0.5
            },
            ProfilingLevel.COMPREHENSIVE: {
                "enable_cprofile": True,
                "enable_line_profiler": HAS_LINE_PROFILER,
                "enable_memory_profiler": HAS_MEMORY_PROFILER,
                "sampling_rate": 1.0
            },
            ProfilingLevel.ULTRA_DETAILED: {
                "enable_cprofile": True,
                "enable_line_profiler": HAS_LINE_PROFILER,
                "enable_memory_profiler": HAS_MEMORY_PROFILER,
                "sampling_rate": 1.0,
                "enable_gc_tracking": True,
                "enable_thread_profiling": True
            }
        }
        
        return configs[self.profiling_level]
    
    async def start_profiling(self) -> Any:
        """Start performance profiling."""
        if self.profiling_active or self.profiling_level == ProfilingLevel.DISABLED:
            return
        
        self.profiling_active = True
        
        # Start real-time monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance profiling started")
    
    async def stop_profiling(self) -> Any:
        """Stop performance profiling."""
        if not self.profiling_active:
            return
        
        self.profiling_active = False
        
        # Stop monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Generate final report
        report = self.generate_performance_report()
        
        logger.info("Performance profiling stopped", extra={"final_report": report})
    
    def profile_function(self, category: PerformanceCategory = PerformanceCategory.REQUEST_PROCESSING):
        """Decorator for profiling individual functions."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                if not self.profiling_active:
                    return await func(*args, **kwargs)
                
                return await self._profile_async_function(func, category, *args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                if not self.profiling_active:
                    return func(*args, **kwargs)
                
                return self._profile_sync_function(func, category, *args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    async def _profile_async_function(
        self,
        func: Callable,
        category: PerformanceCategory,
        *args,
        **kwargs
    ) -> Any:
        """Profile async function execution."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = time.process_time()
        
        error_count = 0
        result = None
        
        # Start profiling if enabled
        profiler = None
        if self.profiling_config.get("enable_cprofile", False):
            profiler = cProfile.Profile()
            profiler.enable()
        
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            error_count = 1
            logger.error(f"Function {func.__name__} error: {e}")
            raise
        finally:
            # Stop profiling
            if profiler:
                profiler.disable()
            
            # Calculate metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = time.process_time()
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = max(0, end_memory - start_memory)
            cpu_time_ms = (end_cpu - start_cpu) * 1000
            
            # Record metrics
            metric = PerformanceMetric(
                category=category,
                operation_name=func.__name__,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_time_ms=cpu_time_ms,
                io_operations=0,  # Would need instrumentation
                cache_hits=0,
                cache_misses=0,
                error_count=error_count
            )
            
            self._record_metric(metric)
            
            # Update function call statistics
            self._update_function_stats(func.__name__, execution_time_ms)
            
            # Store profiler data
            if profiler:
                self._store_profiler_data(func.__name__, profiler)
        
        return result
    
    def _profile_sync_function(
        self,
        func: Callable,
        category: PerformanceCategory,
        *args,
        **kwargs
    ) -> Any:
        """Profile sync function execution."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = time.process_time()
        
        error_count = 0
        result = None
        
        # Start profiling if enabled
        profiler = None
        if self.profiling_config.get("enable_cprofile", False):
            profiler = cProfile.Profile()
            profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error_count = 1
            logger.error(f"Function {func.__name__} error: {e}")
            raise
        finally:
            # Stop profiling
            if profiler:
                profiler.disable()
            
            # Calculate metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = time.process_time()
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = max(0, end_memory - start_memory)
            cpu_time_ms = (end_cpu - start_cpu) * 1000
            
            # Record metrics
            metric = PerformanceMetric(
                category=category,
                operation_name=func.__name__,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_time_ms=cpu_time_ms,
                io_operations=0,
                cache_hits=0,
                cache_misses=0,
                error_count=error_count
            )
            
            self._record_metric(metric)
            
            # Update function call statistics
            self._update_function_stats(func.__name__, execution_time_ms)
            
            # Store profiler data
            if profiler:
                self._store_profiler_data(func.__name__, profiler)
        
        return result
    
    @contextmanager
    def profile_block(
        self,
        block_name: str,
        category: PerformanceCategory = PerformanceCategory.REQUEST_PROCESSING
    ):
        """Context manager for profiling code blocks."""
        if not self.profiling_active:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = max(0, end_memory - start_memory)
            
            metric = PerformanceMetric(
                category=category,
                operation_name=block_name,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_time_ms=0,
                io_operations=0,
                cache_hits=0,
                cache_misses=0,
                error_count=0
            )
            
            self._record_metric(metric)
    
    def _record_metric(self, metric: PerformanceMetric):
        """Record performance metric."""
        self.metrics_history.append(metric)
        self.operation_stats[metric.operation_name].append(metric.execution_time_ms)
        
        # Check for bottlenecks
        self._check_for_bottlenecks(metric)
    
    def _update_function_stats(self, function_name: str, execution_time_ms: float):
        """Update function call statistics."""
        stats = self.function_calls[function_name]
        stats["count"] += 1
        stats["total_time"] += execution_time_ms
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["max_time"] = max(stats["max_time"], execution_time_ms)
        stats["min_time"] = min(stats["min_time"], execution_time_ms)
    
    def _store_profiler_data(self, function_name: str, profiler: cProfile.Profile):
        """Store profiler data for later analysis."""
        # Convert profiler stats to string
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profiler_output = s.getvalue()
        
        # Store in a way that can be retrieved later
        # This is a simplified version - in production, you might want to store in a database
        logger.debug(f"Profiler data for {function_name}", extra={"profiler_output": profiler_output})
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    async def _monitoring_loop(self) -> Any:
        """Real-time monitoring loop."""
        while self.profiling_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Analyze recent metrics for bottlenecks
                await self._analyze_recent_metrics()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self) -> Any:
        """Collect system-wide performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # GC stats
            gc_stats = gc.get_stats()
            
            system_metric = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_read_mb": disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                "disk_write_mb": disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                "network_sent_mb": network_io.bytes_sent / (1024 * 1024) if network_io else 0,
                "network_recv_mb": network_io.bytes_recv / (1024 * 1024) if network_io else 0,
                "gc_collections": sum(stat["collections"] for stat in gc_stats)
            }
            
            # Check system thresholds
            if cpu_percent > self.performance_thresholds["high_cpu_percent"]:
                await self._detect_cpu_bottleneck(cpu_percent)
            
            if memory.percent > 85:  # High memory usage
                await self._detect_memory_bottleneck(memory.percent)
                
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    async def _analyze_recent_metrics(self) -> Any:
        """Analyze recent metrics for performance issues."""
        if len(self.metrics_history) < 10:
            return
        
        # Get recent metrics (last 10)
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Analyze for patterns
        slow_operations = [
            m for m in recent_metrics
            if m.execution_time_ms > self.performance_thresholds["slow_request_ms"]
        ]
        
        high_memory_operations = [
            m for m in recent_metrics
            if m.memory_usage_mb > self.performance_thresholds["high_memory_mb"]
        ]
        
        error_operations = [
            m for m in recent_metrics
            if m.error_count > 0
        ]
        
        # Detect bottlenecks
        if slow_operations:
            await self._detect_performance_bottleneck(slow_operations)
        
        if high_memory_operations:
            await self._detect_memory_pressure(high_memory_operations)
        
        if error_operations:
            await self._detect_error_patterns(error_operations)
    
    def _check_for_bottlenecks(self, metric: PerformanceMetric):
        """Check individual metric for bottlenecks."""
        bottlenecks = []
        
        # Slow execution time
        if metric.execution_time_ms > self.performance_thresholds["slow_request_ms"]:
            bottlenecks.append(BottleneckDetection(
                bottleneck_type=BottleneckType.CPU_BOUND,
                severity=min(1.0, metric.execution_time_ms / (self.performance_thresholds["slow_request_ms"] * 2)),
                affected_operations=[metric.operation_name],
                recommendations=[
                    "Consider optimizing algorithm complexity",
                    "Implement caching for expensive operations",
                    "Use async/await for I/O operations"
                ],
                metrics={"execution_time_ms": metric.execution_time_ms}
            ))
        
        # High memory usage
        if metric.memory_usage_mb > self.performance_thresholds["high_memory_mb"]:
            bottlenecks.append(BottleneckDetection(
                bottleneck_type=BottleneckType.MEMORY_BOUND,
                severity=min(1.0, metric.memory_usage_mb / (self.performance_thresholds["high_memory_mb"] * 2)),
                affected_operations=[metric.operation_name],
                recommendations=[
                    "Implement memory pooling",
                    "Use generators for large datasets",
                    "Implement lazy loading",
                    "Add garbage collection optimization"
                ],
                metrics={"memory_usage_mb": metric.memory_usage_mb}
            ))
        
        # Store detected bottlenecks
        self.bottlenecks.extend(bottlenecks)
        
        # Keep only recent bottlenecks (last 100)
        if len(self.bottlenecks) > 100:
            self.bottlenecks = self.bottlenecks[-100:]
    
    async def _detect_cpu_bottleneck(self, cpu_percent: float):
        """Detect CPU bottleneck."""
        bottleneck = BottleneckDetection(
            bottleneck_type=BottleneckType.CPU_BOUND,
            severity=min(1.0, cpu_percent / 100.0),
            affected_operations=["system_wide"],
            recommendations=[
                "Scale horizontally with more workers",
                "Optimize CPU-intensive algorithms",
                "Use process pooling for CPU-bound tasks",
                "Implement request throttling"
            ],
            metrics={"cpu_percent": cpu_percent}
        )
        
        self.bottlenecks.append(bottleneck)
        
        logger.warning(
            "CPU bottleneck detected",
            extra={"cpu_percent": cpu_percent, "severity": bottleneck.severity}
        )
    
    async def _detect_memory_bottleneck(self, memory_percent: float):
        """Detect memory bottleneck."""
        bottleneck = BottleneckDetection(
            bottleneck_type=BottleneckType.MEMORY_BOUND,
            severity=min(1.0, memory_percent / 100.0),
            affected_operations=["system_wide"],
            recommendations=[
                "Implement memory optimization",
                "Use memory pooling",
                "Add garbage collection tuning",
                "Scale to larger instances"
            ],
            metrics={"memory_percent": memory_percent}
        )
        
        self.bottlenecks.append(bottleneck)
        
        logger.warning(
            "Memory bottleneck detected",
            extra={"memory_percent": memory_percent, "severity": bottleneck.severity}
        )
    
    async def _detect_performance_bottleneck(self, slow_operations: List[PerformanceMetric]):
        """Detect performance bottleneck from slow operations."""
        operation_names = [op.operation_name for op in slow_operations]
        avg_time = np.mean([op.execution_time_ms for op in slow_operations])
        
        bottleneck = BottleneckDetection(
            bottleneck_type=BottleneckType.CPU_BOUND,
            severity=min(1.0, avg_time / (self.performance_thresholds["slow_request_ms"] * 2)),
            affected_operations=operation_names,
            recommendations=[
                "Profile and optimize slow functions",
                "Implement async processing",
                "Add result caching",
                "Use background task processing"
            ],
            metrics={"avg_execution_time_ms": avg_time, "slow_operations_count": len(slow_operations)}
        )
        
        self.bottlenecks.append(bottleneck)
    
    async def _detect_memory_pressure(self, high_memory_operations: List[PerformanceMetric]):
        """Detect memory pressure from high memory operations."""
        operation_names = [op.operation_name for op in high_memory_operations]
        avg_memory = np.mean([op.memory_usage_mb for op in high_memory_operations])
        
        bottleneck = BottleneckDetection(
            bottleneck_type=BottleneckType.MEMORY_BOUND,
            severity=min(1.0, avg_memory / (self.performance_thresholds["high_memory_mb"] * 2)),
            affected_operations=operation_names,
            recommendations=[
                "Implement memory optimization",
                "Use streaming for large data",
                "Add memory pooling",
                "Implement lazy loading"
            ],
            metrics={"avg_memory_usage_mb": avg_memory, "high_memory_operations_count": len(high_memory_operations)}
        )
        
        self.bottlenecks.append(bottleneck)
    
    async def _detect_error_patterns(self, error_operations: List[PerformanceMetric]):
        """Detect error patterns."""
        operation_names = [op.operation_name for op in error_operations]
        error_rate = len(error_operations) / len(self.metrics_history) if self.metrics_history else 0
        
        if error_rate > self.performance_thresholds["error_rate"]:
            bottleneck = BottleneckDetection(
                bottleneck_type=BottleneckType.IO_BOUND,  # Assuming errors are often I/O related
                severity=min(1.0, error_rate / self.performance_thresholds["error_rate"]),
                affected_operations=operation_names,
                recommendations=[
                    "Implement retry logic",
                    "Add circuit breaker pattern",
                    "Improve error handling",
                    "Add request validation"
                ],
                metrics={"error_rate": error_rate, "error_operations_count": len(error_operations)}
            )
            
            self.bottlenecks.append(bottleneck)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No performance data collected"}
        
        # Calculate overall statistics
        total_metrics = len(self.metrics_history)
        avg_execution_time = np.mean([m.execution_time_ms for m in self.metrics_history])
        p95_execution_time = np.percentile([m.execution_time_ms for m in self.metrics_history], 95)
        avg_memory_usage = np.mean([m.memory_usage_mb for m in self.metrics_history])
        
        # Function statistics
        function_stats = {}
        for func_name, stats in self.function_calls.items():
            if stats["count"] > 0:
                function_stats[func_name] = {
                    "call_count": stats["count"],
                    "avg_time_ms": stats["avg_time"],
                    "max_time_ms": stats["max_time"],
                    "min_time_ms": stats["min_time"] if stats["min_time"] != float('inf') else 0,
                    "total_time_ms": stats["total_time"]
                }
        
        # Top slowest functions
        slowest_functions = sorted(
            function_stats.items(),
            key=lambda x: x[1]["avg_time_ms"],
            reverse=True
        )[:10]
        
        # Bottleneck summary
        bottleneck_summary = {}
        for bottleneck in self.bottlenecks[-20:]:  # Last 20 bottlenecks
            bt_type = bottleneck.bottleneck_type.name
            if bt_type not in bottleneck_summary:
                bottleneck_summary[bt_type] = {
                    "count": 0,
                    "max_severity": 0.0,
                    "recommendations": set()
                }
            
            bottleneck_summary[bt_type]["count"] += 1
            bottleneck_summary[bt_type]["max_severity"] = max(
                bottleneck_summary[bt_type]["max_severity"],
                bottleneck.severity
            )
            bottleneck_summary[bt_type]["recommendations"].update(bottleneck.recommendations)
        
        # Convert sets to lists for JSON serialization
        for bt_data in bottleneck_summary.values():
            bt_data["recommendations"] = list(bt_data["recommendations"])
        
        # Performance trends
        if len(self.metrics_history) >= 2:
            recent_metrics = list(self.metrics_history)[-min(100, len(self.metrics_history)):]
            older_metrics = list(self.metrics_history)[-min(200, len(self.metrics_history)):-100] if len(self.metrics_history) > 100 else []
            
            if older_metrics:
                recent_avg = np.mean([m.execution_time_ms for m in recent_metrics])
                older_avg = np.mean([m.execution_time_ms for m in older_metrics])
                performance_trend = "improving" if recent_avg < older_avg else "degrading"
                trend_percentage = abs((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
            else:
                performance_trend = "insufficient_data"
                trend_percentage = 0
        else:
            performance_trend = "insufficient_data"
            trend_percentage = 0
        
        report = {
            "profiling_level": self.profiling_level.name,
            "collection_period": {
                "start": self.metrics_history[0].timestamp.isoformat() if self.metrics_history else None,
                "end": self.metrics_history[-1].timestamp.isoformat() if self.metrics_history else None,
                "total_metrics": total_metrics
            },
            "overall_performance": {
                "avg_execution_time_ms": round(avg_execution_time, 2),
                "p95_execution_time_ms": round(p95_execution_time, 2),
                "avg_memory_usage_mb": round(avg_memory_usage, 2)
            },
            "function_statistics": function_stats,
            "slowest_functions": [{"name": name, **stats} for name, stats in slowest_functions],
            "bottleneck_summary": bottleneck_summary,
            "performance_trend": {
                "direction": performance_trend,
                "change_percentage": round(trend_percentage, 2)
            },
            "recommendations": self._generate_optimization_recommendations(bottleneck_summary, function_stats)
        }
        
        return report
    
    def _generate_optimization_recommendations(
        self,
        bottleneck_summary: Dict[str, Any],
        function_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        # CPU optimization recommendations
        if "CPU_BOUND" in bottleneck_summary:
            cpu_data = bottleneck_summary["CPU_BOUND"]
            if cpu_data["max_severity"] > 0.7:
                recommendations.extend([
                    "🔥 HIGH PRIORITY: Optimize CPU-intensive operations",
                    "Consider implementing async processing for blocking operations",
                    "Use process pooling for CPU-bound tasks",
                    "Implement request throttling and rate limiting"
                ])
        
        # Memory optimization recommendations
        if "MEMORY_BOUND" in bottleneck_summary:
            memory_data = bottleneck_summary["MEMORY_BOUND"]
            if memory_data["max_severity"] > 0.7:
                recommendations.extend([
                    "🔥 HIGH PRIORITY: Implement memory optimization",
                    "Use memory pooling for frequent allocations",
                    "Implement lazy loading for large objects",
                    "Add garbage collection tuning"
                ])
        
        # Function-specific recommendations
        slow_functions = [
            name for name, stats in function_stats.items()
            if stats["avg_time_ms"] > self.performance_thresholds["slow_request_ms"]
        ]
        
        if slow_functions:
            recommendations.extend([
                f"🎯 Optimize slow functions: {', '.join(slow_functions[:5])}",
                "Profile individual functions for bottlenecks",
                "Consider caching results of expensive operations"
            ])
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.extend([
                "✅ Performance looks good!",
                "Continue monitoring for optimal performance",
                "Consider implementing predictive caching"
            ])
        
        return recommendations
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 operations
        
        return {
            "current_operations": len(recent_metrics),
            "avg_execution_time_ms": np.mean([m.execution_time_ms for m in recent_metrics]),
            "avg_memory_usage_mb": np.mean([m.memory_usage_mb for m in recent_metrics]),
            "recent_errors": sum(m.error_count for m in recent_metrics),
            "active_bottlenecks": len([b for b in self.bottlenecks[-5:] if b.severity > 0.5]),
            "profiling_active": self.profiling_active,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# =============================================================================
# FastAPI Integration
# =============================================================================

def create_performance_profiler(profiling_level: ProfilingLevel = ProfilingLevel.DETAILED) -> AdvancedPerformanceProfiler:
    """Factory function to create performance profiler."""
    return AdvancedPerformanceProfiler(profiling_level)

# Example FastAPI middleware integration
class PerformanceProfilingMiddleware:
    """FastAPI middleware for automatic performance profiling."""
    
    def __init__(self, profiler: AdvancedPerformanceProfiler):
        
    """__init__ function."""
self.profiler = profiler
    
    async def __call__(self, request, call_next) -> Any:
        """Profile FastAPI requests."""
        if not self.profiler.profiling_active:
            return await call_next(request)
        
        start_time = time.time()
        start_memory = self.profiler._get_memory_usage()
        
        try:
            response = await call_next(request)
            error_count = 0
        except Exception as e:
            error_count = 1
            logger.error(f"Request error: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = self.profiler._get_memory_usage()
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = max(0, end_memory - start_memory)
            
            metric = PerformanceMetric(
                category=PerformanceCategory.REQUEST_PROCESSING,
                operation_name=f"{request.method} {request.url.path}",
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_time_ms=0,
                io_operations=0,
                cache_hits=0,
                cache_misses=0,
                error_count=error_count
            )
            
            self.profiler._record_metric(metric)
        
        return response 