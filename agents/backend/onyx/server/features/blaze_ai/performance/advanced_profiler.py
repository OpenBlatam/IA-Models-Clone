"""
Advanced Performance Profiler for Blaze AI System.

This module provides comprehensive performance profiling, bottleneck detection,
optimization recommendations, and performance analytics.
"""

from __future__ import annotations

import asyncio
import time
import cProfile
import pstats
import io
import tracemalloc
import psutil
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable, Protocol
from collections import defaultdict, deque
import threading
import functools
import inspect
import linecache
import os
from contextlib import contextmanager, asynccontextmanager
import json
import statistics
from datetime import datetime, timedelta

from ..core.interfaces import CoreConfig, SystemHealth, HealthStatus
from ..utils.logging import get_logger
from ..utils.metrics import MetricsCollector

# =============================================================================
# Profiling Types
# =============================================================================

class ProfilingLevel(Enum):
    """Profiling detail levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"

class ProfilingMode(Enum):
    """Profiling modes."""
    CPU = "cpu"
    MEMORY = "memory"
    I_O = "i_o"
    NETWORK = "network"
    COMBINED = "combined"

@dataclass
class ProfilerConfig:
    """Configuration for performance profiler."""
    profiling_level: ProfilingLevel = ProfilingLevel.DETAILED
    profiling_mode: ProfilingMode = ProfilingMode.COMBINED
    enable_memory_tracking: bool = True
    enable_cpu_profiling: bool = True
    enable_io_profiling: bool = True
    enable_network_profiling: bool = True
    sample_interval: float = 0.1  # seconds
    max_samples: int = 10000
    enable_auto_optimization: bool = True
    performance_threshold: float = 0.8
    enable_bottleneck_detection: bool = True
    enable_optimization_recommendations: bool = True

# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    memory_percent: float
    io_read_bytes: float
    io_write_bytes: float
    io_read_count: int
    io_write_count: int
    network_bytes_sent: float
    network_bytes_recv: float
    network_packets_sent: int
    network_packets_recv: int
    gc_collections: int
    gc_time: float
    active_threads: int
    active_processes: int

@dataclass
class FunctionProfile:
    """Function-level profiling data."""
    function_name: str
    module_name: str
    call_count: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    cumulative_time: float
    line_profiles: Dict[int, Dict[str, Any]]
    memory_usage: float
    memory_peak: float

@dataclass
class BottleneckReport:
    """Bottleneck detection report."""
    severity: str
    category: str
    description: str
    impact: float
    recommendations: List[str]
    metrics: Dict[str, Any]
    timestamp: float

# =============================================================================
# CPU Profiler
# =============================================================================

class CPUProfiler:
    """Advanced CPU profiling with detailed analysis."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.logger = get_logger("cpu_profiler")
        self.profiler = cProfile.Profile()
        self.stats: Optional[pstats.Stats] = None
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.call_stack: List[str] = []
        self.profiling_active = False
        
        # Performance tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_calls = 0
        self.total_time = 0.0
    
    def start_profiling(self) -> None:
        """Start CPU profiling."""
        if self.profiling_active:
            self.logger.warning("CPU profiling already active")
            return
        
        self.profiler.enable()
        self.profiling_active = True
        self.start_time = time.time()
        self.logger.info("CPU profiling started")
    
    def stop_profiling(self) -> None:
        """Stop CPU profiling and collect statistics."""
        if not self.profiling_active:
            self.logger.warning("CPU profiling not active")
            return
        
        self.profiler.disable()
        self.profiling_active = False
        self.end_time = time.time()
        
        # Collect statistics
        self._collect_statistics()
        self.logger.info("CPU profiling stopped")
    
    def _collect_statistics(self) -> None:
        """Collect and analyze profiling statistics."""
        try:
            # Get stats from profiler
            s = io.StringIO()
            self.profiler.print_stats(sort='cumulative', stream=s)
            s.seek(0)
            
            # Parse stats
            self.stats = pstats.Stats(self.profiler)
            
            # Extract function profiles
            self._extract_function_profiles()
            
        except Exception as e:
            self.logger.error(f"Failed to collect CPU statistics: {e}")
    
    def _extract_function_profiles(self) -> None:
        """Extract detailed function profiles from stats."""
        if not self.stats:
            return
        
        # Get top functions by cumulative time
        top_functions = self.stats.get_stats_profile()
        
        for func, (cc, nc, tt, ct, callers) in top_functions.items():
            try:
                # Parse function name
                if isinstance(func, tuple):
                    filename, line_num, func_name = func
                else:
                    filename, line_num, func_name = func.co_filename, func.co_firstlineno, func.co_name
                
                # Create function profile
                profile = FunctionProfile(
                    function_name=func_name,
                    module_name=os.path.basename(filename),
                    call_count=nc,
                    total_time=tt,
                    average_time=tt / nc if nc > 0 else 0,
                    min_time=0,  # Not available in cProfile
                    max_time=0,   # Not available in cProfile
                    cumulative_time=ct,
                    line_profiles={},
                    memory_usage=0,
                    memory_peak=0
                )
                
                self.function_profiles[func_name] = profile
                
            except Exception as e:
                self.logger.error(f"Failed to extract profile for function: {e}")
    
    def get_top_functions(self, limit: int = 10) -> List[FunctionProfile]:
        """Get top functions by cumulative time."""
        sorted_functions = sorted(
            self.function_profiles.values(),
            key=lambda x: x.cumulative_time,
            reverse=True
        )
        return sorted_functions[:limit]
    
    def get_function_profile(self, function_name: str) -> Optional[FunctionProfile]:
        """Get profile for a specific function."""
        return self.function_profiles.get(function_name)
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get CPU profiling summary."""
        return {
            "profiling_active": self.profiling_active,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": (self.end_time - self.start_time) if self.start_time and self.end_time else 0,
            "total_functions_profiled": len(self.function_profiles),
            "top_functions": [
                {
                    "name": f.function_name,
                    "module": f.module_name,
                    "calls": f.call_count,
                    "total_time": f.total_time,
                    "cumulative_time": f.cumulative_time
                }
                for f in self.get_top_functions(5)
            ]
        }

# =============================================================================
# Memory Profiler
# =============================================================================

class MemoryProfiler:
    """Advanced memory profiling with leak detection."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.logger = get_logger("memory_profiler")
        self.tracemalloc_active = False
        self.snapshots: List[tracemalloc.Snapshot] = []
        self.memory_traces: List[Dict[str, Any]] = []
        self.leak_suspects: List[Dict[str, Any]] = []
        
        # Memory tracking
        self.start_memory = 0
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_history: deque = deque(maxlen=1000)
    
    def start_profiling(self) -> None:
        """Start memory profiling."""
        if self.tracemalloc_active:
            self.logger.warning("Memory profiling already active")
            return
        
        try:
            tracemalloc.start()
            self.tracemalloc_active = True
            self.start_memory = tracemalloc.get_traced_memory()[0]
            self.peak_memory = self.start_memory
            self.current_memory = self.start_memory
            
            self.logger.info("Memory profiling started")
            
        except Exception as e:
            self.logger.error(f"Failed to start memory profiling: {e}")
    
    def stop_profiling(self) -> None:
        """Stop memory profiling and collect statistics."""
        if not self.tracemalloc_active:
            self.logger.warning("Memory profiling not active")
            return
        
        try:
            # Take final snapshot
            final_snapshot = tracemalloc.take_snapshot()
            self.snapshots.append(final_snapshot)
            
            # Stop tracemalloc
            tracemalloc.stop()
            self.tracemalloc_active = False
            
            # Analyze memory usage
            self._analyze_memory_usage()
            
            self.logger.info("Memory profiling stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop memory profiling: {e}")
    
    def take_snapshot(self) -> None:
        """Take a memory snapshot."""
        if not self.tracemalloc_active:
            return
        
        try:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append(snapshot)
            
            # Update current memory
            current, peak = tracemalloc.get_traced_memory()
            self.current_memory = current
            self.peak_memory = max(self.peak_memory, peak)
            
            # Store in history
            self.memory_history.append({
                "timestamp": time.time(),
                "current": current,
                "peak": peak
            })
            
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
    
    def _analyze_memory_usage(self) -> None:
        """Analyze memory usage patterns."""
        if len(self.snapshots) < 2:
            return
        
        try:
            # Compare snapshots
            old_snapshot = self.snapshots[-2]
            new_snapshot = self.snapshots[-1]
            
            # Get top memory allocations
            top_stats = new_snapshot.statistics('lineno')
            
            # Check for potential memory leaks
            self._detect_memory_leaks(old_snapshot, new_snapshot)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze memory usage: {e}")
    
    def _detect_memory_leaks(self, old_snapshot: tracemalloc.Snapshot, new_snapshot: tracemalloc.Snapshot) -> None:
        """Detect potential memory leaks."""
        try:
            # Compare memory usage
            stats = new_snapshot.compare_to(old_snapshot, 'lineno')
            
            # Look for significant increases
            for stat in stats:
                if stat.size_diff > 1024 * 1024:  # 1MB threshold
                    self.leak_suspects.append({
                        "file": stat.traceback.format()[0],
                        "size_diff": stat.size_diff,
                        "size": stat.size,
                        "count_diff": stat.count_diff,
                        "timestamp": time.time()
                    })
            
        except Exception as e:
            self.logger.error(f"Failed to detect memory leaks: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory profiling summary."""
        return {
            "profiling_active": self.tracemalloc_active,
            "start_memory": self.start_memory,
            "current_memory": self.current_memory,
            "peak_memory": self.peak_memory,
            "memory_increase": self.current_memory - self.start_memory,
            "snapshots_taken": len(self.snapshots),
            "memory_history_points": len(self.memory_history),
            "leak_suspects": len(self.leak_suspects),
            "recent_leak_suspects": self.leak_suspects[-5:] if self.leak_suspects else []
        }

# =============================================================================
# I/O Profiler
# =============================================================================

class IOProfiler:
    """Advanced I/O profiling with bottleneck detection."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.logger = get_logger("io_profiler")
        self.io_stats: Dict[str, Dict[str, Any]] = {}
        self.io_history: deque = deque(maxlen=1000)
        self.bottlenecks: List[Dict[str, Any]] = []
        
        # I/O tracking
        self.start_io = self._get_io_stats()
        self.current_io = self.start_io
    
    def _get_io_stats(self) -> Dict[str, Any]:
        """Get current I/O statistics."""
        try:
            disk_io = psutil.disk_io_counters()
            return {
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
                "timestamp": time.time()
            }
        except Exception:
            return {
                "read_bytes": 0,
                "write_bytes": 0,
                "read_count": 0,
                "write_count": 0,
                "timestamp": time.time()
            }
    
    def update_io_stats(self) -> None:
        """Update I/O statistics."""
        try:
            self.current_io = self._get_io_stats()
            
            # Calculate deltas
            delta_read = self.current_io["read_bytes"] - self.start_io["read_bytes"]
            delta_write = self.current_io["write_bytes"] - self.start_io["write_bytes"]
            delta_read_count = self.current_io["read_count"] - self.start_io["read_count"]
            delta_write_count = self.current_io["write_count"] - self.start_io["write_count"]
            
            # Store in history
            io_data = {
                "timestamp": time.time(),
                "read_bytes_delta": delta_read,
                "write_bytes_delta": delta_write,
                "read_count_delta": delta_read_count,
                "write_count_delta": delta_write_count,
                "total_read_bytes": self.current_io["read_bytes"],
                "total_write_bytes": self.current_io["write_bytes"]
            }
            
            self.io_history.append(io_data)
            
            # Check for bottlenecks
            self._check_io_bottlenecks(io_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update I/O stats: {e}")
    
    def _check_io_bottlenecks(self, io_data: Dict[str, Any]) -> None:
        """Check for I/O bottlenecks."""
        # High I/O activity threshold
        high_io_threshold = 100 * 1024 * 1024  # 100MB
        
        if (io_data["read_bytes_delta"] > high_io_threshold or 
            io_data["write_bytes_delta"] > high_io_threshold):
            
            bottleneck = {
                "type": "high_io_activity",
                "severity": "warning",
                "description": f"High I/O activity detected: Read {io_data['read_bytes_delta']} bytes, Write {io_data['write_bytes_delta']} bytes",
                "timestamp": time.time(),
                "metrics": io_data
            }
            
            self.bottlenecks.append(bottleneck)
            self.logger.warning(f"I/O bottleneck detected: {bottleneck['description']}")
    
    def get_io_summary(self) -> Dict[str, Any]:
        """Get I/O profiling summary."""
        if not self.io_history:
            return {"error": "No I/O data available"}
        
        # Calculate averages
        read_bytes_deltas = [h["read_bytes_delta"] for h in self.io_history]
        write_bytes_deltas = [h["write_bytes_delta"] for h in self.io_history]
        
        return {
            "total_read_bytes": self.current_io["read_bytes"] - self.start_io["read_bytes"],
            "total_write_bytes": self.current_io["write_bytes"] - self.start_io["write_bytes"],
            "average_read_bytes_per_sample": statistics.mean(read_bytes_deltas),
            "average_write_bytes_per_sample": statistics.mean(write_bytes_deltas),
            "max_read_bytes_per_sample": max(read_bytes_deltas),
            "max_write_bytes_per_sample": max(write_bytes_deltas),
            "io_history_points": len(self.io_history),
            "bottlenecks_detected": len(self.bottlenecks),
            "recent_bottlenecks": self.bottlenecks[-5:] if self.bottlenecks else []
        }

# =============================================================================
# Main Advanced Profiler
# =============================================================================

class AdvancedProfiler:
    """Main advanced profiler coordinating all profiling components."""
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self.logger = get_logger("advanced_profiler")
        
        # Initialize profiling components
        self.cpu_profiler = CPUProfiler(self.config)
        self.memory_profiler = MemoryProfiler(self.config)
        self.io_profiler = IOProfiler(self.config)
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=self.config.max_samples)
        self.bottleneck_reports: List[BottleneckReport] = []
        self.optimization_recommendations: List[str] = []
        
        # Background tasks
        self._profiling_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._start_background_tasks()
    
    def start_profiling(self) -> None:
        """Start comprehensive profiling."""
        self.logger.info("Starting comprehensive performance profiling")
        
        if self.config.enable_cpu_profiling:
            self.cpu_profiler.start_profiling()
        
        if self.config.enable_memory_tracking:
            self.memory_profiler.start_profiling()
        
        if self.config.enable_io_profiling:
            self.io_profiler.update_io_stats()
    
    def stop_profiling(self) -> None:
        """Stop comprehensive profiling."""
        self.logger.info("Stopping comprehensive performance profiling")
        
        if self.config.enable_cpu_profiling:
            self.cpu_profiler.stop_profiling()
        
        if self.config.enable_memory_tracking:
            self.memory_profiler.stop_profiling()
        
        if self.config.enable_io_profiling:
            self.io_profiler.update_io_stats()
    
    @contextmanager
    def profile_context(self, context_name: str):
        """Context manager for profiling specific code sections."""
        try:
            self.logger.info(f"Starting profiling context: {context_name}")
            self.start_profiling()
            yield
        finally:
            self.stop_profiling()
            self.logger.info(f"Completed profiling context: {context_name}")
    
    @asynccontextmanager
    async def async_profile_context(self, context_name: str):
        """Async context manager for profiling specific code sections."""
        try:
            self.logger.info(f"Starting async profiling context: {context_name}")
            self.start_profiling()
            yield
        finally:
            self.stop_profiling()
            self.logger.info(f"Completed async profiling context: {context_name}")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_context(f"function_{func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    
    def profile_async_function(self, func: Callable) -> Callable:
        """Decorator to profile an async function."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.async_profile_context(f"async_function_{func.__name__}"):
                return await func(*args, **kwargs)
        return wrapper
    
    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # I/O metrics
            self.io_profiler.update_io_stats()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # GC metrics
            gc_stats = gc.get_stats()
            gc_collections = sum(stat["collections"] for stat in gc_stats)
            gc_time = sum(stat["collections_time"] for stat in gc_stats)
            
            # Process metrics
            process = psutil.Process()
            active_threads = process.num_threads()
            active_processes = len(psutil.pids())
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_percent,
                memory_usage=memory.used,
                memory_available=memory.available,
                memory_percent=memory.percent,
                io_read_bytes=self.io_profiler.current_io["read_bytes"],
                io_write_bytes=self.io_profiler.current_io["write_bytes"],
                io_read_count=self.io_profiler.current_io["read_count"],
                io_write_count=self.io_profiler.current_io["write_count"],
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                gc_collections=gc_collections,
                gc_time=gc_time,
                active_threads=active_threads,
                active_processes=active_processes
            )
            
            # Store in history
            self.performance_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
            raise
    
    async def detect_bottlenecks(self) -> List[BottleneckReport]:
        """Detect performance bottlenecks."""
        if not self.config.enable_bottleneck_detection:
            return []
        
        bottlenecks = []
        
        try:
            # CPU bottlenecks
            if self.config.enable_cpu_profiling:
                cpu_bottlenecks = self._detect_cpu_bottlenecks()
                bottlenecks.extend(cpu_bottlenecks)
            
            # Memory bottlenecks
            if self.config.enable_memory_tracking:
                memory_bottlenecks = self._detect_memory_bottlenecks()
                bottlenecks.extend(memory_bottlenecks)
            
            # I/O bottlenecks
            if self.config.enable_io_profiling:
                io_bottlenecks = self._detect_io_bottlenecks()
                bottlenecks.extend(io_bottlenecks)
            
            # Store bottlenecks
            self.bottleneck_reports.extend(bottlenecks)
            
        except Exception as e:
            self.logger.error(f"Failed to detect bottlenecks: {e}")
        
        return bottlenecks
    
    def _detect_cpu_bottlenecks(self) -> List[BottleneckReport]:
        """Detect CPU-related bottlenecks."""
        bottlenecks = []
        
        # Check for high CPU usage
        if len(self.performance_history) > 0:
            recent_cpu = [m.cpu_usage for m in list(self.performance_history)[-10:]]
            avg_cpu = statistics.mean(recent_cpu)
            
            if avg_cpu > 80:
                bottlenecks.append(BottleneckReport(
                    severity="high",
                    category="cpu",
                    description=f"High CPU usage detected: {avg_cpu:.1f}%",
                    impact=avg_cpu / 100,
                    recommendations=[
                        "Consider optimizing CPU-intensive operations",
                        "Implement caching for repeated computations",
                        "Use async/await for I/O operations",
                        "Profile specific functions for optimization"
                    ],
                    metrics={"average_cpu_usage": avg_cpu},
                    timestamp=time.time()
                ))
        
        # Check for slow functions
        top_functions = self.cpu_profiler.get_top_functions(5)
        for func in top_functions:
            if func.cumulative_time > 1.0:  # More than 1 second
                bottlenecks.append(BottleneckReport(
                    severity="medium",
                    category="cpu",
                    description=f"Slow function detected: {func.function_name} ({func.cumulative_time:.3f}s)",
                    impact=func.cumulative_time / 10,  # Normalize impact
                    recommendations=[
                        f"Optimize function {func.function_name}",
                        "Consider caching results",
                        "Implement early returns for edge cases",
                        "Use more efficient algorithms"
                    ],
                    metrics={
                        "function_name": func.function_name,
                        "cumulative_time": func.cumulative_time,
                        "call_count": func.call_count
                    },
                    timestamp=time.time()
                ))
        
        return bottlenecks
    
    def _detect_memory_bottlenecks(self) -> List[BottleneckReport]:
        """Detect memory-related bottlenecks."""
        bottlenecks = []
        
        # Check for memory leaks
        if self.memory_profiler.leak_suspects:
            for leak in self.memory_profiler.leak_suspects[-3:]:  # Last 3 suspects
                bottlenecks.append(BottleneckReport(
                    severity="high",
                    category="memory",
                    description=f"Potential memory leak detected: {leak['file']}",
                    impact=min(leak['size_diff'] / (1024 * 1024 * 100), 1.0),  # Normalize to 100MB
                    recommendations=[
                        "Review memory allocation patterns",
                        "Check for circular references",
                        "Implement proper cleanup in destructors",
                        "Use weak references where appropriate"
                    ],
                    metrics=leak,
                    timestamp=time.time()
                ))
        
        # Check for high memory usage
        if len(self.performance_history) > 0:
            recent_memory = [m.memory_percent for m in list(self.performance_history)[-10:]]
            avg_memory = statistics.mean(recent_memory)
            
            if avg_memory > 80:
                bottlenecks.append(BottleneckReport(
                    severity="medium",
                    category="memory",
                    description=f"High memory usage detected: {avg_memory:.1f}%",
                    impact=avg_memory / 100,
                    recommendations=[
                        "Implement memory pooling",
                        "Use generators for large datasets",
                        "Implement pagination for data processing",
                        "Consider using __slots__ for classes"
                    ],
                    metrics={"average_memory_usage": avg_memory},
                    timestamp=time.time()
                ))
        
        return bottlenecks
    
    def _detect_io_bottlenecks(self) -> List[BottleneckReport]:
        """Detect I/O-related bottlenecks."""
        bottlenecks = []
        
        # Check for high I/O activity
        if self.io_profiler.bottlenecks:
            for bottleneck in self.io_profiler.bottlenecks[-3:]:  # Last 3 bottlenecks
                bottlenecks.append(BottleneckReport(
                    severity="medium",
                    category="io",
                    description=bottleneck["description"],
                    impact=0.5,  # Medium impact
                    recommendations=[
                        "Implement I/O buffering",
                        "Use async I/O operations",
                        "Consider caching frequently accessed data",
                        "Optimize file read/write patterns"
                    ],
                    metrics=bottleneck["metrics"],
                    timestamp=time.time()
                ))
        
        return bottlenecks
    
    async def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        if not self.config.enable_optimization_recommendations:
            return []
        
        recommendations = []
        
        try:
            # Analyze bottlenecks
            bottlenecks = await self.detect_bottlenecks()
            
            # Generate recommendations based on bottleneck severity
            high_severity = [b for b in bottlenecks if b.severity == "high"]
            medium_severity = [b for b in bottlenecks if b.severity == "medium"]
            
            if high_severity:
                recommendations.append("🚨 HIGH PRIORITY: Address critical performance bottlenecks immediately")
                for bottleneck in high_severity:
                    recommendations.extend(bottleneck.recommendations[:2])  # Top 2 recommendations
            
            if medium_severity:
                recommendations.append("⚠️ MEDIUM PRIORITY: Consider optimizing these areas for better performance")
                for bottleneck in medium_severity:
                    recommendations.extend(bottleneck.recommendations[:1])  # Top recommendation
            
            # General recommendations based on profiling level
            if self.config.profiling_level == ProfilingLevel.COMPREHENSIVE:
                recommendations.extend([
                    "📊 Use comprehensive profiling to identify optimization opportunities",
                    "🔄 Implement continuous performance monitoring",
                    "🎯 Focus on the 20% of code that causes 80% of performance issues"
                ])
            
            # Store recommendations
            self.optimization_recommendations = recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization recommendations: {e}")
        
        return recommendations
    
    async def get_profiling_summary(self) -> Dict[str, Any]:
        """Get comprehensive profiling summary."""
        return {
            "profiling_config": {
                "level": self.config.profiling_level.value,
                "mode": self.config.profiling_mode.value,
                "cpu_profiling": self.config.enable_cpu_profiling,
                "memory_tracking": self.config.enable_memory_tracking,
                "io_profiling": self.config.enable_io_profiling
            },
            "cpu_profiler": self.cpu_profiler.get_profiling_summary(),
            "memory_profiler": self.memory_profiler.get_memory_summary(),
            "io_profiler": self.io_profiler.get_io_summary(),
            "performance_metrics": {
                "history_points": len(self.performance_history),
                "recent_metrics": [
                    {
                        "timestamp": m.timestamp,
                        "cpu_usage": m.cpu_usage,
                        "memory_percent": m.memory_percent
                    }
                    for m in list(self.performance_history)[-5:]
                ] if self.performance_history else []
            },
            "bottlenecks": {
                "total_detected": len(self.bottleneck_reports),
                "recent_bottlenecks": [
                    {
                        "severity": b.severity,
                        "category": b.category,
                        "description": b.description
                    }
                    for b in self.bottleneck_reports[-5:]
                ] if self.bottleneck_reports else []
            },
            "optimization_recommendations": self.optimization_recommendations
        }
    
    async def _profiling_loop(self):
        """Background profiling loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect performance metrics
                await self.collect_performance_metrics()
                
                # Detect bottlenecks
                await self.detect_bottlenecks()
                
                # Generate recommendations
                await self.generate_optimization_recommendations()
                
                await asyncio.sleep(self.config.sample_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Profiling loop error: {e}")
                await asyncio.sleep(60)
    
    def _start_background_tasks(self):
        """Start background tasks."""
        if self._profiling_task is None or self._profiling_task.done():
            self._profiling_task = asyncio.create_task(self._profiling_loop())
    
    async def shutdown(self):
        """Shutdown the advanced profiler."""
        self.logger.info("Shutting down advanced profiler...")
        self._shutdown_event.set()
        
        if self._profiling_task:
            self._profiling_task.cancel()
            try:
                await self._profiling_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Advanced profiler shutdown complete")

# =============================================================================
# Factory Functions
# =============================================================================

def create_advanced_profiler(config: Optional[ProfilerConfig] = None) -> AdvancedProfiler:
    """Create an advanced profiler instance."""
    return AdvancedProfiler(config)

# Export main classes
__all__ = [
    "AdvancedProfiler",
    "CPUProfiler",
    "MemoryProfiler",
    "IOProfiler",
    "ProfilerConfig",
    "ProfilingLevel",
    "ProfilingMode",
    "PerformanceMetrics",
    "FunctionProfile",
    "BottleneckReport",
    "create_advanced_profiler"
]


