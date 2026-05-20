from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import time
import tracemalloc
import cProfile
import pstats
import io
import sys
import os
import threading
import gc
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import json
import logging
from datetime import datetime, timedelta
from memory_profiler import profile, memory_usage
import psutil
import objgraph
from py-spy import top
import pyinstrument
from pyinstrument import Profiler
import line_profiler
import py-spy
from ...core.domain.entities.linkedin_post import LinkedInPost, PostStatus, PostType, PostTone
from ...application.use_cases.linkedin_post_use_cases import LinkedInPostUseCases
from ...infrastructure.repositories.linkedin_post_repository import LinkedInPostRepository
from ...shared.cache import CacheManager
from ..conftest_advanced import (
        from ...application.use_cases.linkedin_post_use_cases import LinkedInPostUseCases
        from ...infrastructure.repositories.linkedin_post_repository import LinkedInPostRepository
        from ...application.use_cases.linkedin_post_use_cases import LinkedInPostUseCases
        from ...infrastructure.repositories.linkedin_post_repository import LinkedInPostRepository
from typing import Any, List, Dict, Optional
"""
Advanced Debugging Tools with Best Libraries
===========================================

Advanced debugging tools using memory_profiler, tracemalloc, cProfile, and other libraries.
"""


# Advanced debugging libraries

# Our modules

# Import fixtures and factories
    LinkedInPostFactory,
    PostDataFactory,
    test_data_generator
)


class AdvancedDebugger:
    """Advanced debugging utility with comprehensive profiling and monitoring."""
    
    def __init__(self, enable_tracemalloc: bool = True, enable_profiling: bool = True):
        
    """__init__ function."""
self.enable_tracemalloc = enable_tracemalloc
        self.enable_profiling = enable_profiling
        self.tracemalloc_started = False
        self.profiler = None
        self.debug_logger = self._setup_debug_logger()
        
        if self.enable_tracemalloc:
            self._start_tracemalloc()
    
    def _setup_debug_logger(self) -> logging.Logger:
        """Setup debug logger with comprehensive formatting."""
        logger = logging.getLogger("advanced_debugger")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler("advanced_debug.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter with detailed information
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _start_tracemalloc(self) -> Any:
        """Start tracemalloc for memory tracking."""
        if not self.tracemalloc_started:
            tracemalloc.start(25)  # Keep 25 frames
            self.tracemalloc_started = True
            self.debug_logger.info("Tracemalloc started")
    
    def _stop_tracemalloc(self) -> Any:
        """Stop tracemalloc and return memory statistics."""
        if self.tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.tracemalloc_started = False
            
            return {
                "current_memory_mb": current / 1024 / 1024,
                "peak_memory_mb": peak / 1024 / 1024,
                "snapshots": tracemalloc.take_snapshot()
            }
        return None
    
    @contextmanager
    def memory_tracking(self, operation_name: str = "operation"):
        """Context manager for memory tracking."""
        if self.enable_tracemalloc:
            self._start_tracemalloc()
            snapshot1 = tracemalloc.take_snapshot()
        
        start_time = time.time()
        start_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.debug_logger.info(
                f"Memory tracking for {operation_name}: "
                f"Duration: {duration:.3f}s, "
                f"Memory delta: {memory_delta:.2f} MB"
            )
            
            if self.enable_tracemalloc:
                snapshot2 = tracemalloc.take_snapshot()
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                
                self.debug_logger.info(f"Top memory differences for {operation_name}:")
                for stat in top_stats[:5]:
                    self.debug_logger.info(f"  {stat}")
    
    @contextmanager
    def performance_profiling(self, operation_name: str = "operation"):
        """Context manager for performance profiling."""
        if self.enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if self.enable_profiling and self.profiler:
                self.profiler.disable()
                
                # Create stats
                s = io.StringIO()
                ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 functions
                
                self.debug_logger.info(
                    f"Performance profiling for {operation_name}: "
                    f"Duration: {duration:.3f}s"
                )
                self.debug_logger.info(f"Profile stats:\n{s.getvalue()}")
    
    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile a function execution."""
        with self.performance_profiling(func.__name__):
            with self.memory_tracking(func.__name__):
                return func(*args, **kwargs)
    
    async def profile_async_function(self, func: Callable, *args, **kwargs):
        """Profile an async function execution."""
        with self.performance_profiling(func.__name__):
            with self.memory_tracking(func.__name__):
                return await func(*args, **kwargs)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        process = psutil.Process(os.getpid())
        
        return {
            "process_info": {
                "pid": process.pid,
                "name": process.name(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": {
                    "rss_mb": process.memory_info().rss / 1024 / 1024,
                    "vms_mb": process.memory_info().vms / 1024 / 1024
                },
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None
            },
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "disk_usage_percent": psutil.disk_usage('/').percent
            },
            "python_info": {
                "version": sys.version,
                "implementation": sys.implementation.name,
                "gc_stats": gc.get_stats()
            }
        }
    
    def analyze_memory_objects(self, top_n: int = 10):
        """Analyze memory objects using objgraph."""
        self.debug_logger.info("Analyzing memory objects...")
        
        # Get most common object types
        objgraph.show_most_common_types(limit=top_n)
        
        # Get growth statistics
        objgraph.show_growth(limit=top_n)
        
        # Get chain of references for specific objects
        # This is useful for finding memory leaks
        self.debug_logger.info("Memory analysis completed")
    
    def force_garbage_collection(self) -> Any:
        """Force garbage collection and report statistics."""
        self.debug_logger.info("Forcing garbage collection...")
        
        # Get initial stats
        initial_stats = gc.get_stats()
        
        # Force collection
        collected = gc.collect()
        
        # Get final stats
        final_stats = gc.get_stats()
        
        self.debug_logger.info(f"Garbage collection completed: {collected} objects collected")
        self.debug_logger.info(f"Initial stats: {initial_stats}")
        self.debug_logger.info(f"Final stats: {final_stats}")
        
        return {
            "collected_objects": collected,
            "initial_stats": initial_stats,
            "final_stats": final_stats
        }


class MemoryLeakDetector:
    """Advanced memory leak detection utility."""
    
    def __init__(self) -> Any:
        self.snapshots = []
        self.debug_logger = logging.getLogger("memory_leak_detector")
    
    def take_snapshot(self, name: str = None):
        """Take a memory snapshot."""
        snapshot = {
            "timestamp": datetime.now(),
            "name": name or f"snapshot_{len(self.snapshots)}",
            "tracemalloc_snapshot": tracemalloc.take_snapshot(),
            "memory_usage": memory_usage(-1, interval=0.1, timeout=1)[0],
            "gc_stats": gc.get_stats()
        }
        
        self.snapshots.append(snapshot)
        self.debug_logger.info(f"Memory snapshot taken: {snapshot['name']}")
        
        return snapshot
    
    def compare_snapshots(self, snapshot1_name: str, snapshot2_name: str):
        """Compare two memory snapshots."""
        snapshot1 = next((s for s in self.snapshots if s["name"] == snapshot1_name), None)
        snapshot2 = next((s for s in self.snapshots if s["name"] == snapshot2_name), None)
        
        if not snapshot1 or not snapshot2:
            raise ValueError("Snapshot not found")
        
        # Compare tracemalloc snapshots
        top_stats = snapshot2["tracemalloc_snapshot"].compare_to(
            snapshot1["tracemalloc_snapshot"], 'lineno'
        )
        
        # Compare memory usage
        memory_delta = snapshot2["memory_usage"] - snapshot1["memory_usage"]
        
        # Compare GC stats
        gc_delta = {}
        for stat_name in snapshot1["gc_stats"][0].keys():
            gc_delta[stat_name] = (
                snapshot2["gc_stats"][0][stat_name] - 
                snapshot1["gc_stats"][0][stat_name]
            )
        
        comparison = {
            "snapshot1": snapshot1_name,
            "snapshot2": snapshot2_name,
            "memory_delta_mb": memory_delta,
            "gc_stats_delta": gc_delta,
            "top_memory_changes": top_stats[:10]
        }
        
        self.debug_logger.info(f"Snapshot comparison: {comparison}")
        return comparison
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0):
        """Detect potential memory leaks."""
        if len(self.snapshots) < 2:
            return []
        
        leaks = []
        
        for i in range(1, len(self.snapshots)):
            comparison = self.compare_snapshots(
                self.snapshots[i-1]["name"],
                self.snapshots[i]["name"]
            )
            
            if comparison["memory_delta_mb"] > threshold_mb:
                leaks.append({
                    "between_snapshots": f"{self.snapshots[i-1]['name']} -> {self.snapshots[i]['name']}",
                    "memory_increase_mb": comparison["memory_delta_mb"],
                    "top_changes": comparison["top_memory_changes"][:5]
                })
        
        return leaks


class PerformanceAnalyzer:
    """Advanced performance analysis utility."""
    
    def __init__(self) -> Any:
        self.profiler = Profiler()
        self.debug_logger = logging.getLogger("performance_analyzer")
    
    @contextmanager
    def profile_session(self, session_name: str = "session"):
        """Profile a session of operations."""
        self.profiler.start()
        
        try:
            yield
        finally:
            self.profiler.stop()
            
            # Generate HTML report
            html_output = self.profiler.output_html()
            
            # Save to file
            filename = f"profile_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(html_output)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            self.debug_logger.info(f"Performance profile saved to: {filename}")
    
    def analyze_function_performance(self, func: Callable, iterations: int = 1000):
        """Analyze function performance with multiple iterations."""
        times = []
        memory_usage_list = []
        
        for i in range(iterations):
            start_time = time.time()
            start_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
            
            result = func()
            
            end_time = time.time()
            end_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
            
            times.append(end_time - start_time)
            memory_usage_list.append(end_memory - start_memory)
        
        analysis = {
            "iterations": iterations,
            "total_time": sum(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev_time": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_memory_delta": sum(memory_usage_list) / len(memory_usage_list),
            "max_memory_delta": max(memory_usage_list),
            "percentiles": {
                "p50": statistics.quantiles(times, n=2)[0] if len(times) > 1 else times[0],
                "p95": statistics.quantiles(times, n=20)[18] if len(times) > 19 else times[-1],
                "p99": statistics.quantiles(times, n=100)[98] if len(times) > 99 else times[-1]
            }
        }
        
        self.debug_logger.info(f"Function performance analysis: {analysis}")
        return analysis


class AsyncDebugger:
    """Advanced async debugging utility."""
    
    def __init__(self) -> Any:
        self.debug_logger = logging.getLogger("async_debugger")
        self.task_times = {}
        self.task_memory = {}
    
    async def debug_async_function(self, func: Callable, *args, **kwargs):
        """Debug an async function with comprehensive monitoring."""
        task_name = func.__name__
        start_time = time.time()
        start_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
        
        try:
            result = await func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.task_times[task_name] = duration
            self.task_memory[task_name] = memory_delta
            
            self.debug_logger.info(
                f"Async function {task_name}: "
                f"Duration: {duration:.3f}s, "
                f"Memory delta: {memory_delta:.2f} MB"
            )
            
            return result
            
        except Exception as e:
            self.debug_logger.error(f"Async function {task_name} failed: {e}")
            raise
    
    async def debug_concurrent_tasks(self, tasks: List[Callable], max_concurrent: int = 10):
        """Debug multiple concurrent tasks."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task) -> Any:
            async with semaphore:
                return await self.debug_async_function(task)
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[limited_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Analyze results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        analysis = {
            "total_tasks": len(tasks),
            "successful_tasks": len(successful),
            "failed_tasks": len(failed),
            "success_rate": len(successful) / len(tasks),
            "avg_task_time": sum(self.task_times.values()) / len(self.task_times) if self.task_times else 0,
            "avg_memory_delta": sum(self.task_memory.values()) / len(self.task_memory) if self.task_memory else 0
        }
        
        self.debug_logger.info(f"Concurrent tasks analysis: {analysis}")
        return results, analysis


class TestAdvancedDebugging:
    """Advanced debugging tests."""
    
    @pytest.fixture
    def debugger(self) -> Any:
        """Advanced debugger fixture."""
        return AdvancedDebugger()
    
    @pytest.fixture
    def memory_leak_detector(self) -> Any:
        """Memory leak detector fixture."""
        return MemoryLeakDetector()
    
    @pytest.fixture
    def performance_analyzer(self) -> Any:
        """Performance analyzer fixture."""
        return PerformanceAnalyzer()
    
    @pytest.fixture
    def async_debugger(self) -> Any:
        """Async debugger fixture."""
        return AsyncDebugger()
    
    def test_memory_tracking(self, debugger) -> Any:
        """Test memory tracking functionality."""
        with debugger.memory_tracking("test_operation"):
            # Simulate memory allocation
            large_list = [i for i in range(100000)]
            time.sleep(0.1)
        
        # Check that tracking completed
        assert True  # If we get here, tracking worked
    
    def test_performance_profiling(self, debugger) -> Any:
        """Test performance profiling functionality."""
        def test_function():
            
    """test_function function."""
time.sleep(0.1)
            return sum(range(1000))
        
        with debugger.performance_profiling("test_function"):
            result = test_function()
        
        assert result == 499500
    
    def test_system_info(self, debugger) -> Any:
        """Test system information gathering."""
        info = debugger.get_system_info()
        
        assert "process_info" in info
        assert "system_info" in info
        assert "python_info" in info
        assert "pid" in info["process_info"]
        assert "cpu_count" in info["system_info"]
    
    def test_memory_leak_detection(self, memory_leak_detector) -> Any:
        """Test memory leak detection."""
        # Take initial snapshot
        memory_leak_detector.take_snapshot("initial")
        
        # Simulate memory allocation
        large_objects = []
        for i in range(10):
            large_objects.append([j for j in range(10000)])
        
        # Take second snapshot
        memory_leak_detector.take_snapshot("after_allocation")
        
        # Compare snapshots
        comparison = memory_leak_detector.compare_snapshots("initial", "after_allocation")
        
        assert "memory_delta_mb" in comparison
        assert "top_memory_changes" in comparison
    
    def test_garbage_collection(self, debugger) -> Any:
        """Test garbage collection functionality."""
        # Create some objects
        objects = [object() for _ in range(1000)]
        
        # Force garbage collection
        stats = debugger.force_garbage_collection()
        
        assert "collected_objects" in stats
        assert "initial_stats" in stats
        assert "final_stats" in stats
    
    @pytest.mark.asyncio
    async def test_async_debugging(self, async_debugger) -> Any:
        """Test async debugging functionality."""
        async def test_async_function():
            
    """test_async_function function."""
await asyncio.sleep(0.1)
            return "test_result"
        
        result = await async_debugger.debug_async_function(test_async_function)
        
        assert result == "test_result"
        assert len(async_debugger.task_times) == 1
        assert len(async_debugger.task_memory) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_debugging(self, async_debugger) -> Any:
        """Test concurrent task debugging."""
        async def task1():
            
    """task1 function."""
await asyncio.sleep(0.1)
            return "task1"
        
        async def task2():
            
    """task2 function."""
await asyncio.sleep(0.1)
            return "task2"
        
        tasks = [task1, task2]
        results, analysis = await async_debugger.debug_concurrent_tasks(tasks)
        
        assert len(results) == 2
        assert analysis["success_rate"] == 1.0
        assert analysis["total_tasks"] == 2
    
    def test_function_profiling(self, debugger) -> Any:
        """Test function profiling."""
        def test_function():
            
    """test_function function."""
time.sleep(0.1)
            return "test"
        
        result = debugger.profile_function(test_function)
        
        assert result == "test"
    
    @pytest.mark.asyncio
    async def test_async_function_profiling(self, debugger) -> Any:
        """Test async function profiling."""
        async def test_async_function():
            
    """test_async_function function."""
await asyncio.sleep(0.1)
            return "test"
        
        result = await debugger.profile_async_function(test_async_function)
        
        assert result == "test"


class TestLinkedInPostsDebugging:
    """Debugging tests specific to LinkedIn posts functionality."""
    
    @pytest.fixture
    def debugger(self) -> Any:
        """Advanced debugger fixture."""
        return AdvancedDebugger()
    
    @pytest.fixture
    def memory_leak_detector(self) -> Any:
        """Memory leak detector fixture."""
        return MemoryLeakDetector()
    
    @pytest.mark.asyncio
    async def test_post_creation_debugging(self, debugger) -> Any:
        """Debug post creation process."""
        
        repository = LinkedInPostRepository()
        use_cases = LinkedInPostUseCases(repository)
        
        post_data = PostDataFactory()
        
        with debugger.memory_tracking("post_creation"):
            with debugger.performance_profiling("post_creation"):
                result = await use_cases.generate_post(
                    content=post_data["content"],
                    post_type=PostType.ANNOUNCEMENT,
                    tone=PostTone.PROFESSIONAL,
                    target_audience="professionals",
                    industry="technology"
                )
        
        assert result is not None
        assert result.content == post_data["content"]
    
    @pytest.mark.asyncio
    async def test_batch_creation_debugging(self, debugger, memory_leak_detector) -> Any:
        """Debug batch creation process."""
        
        repository = LinkedInPostRepository()
        use_cases = LinkedInPostUseCases(repository)
        
        # Take initial snapshot
        memory_leak_detector.take_snapshot("before_batch")
        
        batch_data = PostDataFactory.build_batch(10)
        
        with debugger.memory_tracking("batch_creation"):
            with debugger.performance_profiling("batch_creation"):
                result = await use_cases.batch_create_posts(batch_data)
        
        # Take final snapshot
        memory_leak_detector.take_snapshot("after_batch")
        
        # Check for memory leaks
        comparison = memory_leak_detector.compare_snapshots("before_batch", "after_batch")
        
        assert len(result) == 10
        assert comparison["memory_delta_mb"] < 100  # Should not leak more than 100MB
    
    @pytest.mark.asyncio
    async def test_cache_operations_debugging(self, debugger) -> Any:
        """Debug cache operations."""
        cache_manager = CacheManager(memory_size=100, memory_ttl=60)
        
        with debugger.memory_tracking("cache_operations"):
            with debugger.performance_profiling("cache_operations"):
                # Test cache operations
                await cache_manager.set("test_key", "test_value")
                value = await cache_manager.get("test_key")
                await cache_manager.delete("test_key")
        
        assert value == "test_value"
    
    def test_system_resources_during_operations(self, debugger) -> Any:
        """Test system resources during LinkedIn posts operations."""
        # Get initial system info
        initial_info = debugger.get_system_info()
        
        # Simulate intensive operations
        for _ in range(10):
            post_data = PostDataFactory()
            batch_data = PostDataFactory.build_batch(5)
        
        # Get final system info
        final_info = debugger.get_system_info()
        
        # Check that system resources are reasonable
        assert final_info["process_info"]["memory_info"]["rss_mb"] < 1000
        assert final_info["process_info"]["cpu_percent"] < 90


# Export classes
__all__ = [
    "AdvancedDebugger",
    "MemoryLeakDetector",
    "PerformanceAnalyzer",
    "AsyncDebugger",
    "TestAdvancedDebugging",
    "TestLinkedInPostsDebugging"
] 