from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import traceback
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging
from functools import wraps
import inspect
import sys
import os
from contextlib import contextmanager
import psutil
import gc
import orjson
from memory_profiler import profile
import tracemalloc
from typing import Any, List, Dict, Optional
"""
Advanced Debug Tools for LinkedIn Posts API
===========================================

Comprehensive debugging utilities for development and troubleshooting.
"""


# Third-party imports


class APIDebugger:
    """
    Advanced API debugger with comprehensive monitoring and analysis.
    """
    
    def __init__(self, enable_logging: bool = True, enable_profiling: bool = True):
        
    """__init__ function."""
self.enable_logging = enable_logging
        self.enable_profiling = enable_profiling
        self.debug_logger = self._setup_logger()
        self.performance_metrics = {}
        self.error_tracker = []
        self.request_tracker = []
        
        # Start memory tracking
        if enable_profiling:
            tracemalloc.start()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup debug logger."""
        logger = logging.getLogger("api_debugger")
        logger.setLevel(logging.DEBUG)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler("api_debug.log")
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_request(self, method: str, url: str, duration: float, status_code: int, **kwargs):
        """Log API request details."""
        if not self.enable_logging:
            return
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "url": url,
            "duration": duration,
            "status_code": status_code,
            **kwargs
        }
        
        self.request_tracker.append(log_data)
        self.debug_logger.info(f"Request: {method} {url} - {duration:.3f}s - {status_code}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context."""
        if not self.enable_logging:
            return
        
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.error_tracker.append(error_data)
        self.debug_logger.error(f"Error: {type(error).__name__}: {error}")
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        if not self.enable_logging:
            return
        
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        metric_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "duration": duration,
            **kwargs
        }
        
        self.performance_metrics[operation].append(metric_data)
        self.debug_logger.info(f"Performance: {operation} - {duration:.3f}s")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get detailed memory snapshot."""
        if not self.enable_profiling:
            return {}
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return {
            "top_allocations": [
                {
                    "file": stat.traceback.format()[0],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                }
                for stat in top_stats[:10]
            ],
            "total_allocated_mb": sum(stat.size for stat in top_stats) / 1024 / 1024
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics."""
        analysis = {}
        
        for operation, metrics in self.performance_metrics.items():
            if not metrics:
                continue
            
            durations = [m["duration"] for m in metrics]
            analysis[operation] = {
                "count": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "p95_duration": sorted(durations)[int(len(durations) * 0.95)],
                "p99_duration": sorted(durations)[int(len(durations) * 0.99)]
            }
        
        return analysis
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        if not self.error_tracker:
            return {"total_errors": 0}
        
        error_types = {}
        for error in self.error_tracker:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_tracker),
            "error_types": error_types,
            "recent_errors": self.error_tracker[-5:]  # Last 5 errors
        }
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_usage": self.get_memory_usage(),
            "memory_snapshot": self.get_memory_snapshot(),
            "performance_analysis": self.analyze_performance(),
            "error_summary": self.get_error_summary(),
            "request_summary": {
                "total_requests": len(self.request_tracker),
                "recent_requests": self.request_tracker[-10:]  # Last 10 requests
            }
        }
    
    def save_debug_report(self, filename: str = None):
        """Save debug report to file."""
        if filename is None:
            filename = f"debug_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_debug_report()
        
        with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2, default=str)
        
        self.debug_logger.info(f"Debug report saved to {filename}")
        return filename


class PerformanceProfiler:
    """
    Performance profiler for detailed analysis.
    """
    
    def __init__(self) -> Any:
        self.profiles = {}
        self.active_profiles = {}
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            duration = end_time - start_time
            memory_diff = end_memory["rss_mb"] - start_memory["rss_mb"]
            
            profile_data = {
                "duration": duration,
                "memory_start_mb": start_memory["rss_mb"],
                "memory_end_mb": end_memory["rss_mb"],
                "memory_diff_mb": memory_diff,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []
            
            self.profiles[operation_name].append(profile_data)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024
        }
    
    def start_profile(self, profile_name: str):
        """Start a named profile."""
        self.active_profiles[profile_name] = {
            "start_time": time.time(),
            "start_memory": self.get_memory_usage()
        }
    
    def end_profile(self, profile_name: str):
        """End a named profile."""
        if profile_name not in self.active_profiles:
            return
        
        start_data = self.active_profiles[profile_name]
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        duration = end_time - start_data["start_time"]
        memory_diff = end_memory["rss_mb"] - start_data["start_memory"]["rss_mb"]
        
        profile_data = {
            "duration": duration,
            "memory_start_mb": start_data["start_memory"]["rss_mb"],
            "memory_end_mb": end_memory["rss_mb"],
            "memory_diff_mb": memory_diff,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if profile_name not in self.profiles:
            self.profiles[profile_name] = []
        
        self.profiles[profile_name].append(profile_data)
        del self.active_profiles[profile_name]
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles."""
        summary = {}
        
        for profile_name, profile_data in self.profiles.items():
            if not profile_data:
                continue
            
            durations = [p["duration"] for p in profile_data]
            memory_diffs = [p["memory_diff_mb"] for p in profile_data]
            
            summary[profile_name] = {
                "count": len(profile_data),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "avg_memory_diff": sum(memory_diffs) / len(memory_diffs),
                "total_memory_diff": sum(memory_diffs)
            }
        
        return summary


class AsyncDebugger:
    """
    Debugger specifically for async operations.
    """
    
    def __init__(self) -> Any:
        self.async_operations = {}
        self.task_tracker = {}
    
    async def debug_async_operation(self, operation_name: str, coro):
        """Debug an async operation."""
        start_time = time.time()
        task_id = id(coro)
        
        self.task_tracker[task_id] = {
            "operation": operation_name,
            "start_time": start_time,
            "status": "running"
        }
        
        try:
            result = await coro
            duration = time.time() - start_time
            
            self.task_tracker[task_id]["status"] = "completed"
            self.task_tracker[task_id]["duration"] = duration
            self.task_tracker[task_id]["success"] = True
            
            if operation_name not in self.async_operations:
                self.async_operations[operation_name] = []
            
            self.async_operations[operation_name].append({
                "duration": duration,
                "success": True,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.task_tracker[task_id]["status"] = "failed"
            self.task_tracker[task_id]["duration"] = duration
            self.task_tracker[task_id]["success"] = False
            self.task_tracker[task_id]["error"] = str(e)
            
            if operation_name not in self.async_operations:
                self.async_operations[operation_name] = []
            
            self.async_operations[operation_name].append({
                "duration": duration,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            raise
    
    def get_async_summary(self) -> Dict[str, Any]:
        """Get summary of async operations."""
        summary = {}
        
        for operation_name, operations in self.async_operations.items():
            if not operations:
                continue
            
            successful = [op for op in operations if op["success"]]
            failed = [op for op in operations if not op["success"]]
            
            durations = [op["duration"] for op in operations]
            
            summary[operation_name] = {
                "total": len(operations),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(operations) if operations else 0,
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0
            }
        
        return summary


class CacheDebugger:
    """
    Debugger for cache operations.
    """
    
    def __init__(self, cache_manager) -> Any:
        self.cache_manager = cache_manager
        self.cache_operations = []
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def debug_get(self, key: str, **kwargs):
        """Debug cache get operation."""
        start_time = time.time()
        
        try:
            result = await self.cache_manager.get(key, **kwargs)
            duration = time.time() - start_time
            
            operation = {
                "type": "get",
                "key": key,
                "duration": duration,
                "hit": result is not None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.cache_operations.append(operation)
            
            if result is not None:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            operation = {
                "type": "get",
                "key": key,
                "duration": duration,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.cache_operations.append(operation)
            raise
    
    async def debug_set(self, key: str, value: Any, **kwargs):
        """Debug cache set operation."""
        start_time = time.time()
        
        try:
            result = await self.cache_manager.set(key, value, **kwargs)
            duration = time.time() - start_time
            
            operation = {
                "type": "set",
                "key": key,
                "duration": duration,
                "success": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.cache_operations.append(operation)
            self.cache_stats["sets"] += 1
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            operation = {
                "type": "set",
                "key": key,
                "duration": duration,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.cache_operations.append(operation)
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_operations = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_operations if total_operations > 0 else 0
        
        return {
            **self.cache_stats,
            "total_operations": total_operations,
            "hit_rate": hit_rate,
            "recent_operations": self.cache_operations[-10:]  # Last 10 operations
        }


# Decorators for easy debugging
def debug_function(func: Callable) -> Callable:
    """Decorator to debug function calls."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        debugger = APIDebugger()
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            debugger.log_performance(
                f"{func.__name__}_async",
                duration,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            debugger.log_error(e, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
                "duration": duration
            })
            
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        debugger = APIDebugger()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            debugger.log_performance(
                f"{func.__name__}_sync",
                duration,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            debugger.log_error(e, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
                "duration": duration
            })
            
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        profiler = PerformanceProfiler()
        
        with profiler.profile_operation(f"{func.__name__}_async"):
            return await func(*args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        profiler = PerformanceProfiler()
        
        with profiler.profile_operation(f"{func.__name__}_sync"):
            return func(*args, **kwargs)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Utility functions
def print_debug_info(info: Dict[str, Any], title: str = "Debug Info"):
    """Print debug information in a formatted way."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print(f"{'='*50}\n")


def save_debug_data(data: Dict[str, Any], filename: str):
    """Save debug data to file."""
    with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(data, f, indent=2, default=str)
    
    print(f"Debug data saved to {filename}")


# Export classes and functions
__all__ = [
    "APIDebugger",
    "PerformanceProfiler",
    "AsyncDebugger",
    "CacheDebugger",
    "debug_function",
    "profile_memory",
    "print_debug_info",
    "save_debug_data"
] 