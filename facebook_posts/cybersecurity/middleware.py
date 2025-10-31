from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import functools
import logging
import traceback
import json
import sys
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque
import weakref
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Middleware for Cybersecurity Toolkit
Implements centralized logging, metrics, and exception handling.
"""


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MetricsData:
    """Metrics data structure."""
    operation: str
    duration: float
    success: bool
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

@dataclass
class LogEntry:
    """Log entry structure."""
    level: str
    message: str
    operation: Optional[str] = None
    user: Optional[str] = None
    target: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    traceback: Optional[str] = None

class MetricsCollector:
    """Centralized metrics collection."""
    
    def __init__(self, max_history: int = 1000):
        
    """__init__ function."""
self.metrics: deque = deque(maxlen=max_history)
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.errors: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_metric(self, metric: MetricsData):
        """Record a metric."""
        with self._lock:
            self.metrics.append(metric)
            self.counters[metric.operation] += 1
            
            if metric.success:
                self.timers[metric.operation].append(metric.duration)
            else:
                self.errors[metric.operation] += 1
                if metric.error_code:
                    self.errors[f"{metric.operation}_{metric.error_code}"] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            summary = {
                "total_operations": len(self.metrics),
                "operations": {},
                "errors": dict(self.errors),
                "performance": {}
            }
            
            # Calculate per-operation metrics
            for operation in self.counters:
                if operation in self.timers and self.timers[operation]:
                    times = self.timers[operation]
                    summary["operations"][operation] = {
                        "count": self.counters[operation],
                        "error_count": self.errors.get(operation, 0),
                        "success_rate": (self.counters[operation] - self.errors.get(operation, 0)) / self.counters[operation],
                        "avg_duration": sum(times) / len(times),
                        "min_duration": min(times),
                        "max_duration": max(times),
                        "p95_duration": sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0
                    }
            
            return summary
    
    def clear_metrics(self) -> Any:
        """Clear all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.timers.clear()
            self.errors.clear()

class CentralizedLogger:
    """Centralized logging with structured output."""
    
    def __init__(self, log_file: Optional[str] = None, max_entries: int = 10000):
        
    """__init__ function."""
self.log_entries: deque = deque(maxlen=max_entries)
        self.log_file = log_file
        self._lock = threading.Lock()
        
        # Configure file logging if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
    
    def log(self, level: str, message: str, operation: Optional[str] = None,
            user: Optional[str] = None, target: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
        """Log an entry."""
        entry = LogEntry(
            level=level,
            message=message,
            operation=operation,
            user=user,
            target=target,
            metadata=metadata,
            timestamp=time.time(),
            traceback=traceback.format_exc() if exception else None
        )
        
        with self._lock:
            self.log_entries.append(entry)
        
        # Also log to standard logger
        log_method = getattr(logger, level.lower(), logger.info)
        log_message = f"[{operation}] {message}"
        if user:
            log_message += f" (User: {user})"
        if target:
            log_message += f" (Target: {target})"
        
        log_method(log_message)
        
        if exception:
            logger.error(f"Exception in {operation}: {exception}")
    
    def get_logs(self, level: Optional[str] = None, operation: Optional[str] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get logs with optional filtering."""
        with self._lock:
            filtered_logs = []
            for entry in reversed(self.log_entries):
                if level and entry.level != level:
                    continue
                if operation and entry.operation != operation:
                    continue
                
                filtered_logs.append(asdict(entry))
                
                if limit and len(filtered_logs) >= limit:
                    break
            
            return filtered_logs

class ExceptionHandler:
    """Centralized exception handling."""
    
    def __init__(self, logger: CentralizedLogger, metrics: MetricsCollector):
        
    """__init__ function."""
self.logger = logger
        self.metrics = metrics
        self.exception_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def handle_exception(self, exception: Exception, operation: str, user: Optional[str] = None,
                        target: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Handle an exception."""
        exception_type = type(exception).__name__
        
        with self._lock:
            self.exception_counts[exception_type] += 1
        
        # Log the exception
        self.logger.log(
            level="ERROR",
            message=f"Exception occurred: {str(exception)}",
            operation=operation,
            user=user,
            target=target,
            metadata=metadata,
            exception=exception
        )
        
        # Record metrics
        self.metrics.record_metric(MetricsData(
            operation=operation,
            duration=0.0,
            success=False,
            error_code=exception_type,
            metadata=metadata,
            timestamp=time.time()
        ))
    
    def get_exception_summary(self) -> Dict[str, Any]:
        """Get exception summary."""
        with self._lock:
            return {
                "total_exceptions": sum(self.exception_counts.values()),
                "exception_counts": dict(self.exception_counts),
                "most_common": sorted(self.exception_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }

class MiddlewareManager:
    """Manages all middleware components."""
    
    def __init__(self, log_file: Optional[str] = None):
        
    """__init__ function."""
self.logger = CentralizedLogger(log_file)
        self.metrics = MetricsCollector()
        self.exception_handler = ExceptionHandler(self.logger, self.metrics)
        self._middleware_stack: List[Callable] = []
    
    def add_middleware(self, middleware_func: Callable):
        """Add middleware to the stack."""
        self._middleware_stack.append(middleware_func)
    
    def apply_middleware(self, func: Callable) -> Callable:
        """Apply all middleware to a function."""
        wrapped_func = func
        
        for middleware in self._middleware_stack:
            wrapped_func = middleware(wrapped_func)
        
        return wrapped_func

def logging_middleware(operation_name: str = None):
    """Middleware for centralized logging."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            # Extract user and target from kwargs if available
            user = kwargs.get('user', 'unknown')
            target = kwargs.get('target', 'unknown')
            
            # Get middleware manager from global context or create new
            middleware_manager = getattr(func, '_middleware_manager', None)
            if middleware_manager:
                middleware_manager.logger.log(
                    level="INFO",
                    message=f"Starting operation: {op_name}",
                    operation=op_name,
                    user=user,
                    target=target,
                    metadata={"args": str(args), "kwargs": str(kwargs)}
                )
            
            try:
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if middleware_manager:
                    middleware_manager.logger.log(
                        level="INFO",
                        message=f"Operation completed successfully: {op_name}",
                        operation=op_name,
                        user=user,
                        target=target,
                        metadata={"duration": duration, "success": True}
                    )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if middleware_manager:
                    middleware_manager.exception_handler.handle_exception(
                        e, op_name, user, target, {"duration": duration}
                    )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            user = kwargs.get('user', 'unknown')
            target = kwargs.get('target', 'unknown')
            
            middleware_manager = getattr(func, '_middleware_manager', None)
            if middleware_manager:
                middleware_manager.logger.log(
                    level="INFO",
                    message=f"Starting operation: {op_name}",
                    operation=op_name,
                    user=user,
                    target=target,
                    metadata={"args": str(args), "kwargs": str(kwargs)}
                )
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if middleware_manager:
                    middleware_manager.logger.log(
                        level="INFO",
                        message=f"Operation completed successfully: {op_name}",
                        operation=op_name,
                        user=user,
                        target=target,
                        metadata={"duration": duration, "success": True}
                    )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if middleware_manager:
                    middleware_manager.exception_handler.handle_exception(
                        e, op_name, user, target, {"duration": duration}
                    )
                
                raise
        
        # Return async or sync wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def metrics_middleware(operation_name: str = None):
    """Middleware for metrics collection."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            middleware_manager = getattr(func, '_middleware_manager', None)
            
            try:
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if middleware_manager:
                    middleware_manager.metrics.record_metric(MetricsData(
                        operation=op_name,
                        duration=duration,
                        success=True,
                        metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
                    ))
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if middleware_manager:
                    middleware_manager.metrics.record_metric(MetricsData(
                        operation=op_name,
                        duration=duration,
                        success=False,
                        error_code=type(e).__name__,
                        metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
                    ))
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            middleware_manager = getattr(func, '_middleware_manager', None)
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if middleware_manager:
                    middleware_manager.metrics.record_metric(MetricsData(
                        operation=op_name,
                        duration=duration,
                        success=True,
                        metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
                    ))
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if middleware_manager:
                    middleware_manager.metrics.record_metric(MetricsData(
                        operation=op_name,
                        duration=duration,
                        success=False,
                        error_code=type(e).__name__,
                        metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
                    ))
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def exception_handling_middleware(operation_name: str = None):
    """Middleware for exception handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            
            middleware_manager = getattr(func, '_middleware_manager', None)
            
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                user = kwargs.get('user', 'unknown')
                target = kwargs.get('target', 'unknown')
                
                if middleware_manager:
                    middleware_manager.exception_handler.handle_exception(
                        e, op_name, user, target
                    )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            
            middleware_manager = getattr(func, '_middleware_manager', None)
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                user = kwargs.get('user', 'unknown')
                target = kwargs.get('target', 'unknown')
                
                if middleware_manager:
                    middleware_manager.exception_handler.handle_exception(
                        e, op_name, user, target
                    )
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def apply_middleware(func: Callable, operation_name: str = None, 
                    middleware_manager: MiddlewareManager = None) -> Callable:
    """Apply all middleware to a function."""
    if middleware_manager:
        func._middleware_manager = middleware_manager
    
    # Apply all middleware decorators
    func = logging_middleware(operation_name)(func)
    func = metrics_middleware(operation_name)(func)
    func = exception_handling_middleware(operation_name)(func)
    
    return func

# Global middleware manager instance
global_middleware_manager = MiddlewareManager()

def get_middleware_manager() -> MiddlewareManager:
    """Get the global middleware manager."""
    return global_middleware_manager

def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary from global middleware manager."""
    return global_middleware_manager.metrics.get_metrics_summary()

def get_logs(level: Optional[str] = None, operation: Optional[str] = None,
             limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get logs from global middleware manager."""
    return global_middleware_manager.logger.get_logs(level, operation, limit)

def get_exception_summary() -> Dict[str, Any]:
    """Get exception summary from global middleware manager."""
    return global_middleware_manager.exception_handler.get_exception_summary()

def clear_metrics():
    """Clear metrics from global middleware manager."""
    global_middleware_manager.metrics.clear_metrics() 