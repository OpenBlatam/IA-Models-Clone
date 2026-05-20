from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import functools
import inspect
import json
import logging
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple, Type, TypeVar, Awaitable
from enum import Enum
import threading
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import uuid
import hashlib
import base64
from datetime import datetime, timedelta
import statistics
import weakref
    import fastapi
    from fastapi import FastAPI, Request, Response, HTTPException
    from fastapi.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.types import ASGIApp
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    import redis
from typing import Any, List, Dict, Optional
"""
Middleware and Decorators for Centralized Logging, Metrics, and Exception Handling
=================================================================================

This module provides comprehensive middleware and decorator implementations for
centralized logging, metrics collection, and exception handling.

Features:
- Middleware for FastAPI and web frameworks
- Decorators for function and class methods
- Centralized logging with structured data
- Metrics collection and monitoring
- Exception handling and error reporting
- Performance monitoring and profiling
- Request/response tracking
- Security logging and audit trails
- Customizable middleware chains
- Async support and context management

Author: AI Assistant
License: MIT
"""


try:
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ExceptionSeverity(Enum):
    """Exception severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime = field(default_factory=datetime.now)
    level: LogLevel = LogLevel.INFO
    message: str = ""
    function_name: str = ""
    module_name: str = ""
    line_number: int = 0
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None


@dataclass
class MetricEntry:
    """Metric entry."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""


@dataclass
class ExceptionInfo:
    """Exception information."""
    exception_type: str
    exception_message: str
    severity: ExceptionSeverity
    function_name: str
    module_name: str
    line_number: int
    stack_trace: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceData:
    """Performance monitoring data."""
    function_name: str
    execution_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MiddlewareConfig:
    """Middleware configuration."""
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_exception_handling: bool = True
    enable_performance_monitoring: bool = True
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    metrics_backend: str = "memory"  # memory, prometheus, redis
    exception_reporting: bool = True
    performance_threshold: float = 1.0  # seconds
    request_id_header: str = "X-Request-ID"
    correlation_id_header: str = "X-Correlation-ID"
    user_id_header: str = "X-User-ID"
    session_id_header: str = "X-Session-ID"


class LoggingMiddleware:
    """Centralized logging middleware."""
    
    def __init__(self, config: MiddlewareConfig):
        """Initialize logging middleware."""
        self.config = config
        self.logger = logging.getLogger("middleware.logging")
        self.log_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        
        # Setup logging format
        if config.log_format == "json":
            self._setup_json_logging()
        else:
            self._setup_standard_logging()
    
    def _setup_json_logging(self) -> Any:
        """Setup JSON logging format."""
        class JSONFormatter(logging.Formatter):
            def format(self, record) -> Any:
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                        log_entry[key] = value
                
                return json.dumps(log_entry)
        
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, self.config.log_level.value.upper()))
    
    def _setup_standard_logging(self) -> Any:
        """Setup standard logging format."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, self.config.log_level.value.upper()))
    
    def log_request(self, request: Request, response: Response, execution_time: float):
        """Log HTTP request/response."""
        if not self.config.enable_logging:
            return
        
        # Extract headers
        request_id = request.headers.get(self.config.request_id_header)
        correlation_id = request.headers.get(self.config.correlation_id_header)
        user_id = request.headers.get(self.config.user_id_header)
        session_id = request.headers.get(self.config.session_id_header)
        
        # Create log entry
        log_entry = LogEntry(
            level=LogLevel.INFO,
            message=f"{request.method} {request.url.path} - {response.status_code}",
            function_name="http_request",
            module_name="middleware",
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id,
            extra_data={
                'method': request.method,
                'url': str(request.url),
                'status_code': response.status_code,
                'execution_time': execution_time,
                'user_agent': request.headers.get('user-agent'),
                'ip_address': request.client.host if request.client else None
            },
            performance_data=PerformanceData(
                function_name="http_request",
                execution_time=execution_time
            )
        )
        
        self._log_entry(log_entry)
    
    def log_function_call(self, function_name: str, module_name: str, line_number: int,
                         args: tuple, kwargs: dict, result: Any, execution_time: float,
                         exception: Optional[Exception] = None):
        """Log function call."""
        if not self.config.enable_logging:
            return
        
        # Determine log level
        level = LogLevel.ERROR if exception else LogLevel.INFO
        
        # Create log entry
        log_entry = LogEntry(
            level=level,
            message=f"Function call: {function_name}",
            function_name=function_name,
            module_name=module_name,
            line_number=line_number,
            extra_data={
                'args': str(args),
                'kwargs': str(kwargs),
                'result': str(result) if result is not None else None,
                'execution_time': execution_time
            },
            performance_data=PerformanceData(
                function_name=function_name,
                execution_time=execution_time
            )
        )
        
        # Add exception info if present
        if exception:
            log_entry.exception_info = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        self._log_entry(log_entry)
    
    def _log_entry(self, entry: LogEntry):
        """Log entry to configured backend."""
        with self._lock:
            # Add to history
            self.log_history.append(entry)
            
            # Log to configured backend
            log_message = self._format_log_entry(entry)
            
            if entry.level == LogLevel.DEBUG:
                self.logger.debug(log_message)
            elif entry.level == LogLevel.INFO:
                self.logger.info(log_message)
            elif entry.level == LogLevel.WARNING:
                self.logger.warning(log_message)
            elif entry.level == LogLevel.ERROR:
                self.logger.error(log_message)
            elif entry.level == LogLevel.CRITICAL:
                self.logger.critical(log_message)
    
    def _format_log_entry(self, entry: LogEntry) -> str:
        """Format log entry for output."""
        if self.config.log_format == "json":
            return json.dumps(asdict(entry), default=str)
        else:
            return f"{entry.timestamp} - {entry.level.value.upper()} - {entry.message} - {entry.function_name}"
    
    def get_log_history(self, limit: int = 100) -> List[LogEntry]:
        """Get recent log history."""
        with self._lock:
            return list(self.log_history)[-limit:]


class MetricsMiddleware:
    """Centralized metrics middleware."""
    
    def __init__(self, config: MiddlewareConfig):
        """Initialize metrics middleware."""
        self.config = config
        self.metrics: Dict[str, Any] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and config.metrics_backend == "prometheus":
            self._setup_prometheus_metrics()
        else:
            self._setup_memory_metrics()
    
    def _setup_prometheus_metrics(self) -> Any:
        """Setup Prometheus metrics."""
        self.request_counter = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint']
        )
        
        self.function_counter = Counter(
            'function_calls_total',
            'Total function calls',
            ['function_name', 'module_name']
        )
        
        self.function_duration = Histogram(
            'function_duration_seconds',
            'Function execution duration',
            ['function_name', 'module_name']
        )
        
        self.exception_counter = Counter(
            'exceptions_total',
            'Total exceptions',
            ['exception_type', 'function_name']
        )
    
    def _setup_memory_metrics(self) -> Any:
        """Setup in-memory metrics."""
        self.request_counter = None
        self.request_duration = None
        self.function_counter = None
        self.function_duration = None
        self.exception_counter = None
    
    def record_request_metric(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metric."""
        if not self.config.enable_metrics:
            return
        
        metric_entry = MetricEntry(
            name="http_request",
            value=1,
            metric_type=MetricType.COUNTER,
            labels={
                'method': method,
                'endpoint': endpoint,
                'status_code': str(status_code)
            },
            description=f"HTTP request: {method} {endpoint}"
        )
        
        self._record_metric(metric_entry)
        
        # Record duration
        duration_metric = MetricEntry(
            name="http_request_duration",
            value=duration,
            metric_type=MetricType.HISTOGRAM,
            labels={
                'method': method,
                'endpoint': endpoint
            },
            description=f"HTTP request duration: {method} {endpoint}"
        )
        
        self._record_metric(duration_metric)
        
        # Update Prometheus metrics if available
        if self.request_counter:
            self.request_counter.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_function_metric(self, function_name: str, module_name: str, duration: float, success: bool = True):
        """Record function call metric."""
        if not self.config.enable_metrics:
            return
        
        metric_entry = MetricEntry(
            name="function_call",
            value=1,
            metric_type=MetricType.COUNTER,
            labels={
                'function_name': function_name,
                'module_name': module_name,
                'success': str(success)
            },
            description=f"Function call: {function_name}"
        )
        
        self._record_metric(metric_entry)
        
        # Record duration
        duration_metric = MetricEntry(
            name="function_duration",
            value=duration,
            metric_type=MetricType.HISTOGRAM,
            labels={
                'function_name': function_name,
                'module_name': module_name
            },
            description=f"Function duration: {function_name}"
        )
        
        self._record_metric(duration_metric)
        
        # Update Prometheus metrics if available
        if self.function_counter:
            self.function_counter.labels(function_name=function_name, module_name=module_name).inc()
            self.function_duration.labels(function_name=function_name, module_name=module_name).observe(duration)
    
    def record_exception_metric(self, exception_type: str, function_name: str):
        """Record exception metric."""
        if not self.config.enable_metrics:
            return
        
        metric_entry = MetricEntry(
            name="exception",
            value=1,
            metric_type=MetricType.COUNTER,
            labels={
                'exception_type': exception_type,
                'function_name': function_name
            },
            description=f"Exception: {exception_type} in {function_name}"
        )
        
        self._record_metric(metric_entry)
        
        # Update Prometheus metrics if available
        if self.exception_counter:
            self.exception_counter.labels(exception_type=exception_type, function_name=function_name).inc()
    
    def _record_metric(self, metric: MetricEntry):
        """Record metric to backend."""
        with self._lock:
            self.metrics[metric.name].append(metric)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return {
                'metrics': dict(self.metrics),
                'summary': self._calculate_metrics_summary()
            }
    
    def _calculate_metrics_summary(self) -> Dict[str, Any]:
        """Calculate metrics summary."""
        summary = {}
        
        for metric_name, metric_list in self.metrics.items():
            if not metric_list:
                continue
            
            values = [m.value for m in metric_list]
            summary[metric_name] = {
                'count': len(values),
                'total': sum(values),
                'average': statistics.mean(values),
                'min': min(values),
                'max': max(values)
            }
            
            if len(values) > 1:
                summary[metric_name]['std_dev'] = statistics.stdev(values)
        
        return summary


class ExceptionHandlingMiddleware:
    """Centralized exception handling middleware."""
    
    def __init__(self, config: MiddlewareConfig):
        """Initialize exception handling middleware."""
        self.config = config
        self.exceptions: List[ExceptionInfo] = []
        self._lock = threading.Lock()
        self.exception_handlers: Dict[Type[Exception], Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> Any:
        """Register default exception handlers."""
        self.register_handler(ValueError, self._handle_value_error)
        self.register_handler(TypeError, self._handle_type_error)
        self.register_handler(KeyError, self._handle_key_error)
        self.register_handler(AttributeError, self._handle_attribute_error)
        self.register_handler(FileNotFoundError, self._handle_file_not_found)
        self.register_handler(PermissionError, self._handle_permission_error)
        self.register_handler(TimeoutError, self._handle_timeout_error)
        self.register_handler(ConnectionError, self._handle_connection_error)
    
    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register custom exception handler."""
        self.exception_handlers[exception_type] = handler
    
    def handle_exception(self, exception: Exception, function_name: str, module_name: str,
                        line_number: int, context_data: Dict[str, Any] = None) -> ExceptionInfo:
        """Handle exception and return exception info."""
        if not self.config.enable_exception_handling:
            return None
        
        # Determine severity
        severity = self._determine_severity(exception)
        
        # Create exception info
        exception_info = ExceptionInfo(
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            severity=severity,
            function_name=function_name,
            module_name=module_name,
            line_number=line_number,
            stack_trace=traceback.format_exc(),
            context_data=context_data or {}
        )
        
        # Store exception
        with self._lock:
            self.exceptions.append(exception_info)
        
        # Call custom handler if available
        handler = self._get_handler(exception)
        if handler:
            try:
                handler(exception_info)
            except Exception as e:
                logger.error(f"Exception handler failed: {e}")
        
        # Log exception
        if self.config.exception_reporting:
            self._log_exception(exception_info)
        
        return exception_info
    
    def _determine_severity(self, exception: Exception) -> ExceptionSeverity:
        """Determine exception severity."""
        exception_type = type(exception)
        
        # Critical exceptions
        if exception_type in (SystemError, KeyboardInterrupt, MemoryError):
            return ExceptionSeverity.CRITICAL
        
        # High severity exceptions
        if exception_type in (OSError, ConnectionError, TimeoutError):
            return ExceptionSeverity.HIGH
        
        # Medium severity exceptions
        if exception_type in (ValueError, TypeError, KeyError, AttributeError):
            return ExceptionSeverity.MEDIUM
        
        # Low severity exceptions
        return ExceptionSeverity.LOW
    
    def _get_handler(self, exception: Exception) -> Optional[Callable]:
        """Get handler for exception type."""
        exception_type = type(exception)
        
        # Check exact type match
        if exception_type in self.exception_handlers:
            return self.exception_handlers[exception_type]
        
        # Check parent class matches
        for base_type, handler in self.exception_handlers.items():
            if isinstance(exception, base_type):
                return handler
        
        return None
    
    def _log_exception(self, exception_info: ExceptionInfo):
        """Log exception information."""
        log_level = {
            ExceptionSeverity.LOW: logging.DEBUG,
            ExceptionSeverity.MEDIUM: logging.WARNING,
            ExceptionSeverity.HIGH: logging.ERROR,
            ExceptionSeverity.CRITICAL: logging.CRITICAL
        }.get(exception_info.severity, logging.ERROR)
        
        logger.log(log_level, f"Exception in {exception_info.function_name}: {exception_info.exception_message}")
    
    # Default exception handlers
    def _handle_value_error(self, exception_info: ExceptionInfo):
        """Handle ValueError."""
        logger.warning(f"Value error in {exception_info.function_name}: {exception_info.exception_message}")
    
    def _handle_type_error(self, exception_info: ExceptionInfo):
        """Handle TypeError."""
        logger.warning(f"Type error in {exception_info.function_name}: {exception_info.exception_message}")
    
    def _handle_key_error(self, exception_info: ExceptionInfo):
        """Handle KeyError."""
        logger.warning(f"Key error in {exception_info.function_name}: {exception_info.exception_message}")
    
    def _handle_attribute_error(self, exception_info: ExceptionInfo):
        """Handle AttributeError."""
        logger.warning(f"Attribute error in {exception_info.function_name}: {exception_info.exception_message}")
    
    def _handle_file_not_found(self, exception_info: ExceptionInfo):
        """Handle FileNotFoundError."""
        logger.error(f"File not found in {exception_info.function_name}: {exception_info.exception_message}")
    
    def _handle_permission_error(self, exception_info: ExceptionInfo):
        """Handle PermissionError."""
        logger.error(f"Permission error in {exception_info.function_name}: {exception_info.exception_message}")
    
    def _handle_timeout_error(self, exception_info: ExceptionInfo):
        """Handle TimeoutError."""
        logger.error(f"Timeout error in {exception_info.function_name}: {exception_info.exception_message}")
    
    def _handle_connection_error(self, exception_info: ExceptionInfo):
        """Handle ConnectionError."""
        logger.error(f"Connection error in {exception_info.function_name}: {exception_info.exception_message}")
    
    def get_exceptions(self, limit: int = 100) -> List[ExceptionInfo]:
        """Get recent exceptions."""
        with self._lock:
            return list(self.exceptions)[-limit:]


class PerformanceMiddleware:
    """Performance monitoring middleware."""
    
    def __init__(self, config: MiddlewareConfig):
        """Initialize performance middleware."""
        self.config = config
        self.performance_data: List[PerformanceData] = []
        self._lock = threading.Lock()
        self.slow_functions: Dict[str, List[float]] = defaultdict(list)
    
    def record_performance(self, function_name: str, execution_time: float,
                          memory_usage: Optional[float] = None, cpu_usage: Optional[float] = None,
                          input_size: Optional[int] = None, output_size: Optional[int] = None):
        """Record performance data."""
        if not self.config.enable_performance_monitoring:
            return
        
        performance_data = PerformanceData(
            function_name=function_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            input_size=input_size,
            output_size=output_size
        )
        
        with self._lock:
            self.performance_data.append(performance_data)
            
            # Track slow functions
            if execution_time > self.config.performance_threshold:
                self.slow_functions[function_name].append(execution_time)
        
        # Log slow functions
        if execution_time > self.config.performance_threshold:
            logger.warning(f"Slow function detected: {function_name} took {execution_time:.2f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.performance_data:
                return {}
            
            execution_times = [p.execution_time for p in self.performance_data]
            
            summary = {
                'total_calls': len(self.performance_data),
                'average_execution_time': statistics.mean(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'slow_functions': dict(self.slow_functions)
            }
            
            if len(execution_times) > 1:
                summary['std_dev_execution_time'] = statistics.stdev(execution_times)
            
            return summary
    
    def get_slow_functions(self, threshold: Optional[float] = None) -> Dict[str, List[float]]:
        """Get functions that exceed performance threshold."""
        threshold = threshold or self.config.performance_threshold
        
        with self._lock:
            return {
                func: times for func, times in self.slow_functions.items()
                if any(t > threshold for t in times)
            }


class MiddlewareManager:
    """Centralized middleware manager."""
    
    def __init__(self, config: MiddlewareConfig):
        """Initialize middleware manager."""
        self.config = config
        self.logging_middleware = LoggingMiddleware(config)
        self.metrics_middleware = MetricsMiddleware(config)
        self.exception_middleware = ExceptionHandlingMiddleware(config)
        self.performance_middleware = PerformanceMiddleware(config)
        self._lock = threading.Lock()
    
    def log_request(self, request: Request, response: Response, execution_time: float):
        """Log HTTP request."""
        self.logging_middleware.log_request(request, response, execution_time)
        self.metrics_middleware.record_request_metric(
            request.method, request.url.path, response.status_code, execution_time
        )
    
    def log_function_call(self, function_name: str, module_name: str, line_number: int,
                         args: tuple, kwargs: dict, result: Any, execution_time: float,
                         exception: Optional[Exception] = None):
        """Log function call."""
        self.logging_middleware.log_function_call(
            function_name, module_name, line_number, args, kwargs, result, execution_time, exception
        )
        
        self.metrics_middleware.record_function_metric(
            function_name, module_name, execution_time, exception is None
        )
        
        self.performance_middleware.record_performance(
            function_name, execution_time
        )
        
        if exception:
            self.exception_middleware.handle_exception(
                exception, function_name, module_name, line_number
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get middleware summary."""
        return {
            'logging': {
                'recent_logs': len(self.logging_middleware.get_log_history())
            },
            'metrics': self.metrics_middleware.get_metrics(),
            'exceptions': {
                'recent_exceptions': len(self.exception_middleware.get_exceptions())
            },
            'performance': self.performance_middleware.get_performance_summary()
        }


# Decorators
def log_function_call(middleware_manager: MiddlewareManager):
    """Decorator to log function calls."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function info
            function_name = func.__name__
            module_name = func.__module__
            line_number = inspect.getsourcelines(func)[1]
            
            # Record start time
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful execution
                middleware_manager.log_function_call(
                    function_name, module_name, line_number,
                    args, kwargs, result, execution_time
                )
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log failed execution
                middleware_manager.log_function_call(
                    function_name, module_name, line_number,
                    args, kwargs, None, execution_time, e
                )
                
                raise
        
        return wrapper
    return decorator


def async_log_function_call(middleware_manager: MiddlewareManager):
    """Decorator to log async function calls."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get function info
            function_name = func.__name__
            module_name = func.__module__
            line_number = inspect.getsourcelines(func)[1]
            
            # Record start time
            start_time = time.time()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful execution
                middleware_manager.log_function_call(
                    function_name, module_name, line_number,
                    args, kwargs, result, execution_time
                )
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log failed execution
                middleware_manager.log_function_call(
                    function_name, module_name, line_number,
                    args, kwargs, None, execution_time, e
                )
                
                raise
        
        return wrapper
    return decorator


def monitor_performance(threshold: float = 1.0):
    """Decorator to monitor function performance."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log slow functions
                if execution_time > threshold:
                    logger.warning(f"Slow function: {func.__name__} took {execution_time:.2f}s")
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


def handle_exceptions(exception_handlers: Dict[Type[Exception], Callable] = None):
    """Decorator to handle exceptions."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Call custom handler if available
                if exception_handlers and type(e) in exception_handlers:
                    exception_handlers[type(e)](e)
                else:
                    # Default handling
                    logger.error(f"Exception in {func.__name__}: {e}")
                    raise
        
        return wrapper
    return decorator


# FastAPI Middleware
if FASTAPI_AVAILABLE:
    class FastAPIMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for logging, metrics, and exception handling."""
        
        def __init__(self, app: ASGIApp, middleware_manager: MiddlewareManager):
            
    """__init__ function."""
super().__init__(app)
            self.middleware_manager = middleware_manager
        
        async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
            # Record start time
            start_time = time.time()
            
            try:
                # Process request
                response = await call_next(request)
                execution_time = time.time() - start_time
                
                # Log request
                self.middleware_manager.log_request(request, response, execution_time)
                
                return response
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Create error response
                error_response = Response(
                    content=json.dumps({"error": str(e)}),
                    status_code=500,
                    media_type="application/json"
                )
                
                # Log error request
                self.middleware_manager.log_request(request, error_response, execution_time)
                
                return error_response


# Context managers
@contextmanager
def performance_context(function_name: str, middleware_manager: MiddlewareManager):
    """Context manager for performance monitoring."""
    start_time = time.time()
    
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        middleware_manager.performance_middleware.record_performance(
            function_name, execution_time
        )


@asynccontextmanager
async def async_performance_context(function_name: str, middleware_manager: MiddlewareManager):
    """Async context manager for performance monitoring."""
    start_time = time.time()
    
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        middleware_manager.performance_middleware.record_performance(
            function_name, execution_time
        )


# Example usage functions
def demonstrate_middleware_usage():
    """Demonstrate middleware usage."""
    # Create middleware configuration
    config = MiddlewareConfig(
        enable_logging=True,
        enable_metrics=True,
        enable_exception_handling=True,
        enable_performance_monitoring=True,
        log_level=LogLevel.INFO,
        log_format="json"
    )
    
    # Create middleware manager
    middleware_manager = MiddlewareManager(config)
    
    # Example function with decorators
    @log_function_call(middleware_manager)
    @monitor_performance(threshold=0.1)
    @handle_exceptions()
    def example_function(x: int, y: int) -> int:
        """Example function with middleware."""
        if x < 0:
            raise ValueError("x must be positive")
        
        time.sleep(0.05)  # Simulate work
        return x + y
    
    # Example async function
    @async_log_function_call(middleware_manager)
    async def example_async_function(x: int, y: int) -> int:
        """Example async function with middleware."""
        await asyncio.sleep(0.05)  # Simulate async work
        return x * y
    
    # Test functions
    print("Testing middleware functionality...")
    
    # Test successful execution
    try:
        result = example_function(5, 3)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test exception handling
    try:
        result = example_function(-1, 3)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test async function
    async def test_async():
        
    """test_async function."""
result = await example_async_function(4, 6)
        print(f"Async result: {result}")
    
    asyncio.run(test_async())
    
    # Show summary
    summary = middleware_manager.get_summary()
    print(f"\nMiddleware Summary:")
    print(json.dumps(summary, indent=2, default=str))


def demonstrate_fastapi_middleware():
    """Demonstrate FastAPI middleware."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available, skipping FastAPI middleware demonstration")
        return
    
    # Create middleware manager
    config = MiddlewareConfig()
    middleware_manager = MiddlewareManager(config)
    
    # Create FastAPI app
    app = FastAPI()
    
    # Add middleware
    app.add_middleware(FastAPIMiddleware, middleware_manager=middleware_manager)
    
    # Example endpoints
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "Hello World"}
    
    @app.get("/slow")
    async def slow_endpoint():
        
    """slow_endpoint function."""
await asyncio.sleep(2)  # Simulate slow operation
        return {"message": "Slow response"}
    
    @app.get("/error")
    async def error_endpoint():
        
    """error_endpoint function."""
raise ValueError("Example error")
    
    print("FastAPI middleware demonstration")
    print("Start the server with: uvicorn main:app --reload")
    print("Test endpoints:")
    print("  GET / - Normal response")
    print("  GET /slow - Slow response")
    print("  GET /error - Error response")


def demonstrate_context_managers():
    """Demonstrate context managers."""
    config = MiddlewareConfig()
    middleware_manager = MiddlewareManager(config)
    
    # Performance context manager
    with performance_context("example_operation", middleware_manager):
        time.sleep(0.1)  # Simulate work
        print("Operation completed")
    
    # Async performance context manager
    async def async_operation():
        
    """async_operation function."""
async with async_performance_context("async_operation", middleware_manager):
            await asyncio.sleep(0.1)  # Simulate async work
            print("Async operation completed")
    
    asyncio.run(async_operation())
    
    # Show performance summary
    performance_summary = middleware_manager.performance_middleware.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(json.dumps(performance_summary, indent=2, default=str))


def main():
    """Main function demonstrating middleware and decorators."""
    logger.info("Starting middleware and decorators examples")
    
    # Demonstrate middleware usage
    try:
        demonstrate_middleware_usage()
    except Exception as e:
        logger.error(f"Middleware usage demonstration failed: {e}")
    
    # Demonstrate FastAPI middleware
    try:
        demonstrate_fastapi_middleware()
    except Exception as e:
        logger.error(f"FastAPI middleware demonstration failed: {e}")
    
    # Demonstrate context managers
    try:
        demonstrate_context_managers()
    except Exception as e:
        logger.error(f"Context managers demonstration failed: {e}")
    
    logger.info("Middleware and decorators examples completed")


match __name__:
    case "__main__":
    main() 