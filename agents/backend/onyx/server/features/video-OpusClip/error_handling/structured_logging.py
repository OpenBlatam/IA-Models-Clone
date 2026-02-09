#!/usr/bin/env python3
"""
Structured Logging for Video-OpusClip
Comprehensive structured logging with context, parameters, and error tracking
"""

import logging
import json
import traceback
import inspect
import sys
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps
import hashlib
import threading
from collections import defaultdict

from .custom_exceptions import VideoOpusClipException


# ============================================================================
# STRUCTURED LOG DATA CLASSES
# ============================================================================

@dataclass
class LogContext:
    """Structured log context information"""
    module: str
    function: str
    line_number: int
    timestamp: datetime
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None


@dataclass
class LogParameters:
    """Structured log parameters"""
    args: List[Any]
    kwargs: Dict[str, Any]
    sensitive_fields: List[str] = None
    
    def __post_init__(self):
        if self.sensitive_fields is None:
            self.sensitive_fields = ['password', 'token', 'secret', 'key', 'api_key']


@dataclass
class ErrorDetails:
    """Structured error details"""
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    error_hash: Optional[str] = None
    severity: str = "ERROR"
    category: Optional[str] = None
    subcategory: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Structured performance metrics"""
    execution_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    database_queries: Optional[int] = None
    network_requests: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None


@dataclass
class StructuredLogEntry:
    """Complete structured log entry"""
    context: LogContext
    level: str
    message: str
    parameters: Optional[LogParameters] = None
    error_details: Optional[ErrorDetails] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    additional_data: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'context': asdict(self.context),
            'level': self.level,
            'message': self.message,
            'timestamp': self.context.timestamp.isoformat(),
            'request_id': self.context.request_id,
            'session_id': self.context.session_id,
            'user_id': self.context.user_id,
            'correlation_id': self.context.correlation_id,
            'module': self.context.module,
            'function': self.context.function,
            'line_number': self.context.line_number,
            'thread_id': self.context.thread_id,
            'process_id': self.context.process_id
        }
        
        if self.parameters:
            result['parameters'] = self._sanitize_parameters()
        
        if self.error_details:
            result['error_details'] = asdict(self.error_details)
        
        if self.performance_metrics:
            result['performance_metrics'] = asdict(self.performance_metrics)
        
        if self.additional_data:
            result['additional_data'] = self.additional_data
        
        if self.tags:
            result['tags'] = self.tags
        
        return result
    
    def _sanitize_parameters(self) -> Dict[str, Any]:
        """Sanitize parameters by masking sensitive fields"""
        if not self.parameters:
            return {}
        
        sanitized = {
            'args': self.parameters.args,
            'kwargs': self.parameters.kwargs.copy()
        }
        
        # Mask sensitive fields in kwargs
        for field in self.parameters.sensitive_fields:
            if field in sanitized['kwargs']:
                sanitized['kwargs'][field] = '***MASKED***'
        
        return sanitized


# ============================================================================
# STRUCTURED LOGGER CLASS
# ============================================================================

class StructuredLogger:
    """Structured logger with context tracking and error analysis"""
    
    def __init__(
        self,
        name: str = "video_opusclip",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        error_file: Optional[str] = None,
        performance_file: Optional[str] = None,
        enable_console: bool = True,
        enable_json: bool = True,
        max_log_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5
    ):
        self.name = name
        self.level = level
        self.log_file = log_file
        self.error_file = error_file
        self.performance_file = performance_file
        self.enable_console = enable_console
        self.enable_json = enable_json
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        
        # Initialize loggers
        self.logger = self._setup_logger()
        self.error_logger = self._setup_error_logger()
        self.performance_logger = self._setup_performance_logger()
        
        # Context tracking
        self._context_stack = []
        self._request_context = threading.local()
        self._error_tracker = ErrorTracker()
        self._performance_tracker = PerformanceTracker()
        
        # Statistics
        self._stats = {
            'total_logs': 0,
            'error_logs': 0,
            'warning_logs': 0,
            'info_logs': 0,
            'debug_logs': 0,
            'performance_logs': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main logger"""
        logger = logging.getLogger(f"{self.name}_main")
        logger.setLevel(self.level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            
            if self.enable_json:
                console_handler.setFormatter(StructuredJSONFormatter())
            else:
                console_handler.setFormatter(StructuredTextFormatter())
            
            logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_log_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.level)
            
            if self.enable_json:
                file_handler.setFormatter(StructuredJSONFormatter())
            else:
                file_handler.setFormatter(StructuredTextFormatter())
            
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_error_logger(self) -> logging.Logger:
        """Setup error logger"""
        if not self.error_file:
            return self.logger
        
        error_logger = logging.getLogger(f"{self.name}_errors")
        error_logger.setLevel(logging.ERROR)
        
        # Clear existing handlers
        error_logger.handlers.clear()
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.error_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredJSONFormatter())
        error_logger.addHandler(error_handler)
        
        return error_logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance logger"""
        if not self.performance_file:
            return self.logger
        
        perf_logger = logging.getLogger(f"{self.name}_performance")
        perf_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        perf_logger.handlers.clear()
        
        # Performance file handler
        perf_handler = logging.handlers.RotatingFileHandler(
            self.performance_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredJSONFormatter())
        perf_logger.addHandler(perf_handler)
        
        return perf_logger
    
    def _get_current_context(self) -> LogContext:
        """Get current logging context"""
        frame = inspect.currentframe().f_back.f_back
        module = frame.f_globals.get('__name__', 'unknown')
        function = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # Get request context if available
        request_id = getattr(self._request_context, 'request_id', None)
        session_id = getattr(self._request_context, 'session_id', None)
        user_id = getattr(self._request_context, 'user_id', None)
        correlation_id = getattr(self._request_context, 'correlation_id', None)
        
        return LogContext(
            module=module,
            function=function,
            line_number=line_number,
            timestamp=datetime.now(timezone.utc),
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            correlation_id=correlation_id,
            thread_id=threading.get_ident(),
            process_id=os.getpid()
        )
    
    def _create_log_entry(
        self,
        level: str,
        message: str,
        parameters: Optional[LogParameters] = None,
        error_details: Optional[ErrorDetails] = None,
        performance_metrics: Optional[PerformanceMetrics] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> StructuredLogEntry:
        """Create structured log entry"""
        context = self._get_current_context()
        
        return StructuredLogEntry(
            context=context,
            level=level,
            message=message,
            parameters=parameters,
            error_details=error_details,
            performance_metrics=performance_metrics,
            additional_data=additional_data,
            tags=tags
        )
    
    def _log_entry(self, log_entry: StructuredLogEntry) -> None:
        """Log structured entry"""
        # Update statistics
        self._stats['total_logs'] += 1
        self._stats[f'{log_entry.level.lower()}_logs'] += 1
        
        # Track errors
        if log_entry.error_details:
            self._error_tracker.track_error(log_entry.error_details)
        
        # Track performance
        if log_entry.performance_metrics:
            self._stats['performance_logs'] += 1
            self._performance_tracker.track_performance(log_entry.performance_metrics)
        
        # Choose appropriate logger
        if log_entry.error_details:
            logger = self.error_logger
        elif log_entry.performance_metrics:
            logger = self.performance_logger
        else:
            logger = self.logger
        
        # Log the entry
        log_data = log_entry.to_dict()
        log_message = json.dumps(log_data, default=str)
        
        if log_entry.level.upper() == 'DEBUG':
            logger.debug(log_message)
        elif log_entry.level.upper() == 'INFO':
            logger.info(log_message)
        elif log_entry.level.upper() == 'WARNING':
            logger.warning(log_message)
        elif log_entry.level.upper() == 'ERROR':
            logger.error(log_message)
        elif log_entry.level.upper() == 'CRITICAL':
            logger.critical(log_message)
    
    def debug(
        self,
        message: str,
        *args,
        **kwargs
    ) -> None:
        """Log debug message with structured context"""
        parameters = LogParameters(args=list(args), kwargs=kwargs)
        log_entry = self._create_log_entry('DEBUG', message, parameters=parameters)
        self._log_entry(log_entry)
    
    def info(
        self,
        message: str,
        *args,
        **kwargs
    ) -> None:
        """Log info message with structured context"""
        parameters = LogParameters(args=list(args), kwargs=kwargs)
        log_entry = self._create_log_entry('INFO', message, parameters=parameters)
        self._log_entry(log_entry)
    
    def warning(
        self,
        message: str,
        *args,
        **kwargs
    ) -> None:
        """Log warning message with structured context"""
        parameters = LogParameters(args=list(args), kwargs=kwargs)
        log_entry = self._create_log_entry('WARNING', message, parameters=parameters)
        self._log_entry(log_entry)
    
    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        *args,
        **kwargs
    ) -> None:
        """Log error message with structured context and error details"""
        parameters = LogParameters(args=list(args), kwargs=kwargs)
        
        error_details = None
        if error:
            error_details = self._extract_error_details(error)
        
        log_entry = self._create_log_entry('ERROR', message, parameters=parameters, error_details=error_details)
        self._log_entry(log_entry)
    
    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        *args,
        **kwargs
    ) -> None:
        """Log critical message with structured context and error details"""
        parameters = LogParameters(args=list(args), kwargs=kwargs)
        
        error_details = None
        if error:
            error_details = self._extract_error_details(error)
        
        log_entry = self._create_log_entry('CRITICAL', message, parameters=parameters, error_details=error_details)
        self._log_entry(log_entry)
    
    def _extract_error_details(self, error: Exception) -> ErrorDetails:
        """Extract structured error details from exception"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Generate error hash
        error_content = f"{error_type}:{error_message}"
        error_hash = hashlib.md5(error_content.encode()).hexdigest()
        
        # Get stack trace
        stack_trace = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        
        # Determine category and subcategory
        category, subcategory = self._categorize_error(error)
        
        # Determine severity
        severity = self._determine_severity(error)
        
        return ErrorDetails(
            error_type=error_type,
            error_message=error_message,
            error_code=getattr(error, 'error_code', None),
            stack_trace=stack_trace,
            error_hash=error_hash,
            severity=severity,
            category=category,
            subcategory=subcategory
        )
    
    def _categorize_error(self, error: Exception) -> Tuple[str, str]:
        """Categorize error based on type and context"""
        error_type = type(error).__name__
        
        # Import error categories
        from .custom_exceptions import (
            ValidationError, SecurityError, ScanningError, EnumerationError,
            AttackError, DatabaseError, NetworkError, FileSystemError, ConfigurationError
        )
        
        if isinstance(error, ValidationError):
            return "validation", "input_validation"
        elif isinstance(error, SecurityError):
            return "security", "authentication"
        elif isinstance(error, ScanningError):
            return "scanning", "port_scan"
        elif isinstance(error, EnumerationError):
            return "enumeration", "dns_enumeration"
        elif isinstance(error, AttackError):
            return "attack", "brute_force"
        elif isinstance(error, DatabaseError):
            return "database", "connection"
        elif isinstance(error, NetworkError):
            return "network", "connection_timeout"
        elif isinstance(error, FileSystemError):
            return "filesystem", "file_not_found"
        elif isinstance(error, ConfigurationError):
            return "configuration", "missing_config"
        else:
            return "general", "unknown"
    
    def _determine_severity(self, error: Exception) -> str:
        """Determine error severity"""
        error_type = type(error).__name__
        
        critical_errors = ['KeyboardInterrupt', 'SystemExit', 'MemoryError']
        if error_type in critical_errors:
            return "CRITICAL"
        
        # Check if it's a VideoOpusClipException with severity
        if hasattr(error, 'severity'):
            return error.severity
        
        return "ERROR"
    
    @contextmanager
    def context(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ):
        """Context manager for setting request context"""
        # Set context
        if request_id:
            self._request_context.request_id = request_id
        if session_id:
            self._request_context.session_id = session_id
        if user_id:
            self._request_context.user_id = user_id
        if correlation_id:
            self._request_context.correlation_id = correlation_id
        
        try:
            yield self
        finally:
            # Clear context
            if hasattr(self._request_context, 'request_id'):
                delattr(self._request_context, 'request_id')
            if hasattr(self._request_context, 'session_id'):
                delattr(self._request_context, 'session_id')
            if hasattr(self._request_context, 'user_id'):
                delattr(self._request_context, 'user_id')
            if hasattr(self._request_context, 'correlation_id'):
                delattr(self._request_context, 'correlation_id')
    
    @contextmanager
    def performance_tracking(self, operation: str):
        """Context manager for performance tracking"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory if start_memory and end_memory else None
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage
            )
            
            log_entry = self._create_log_entry(
                'INFO',
                f"Performance metrics for {operation}",
                performance_metrics=metrics,
                additional_data={'operation': operation}
            )
            self._log_entry(log_entry)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            **self._stats,
            'error_analysis': self._error_tracker.get_analysis(),
            'performance_analysis': self._performance_tracker.get_analysis()
        }


# ============================================================================
# LOG FORMATTERS
# ============================================================================

class StructuredJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        try:
            # Parse the message as JSON if it's already structured
            log_data = json.loads(record.getMessage())
            return json.dumps(log_data, default=str, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # Fallback to simple JSON format
            return json.dumps({
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line_number': record.lineno,
                'logger': record.name
            }, default=str, ensure_ascii=False)


class StructuredTextFormatter(logging.Formatter):
    """Text formatter for structured logging"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured text"""
        try:
            # Try to parse as JSON and format nicely
            log_data = json.loads(record.getMessage())
            
            # Extract key information
            context = log_data.get('context', {})
            module = context.get('module', record.module)
            function = context.get('function', record.funcName)
            line_number = context.get('line_number', record.lineno)
            request_id = context.get('request_id', 'N/A')
            user_id = context.get('user_id', 'N/A')
            
            # Format message
            formatted_message = f"{log_data.get('message', record.getMessage())}"
            
            # Add context information
            if request_id != 'N/A':
                formatted_message += f" | RequestID: {request_id}"
            if user_id != 'N/A':
                formatted_message += f" | UserID: {user_id}"
            
            # Add error details if present
            error_details = log_data.get('error_details')
            if error_details:
                formatted_message += f" | Error: {error_details.get('error_type')}: {error_details.get('error_message')}"
            
            # Add performance metrics if present
            perf_metrics = log_data.get('performance_metrics')
            if perf_metrics:
                execution_time = perf_metrics.get('execution_time', 0)
                formatted_message += f" | ExecutionTime: {execution_time:.3f}s"
            
            return f"{self.formatTime(record)} | {record.levelname} | {module}.{function}:{line_number} | {formatted_message}"
            
        except (json.JSONDecodeError, TypeError):
            # Fallback to standard format
            return super().format(record)


# ============================================================================
# ERROR TRACKER
# ============================================================================

class ErrorTracker:
    """Track and analyze errors"""
    
    def __init__(self):
        self.errors = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.error_timestamps = defaultdict(list)
    
    def track_error(self, error_details: ErrorDetails) -> None:
        """Track an error"""
        error_key = error_details.error_hash or f"{error_details.error_type}:{error_details.error_message}"
        
        self.errors[error_key].append(error_details)
        self.error_counts[error_key] += 1
        self.error_timestamps[error_key].append(error_details.timestamp)
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get error analysis"""
        analysis = {
            'total_errors': sum(self.error_counts.values()),
            'unique_errors': len(self.error_counts),
            'most_common_errors': [],
            'error_categories': defaultdict(int),
            'error_severities': defaultdict(int),
            'recent_errors': []
        }
        
        # Most common errors
        sorted_errors = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)
        analysis['most_common_errors'] = [
            {
                'error_type': error_key.split(':')[0] if ':' in error_key else error_key,
                'error_message': error_key.split(':', 1)[1] if ':' in error_key else '',
                'count': count,
                'last_occurrence': max(self.error_timestamps[error_key]).isoformat() if self.error_timestamps[error_key] else None
            }
            for error_key, count in sorted_errors[:10]
        ]
        
        # Error categories and severities
        for error_list in self.errors.values():
            for error in error_list:
                if error.category:
                    analysis['error_categories'][error.category] += 1
                if error.severity:
                    analysis['error_severities'][error.severity] += 1
        
        # Recent errors (last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        for error_list in self.errors.values():
            for error in error_list:
                if error.timestamp > cutoff_time:
                    analysis['recent_errors'].append({
                        'error_type': error.error_type,
                        'error_message': error.error_message,
                        'timestamp': error.timestamp.isoformat(),
                        'category': error.category,
                        'severity': error.severity
                    })
        
        # Sort recent errors by timestamp
        analysis['recent_errors'].sort(key=lambda x: x['timestamp'], reverse=True)
        analysis['recent_errors'] = analysis['recent_errors'][:50]  # Limit to 50 most recent
        
        return analysis


# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

class PerformanceTracker:
    """Track and analyze performance metrics"""
    
    def __init__(self):
        self.metrics = []
        self.operation_stats = defaultdict(list)
    
    def track_performance(self, metrics: PerformanceMetrics) -> None:
        """Track performance metrics"""
        self.metrics.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get performance analysis"""
        if not self.metrics:
            return {'total_operations': 0}
        
        execution_times = [m.execution_time for m in self.metrics]
        memory_usages = [m.memory_usage for m in self.metrics if m.memory_usage is not None]
        
        analysis = {
            'total_operations': len(self.metrics),
            'execution_time': {
                'min': min(execution_times),
                'max': max(execution_times),
                'avg': sum(execution_times) / len(execution_times),
                'p95': sorted(execution_times)[int(len(execution_times) * 0.95)],
                'p99': sorted(execution_times)[int(len(execution_times) * 0.99)]
            }
        }
        
        if memory_usages:
            analysis['memory_usage'] = {
                'min': min(memory_usages),
                'max': max(memory_usages),
                'avg': sum(memory_usages) / len(memory_usages),
                'p95': sorted(memory_usages)[int(len(memory_usages) * 0.95)],
                'p99': sorted(memory_usages)[int(len(memory_usages) * 0.99)]
            }
        
        return analysis


# ============================================================================
# LOGGING DECORATORS
# ============================================================================

def log_function_call(logger: StructuredLogger):
    """Decorator to log function calls with parameters and performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log function entry
            logger.info(
                f"Function {func.__name__} called",
                *args,
                **kwargs,
                tags=['function_call', 'entry']
            )
            
            # Track performance
            with logger.performance_tracking(func.__name__):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful completion
                    logger.info(
                        f"Function {func.__name__} completed successfully",
                        result=result,
                        tags=['function_call', 'success']
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log error
                    logger.error(
                        f"Function {func.__name__} failed",
                        error=e,
                        *args,
                        **kwargs,
                        tags=['function_call', 'error']
                    )
                    raise
        
        return wrapper
    return decorator


def log_errors(logger: StructuredLogger, reraise: bool = True):
    """Decorator to log errors with structured context"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}",
                    error=e,
                    *args,
                    **kwargs,
                    tags=['error_handler', 'caught_exception']
                )
                if reraise:
                    raise
        
        return wrapper
    return decorator


# ============================================================================
# GLOBAL LOGGER INSTANCE
# ============================================================================

# Create global logger instance
_global_logger = None

def get_logger(
    name: str = "video_opusclip",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    error_file: Optional[str] = None,
    performance_file: Optional[str] = None,
    enable_console: bool = True,
    enable_json: bool = True
) -> StructuredLogger:
    """Get or create global structured logger"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = StructuredLogger(
            name=name,
            level=level,
            log_file=log_file,
            error_file=error_file,
            performance_file=performance_file,
            enable_console=enable_console,
            enable_json=enable_json
        )
    
    return _global_logger


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of structured logging
    print("📝 Structured Logging Example")
    
    # Create logger
    logger = get_logger(
        name="video_opusclip_example",
        log_file="logs/app.log",
        error_file="logs/errors.log",
        performance_file="logs/performance.log",
        enable_console=True,
        enable_json=True
    )
    
    # Set request context
    with logger.context(
        request_id="req_123",
        session_id="sess_456",
        user_id="user_789",
        correlation_id="corr_101"
    ):
        # Log different types of messages
        logger.info("Application started", version="1.0.0", environment="development")
        
        logger.debug("Processing scan request", target="192.168.1.100", scan_type="port_scan")
        
        logger.warning("High memory usage detected", memory_usage=85.5, threshold=80.0)
        
        # Log error with exception
        try:
            raise ValueError("Invalid target address")
        except Exception as e:
            logger.error("Scan failed", error=e, target="invalid_target", scan_type="port_scan")
        
        # Performance tracking
        with logger.performance_tracking("database_query"):
            import time
            time.sleep(0.1)  # Simulate database query
        
        # Function call logging
        @log_function_call(logger)
        def example_function(param1: str, param2: int) -> str:
            return f"Processed {param1} with value {param2}"
        
        result = example_function("test_param", 42)
        print(f"Function result: {result}")
        
        # Error handling decorator
        @log_errors(logger, reraise=False)
        def risky_function():
            raise RuntimeError("Something went wrong")
        
        risky_function()  # This will be logged but not re-raised
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"\n📊 Logging Statistics:")
    print(f"Total logs: {stats['total_logs']}")
    print(f"Error logs: {stats['error_logs']}")
    print(f"Performance logs: {stats['performance_logs']}")
    
    if stats['error_analysis']['total_errors'] > 0:
        print(f"\n🚨 Error Analysis:")
        print(f"Total errors: {stats['error_analysis']['total_errors']}")
        print(f"Unique errors: {stats['error_analysis']['unique_errors']}")
        
        if stats['error_analysis']['most_common_errors']:
            print("Most common errors:")
            for error in stats['error_analysis']['most_common_errors'][:3]:
                print(f"  - {error['error_type']}: {error['error_message']} (count: {error['count']})")
    
    if stats['performance_analysis']['total_operations'] > 0:
        print(f"\n⚡ Performance Analysis:")
        perf = stats['performance_analysis']
        print(f"Total operations: {perf['total_operations']}")
        print(f"Average execution time: {perf['execution_time']['avg']:.3f}s")
        print(f"95th percentile: {perf['execution_time']['p95']:.3f}s") 