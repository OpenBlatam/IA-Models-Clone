"""
Advanced Logging System

This module provides a comprehensive logging system for the refactored HeyGen AI
architecture with structured logging, performance monitoring, and intelligent
log analysis.
"""

import logging
import json
import sys
import traceback
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import queue
import hashlib
import uuid
from contextvars import ContextVar
import psutil
import gc
from collections import defaultdict, deque
import re
import inspect
from functools import wraps
import socket
import os


# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(str, Enum):
    """Log categories."""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    AUDIT = "audit"
    DEBUG = "debug"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL = "external"


@dataclass
class LogEntry:
    """Log entry structure."""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    module: str
    function: str
    line_number: int
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io_bytes: int
    gc_collections: int
    thread_count: int
    process_count: int


class LogFilter:
    """Advanced log filtering system."""
    
    def __init__(self):
        self.filters: List[Callable[[LogEntry], bool]] = []
        self.sensitive_patterns = [
            r'password["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'token["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'secret["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'key["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'api_key["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'access_token["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'refresh_token["\']?\s*[:=]\s*["\']?([^"\']+)',
        ]
        self.sensitive_replacement = "***REDACTED***"
    
    def add_filter(self, filter_func: Callable[[LogEntry], bool]):
        """Add a filter function."""
        self.filters.append(filter_func)
    
    def should_log(self, entry: LogEntry) -> bool:
        """Check if log entry should be logged."""
        return all(filter_func(entry) for filter_func in self.filters)
    
    def sanitize_message(self, message: str) -> str:
        """Sanitize message by removing sensitive information."""
        sanitized = message
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(pattern, f'\\1={self.sensitive_replacement}', sanitized, flags=re.IGNORECASE)
        return sanitized


class LogFormatter:
    """Advanced log formatter with multiple output formats."""
    
    def __init__(self, format_type: str = "json"):
        self.format_type = format_type
        self.formatters = {
            "json": self._format_json,
            "text": self._format_text,
            "structured": self._format_structured,
            "compact": self._format_compact
        }
    
    def format(self, entry: LogEntry) -> str:
        """Format log entry."""
        formatter = self.formatters.get(self.format_type, self._format_json)
        return formatter(entry)
    
    def _format_json(self, entry: LogEntry) -> str:
        """Format as JSON."""
        data = {
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level.value,
            "category": entry.category.value,
            "message": entry.message,
            "module": entry.module,
            "function": entry.function,
            "line_number": entry.line_number,
            "request_id": entry.request_id,
            "user_id": entry.user_id,
            "session_id": entry.session_id,
            "extra_data": entry.extra_data,
            "exception_info": entry.exception_info,
            "performance_metrics": entry.performance_metrics,
            "tags": entry.tags
        }
        return json.dumps(data, default=str)
    
    def _format_text(self, entry: LogEntry) -> str:
        """Format as text."""
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"{timestamp} | {entry.level.value:8} | {entry.category.value:10} | {entry.module}:{entry.function}:{entry.line_number} | {entry.message}"
    
    def _format_structured(self, entry: LogEntry) -> str:
        """Format as structured text."""
        lines = [
            f"[{entry.timestamp.isoformat()}] {entry.level.value} - {entry.category.value}",
            f"  Module: {entry.module}:{entry.function}:{entry.line_number}",
            f"  Message: {entry.message}"
        ]
        
        if entry.request_id:
            lines.append(f"  Request ID: {entry.request_id}")
        if entry.user_id:
            lines.append(f"  User ID: {entry.user_id}")
        if entry.session_id:
            lines.append(f"  Session ID: {entry.session_id}")
        
        if entry.extra_data:
            lines.append(f"  Extra Data: {json.dumps(entry.extra_data, indent=2)}")
        
        if entry.exception_info:
            lines.append(f"  Exception: {entry.exception_info}")
        
        if entry.performance_metrics:
            lines.append(f"  Performance: {json.dumps(entry.performance_metrics, indent=2)}")
        
        if entry.tags:
            lines.append(f"  Tags: {', '.join(entry.tags)}")
        
        return "\n".join(lines)
    
    def _format_compact(self, entry: LogEntry) -> str:
        """Format as compact text."""
        timestamp = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
        return f"{timestamp} {entry.level.value[0]} {entry.category.value[:3]} {entry.module}:{entry.function} {entry.message}"


class LogHandler:
    """Advanced log handler with multiple outputs."""
    
    def __init__(self, name: str):
        self.name = name
        self.handlers: List[logging.Handler] = []
        self.queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.buffer: deque = deque(maxlen=1000)
        self.stats = {
            "total_logs": 0,
            "by_level": defaultdict(int),
            "by_category": defaultdict(int),
            "errors": 0,
            "warnings": 0
        }
    
    def add_handler(self, handler: logging.Handler):
        """Add a log handler."""
        self.handlers.append(handler)
    
    def start(self):
        """Start the log handler worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """Stop the log handler worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
    
    def _worker(self):
        """Worker thread for processing log entries."""
        while self.running:
            try:
                entry = self.queue.get(timeout=1.0)
                self._process_entry(entry)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in log handler worker: {e}")
    
    def _process_entry(self, entry: LogEntry):
        """Process a log entry."""
        # Update statistics
        self.stats["total_logs"] += 1
        self.stats["by_level"][entry.level.value] += 1
        self.stats["by_category"][entry.category.value] += 1
        
        if entry.level == LogLevel.ERROR:
            self.stats["errors"] += 1
        elif entry.level == LogLevel.WARNING:
            self.stats["warnings"] += 1
        
        # Add to buffer
        self.buffer.append(entry)
        
        # Send to handlers
        for handler in self.handlers:
            try:
                handler.emit(entry)
            except Exception as e:
                print(f"Error in log handler {handler}: {e}")
    
    def log(self, entry: LogEntry):
        """Add log entry to queue."""
        self.queue.put(entry)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return dict(self.stats)
    
    def get_recent_logs(self, count: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        return list(self.buffer)[-count:]


class PerformanceMonitor:
    """Performance monitoring for logging."""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        self.metrics_history: deque = deque(maxlen=1000)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = network.bytes_sent + network.bytes_recv
            
            # Garbage collection
            gc_stats = gc.get_stats()
            gc_collections = sum(stat['collections'] for stat in gc_stats)
            
            # Process and thread counts
            thread_count = threading.active_count()
            process_count = len(psutil.pids())
            
            metrics = PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_io_bytes=network_io,
                gc_collections=gc_collections,
                thread_count=thread_count,
                process_count=process_count
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            # Return default metrics on error
            return PerformanceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_io_bytes=0,
                gc_collections=0,
                thread_count=0,
                process_count=0
            )


class AdvancedLoggingSystem:
    """
    Advanced logging system with comprehensive features.
    
    Features:
    - Structured logging with JSON output
    - Performance monitoring
    - Request tracking
    - Log filtering and sanitization
    - Multiple output formats
    - Asynchronous processing
    - Log analysis and statistics
    """
    
    def __init__(
        self,
        name: str = "heygen_ai",
        log_dir: str = "logs",
        level: LogLevel = LogLevel.INFO,
        format_type: str = "json",
        enable_performance_monitoring: bool = True,
        enable_request_tracking: bool = True
    ):
        """
        Initialize the advanced logging system.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Log level
            format_type: Output format type
            enable_performance_monitoring: Enable performance monitoring
            enable_request_tracking: Enable request tracking
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.format_type = format_type
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_request_tracking = enable_request_tracking
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.filter = LogFilter()
        self.formatter = LogFormatter(format_type)
        self.handler = LogHandler(name)
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        # Setup handlers
        self._setup_handlers()
        
        # Start handler
        self.handler.start()
        
        # Log system initialization
        self.info("Advanced logging system initialized", category=LogCategory.SYSTEM)
    
    def _setup_handlers(self):
        """Setup log handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        self.handler.add_handler(console_handler)
        
        # File handler for all logs
        all_logs_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(all_logs_file)
        file_handler.setLevel(logging.DEBUG)
        self.handler.add_handler(file_handler)
        
        # File handler for errors
        error_logs_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.FileHandler(error_logs_file)
        error_handler.setLevel(logging.ERROR)
        self.handler.add_handler(error_handler)
        
        # File handler for security logs
        security_logs_file = self.log_dir / f"{self.name}_security.log"
        security_handler = logging.FileHandler(security_logs_file)
        security_handler.setLevel(logging.DEBUG)
        self.handler.add_handler(security_handler)
    
    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        extra_data: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        tags: Optional[List[str]] = None
    ) -> LogEntry:
        """Create a log entry."""
        # Get caller information
        frame = inspect.currentframe().f_back.f_back
        module = frame.f_globals.get('__name__', 'unknown')
        function = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # Get context information
        request_id = request_id_var.get() if self.enable_request_tracking else None
        user_id = user_id_var.get() if self.enable_request_tracking else None
        session_id = session_id_var.get() if self.enable_request_tracking else None
        
        # Get performance metrics
        performance_metrics = {}
        if self.performance_monitor:
            metrics = self.performance_monitor.get_metrics()
            performance_metrics = {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_used_mb": metrics.memory_used_mb,
                "memory_available_mb": metrics.memory_available_mb,
                "disk_usage_percent": metrics.disk_usage_percent,
                "network_io_bytes": metrics.network_io_bytes,
                "gc_collections": metrics.gc_collections,
                "thread_count": metrics.thread_count,
                "process_count": metrics.process_count
            }
        
        # Sanitize message
        sanitized_message = self.filter.sanitize_message(message)
        
        # Get exception info
        exception_info = None
        if exception:
            exception_info = traceback.format_exc()
        
        return LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            category=category,
            message=sanitized_message,
            module=module,
            function=function,
            line_number=line_number,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            extra_data=extra_data or {},
            exception_info=exception_info,
            performance_metrics=performance_metrics,
            tags=tags or []
        )
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if log level should be logged."""
        level_values = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
        return level_values[level] >= level_values[self.level]
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        extra_data: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        tags: Optional[List[str]] = None
    ):
        """Internal logging method."""
        if not self._should_log(level):
            return
        
        entry = self._create_log_entry(
            level=level,
            message=message,
            category=category,
            extra_data=extra_data,
            exception=exception,
            tags=tags
        )
        
        # Apply filters
        if not self.filter.should_log(entry):
            return
        
        # Send to handler
        self.handler.log(entry)
    
    def debug(
        self,
        message: str,
        category: LogCategory = LogCategory.DEBUG,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, category, extra_data, tags=tags)
    
    def info(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log info message."""
        self._log(LogLevel.INFO, message, category, extra_data, tags=tags)
    
    def warning(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, category, extra_data, tags=tags)
    
    def error(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        extra_data: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        tags: Optional[List[str]] = None
    ):
        """Log error message."""
        self._log(LogLevel.ERROR, message, category, extra_data, exception, tags)
    
    def critical(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        extra_data: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        tags: Optional[List[str]] = None
    ):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, category, extra_data, exception, tags)
    
    def security(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log security message."""
        self._log(level, message, LogCategory.SECURITY, extra_data, tags=tags)
    
    def performance(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log performance message."""
        self._log(level, message, LogCategory.PERFORMANCE, extra_data, tags=tags)
    
    def business(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log business message."""
        self._log(level, message, LogCategory.BUSINESS, extra_data, tags=tags)
    
    def audit(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log audit message."""
        self._log(level, message, LogCategory.AUDIT, extra_data, tags=tags)
    
    def api(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log API message."""
        self._log(level, message, LogCategory.API, extra_data, tags=tags)
    
    def database(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log database message."""
        self._log(level, message, LogCategory.DATABASE, extra_data, tags=tags)
    
    def cache(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log cache message."""
        self._log(level, message, LogCategory.CACHE, extra_data, tags=tags)
    
    def external(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log external service message."""
        self._log(level, message, LogCategory.EXTERNAL, extra_data, tags=tags)
    
    def set_request_context(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Set request context for tracking."""
        if request_id:
            request_id_var.set(request_id)
        if user_id:
            user_id_var.set(user_id)
        if session_id:
            session_id_var.set(session_id)
    
    def clear_request_context(self):
        """Clear request context."""
        request_id_var.set(None)
        user_id_var.set(None)
        session_id_var.set(None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return self.handler.get_stats()
    
    def get_recent_logs(self, count: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        return self.handler.get_recent_logs(count)
    
    def add_filter(self, filter_func: Callable[[LogEntry], bool]):
        """Add a log filter."""
        self.filter.add_filter(filter_func)
    
    def set_level(self, level: LogLevel):
        """Set log level."""
        self.level = level
    
    def cleanup(self):
        """Cleanup logging system."""
        self.handler.stop()
        self.info("Logging system cleanup completed", category=LogCategory.SYSTEM)


# Decorator for automatic logging
def log_function_call(
    category: LogCategory = LogCategory.SYSTEM,
    level: LogLevel = LogLevel.DEBUG,
    include_args: bool = True,
    include_result: bool = True
):
    """Decorator to automatically log function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = AdvancedLoggingSystem()
            
            # Log function entry
            extra_data = {}
            if include_args:
                extra_data["args"] = str(args)[:1000]  # Limit length
                extra_data["kwargs"] = str(kwargs)[:1000]
            
            logger.debug(
                f"Entering function {func.__name__}",
                category=category,
                level=level,
                extra_data=extra_data
            )
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Log function exit
                execution_time = time.time() - start_time
                extra_data = {"execution_time": execution_time}
                if include_result:
                    extra_data["result"] = str(result)[:1000]  # Limit length
                
                logger.debug(
                    f"Exiting function {func.__name__}",
                    category=category,
                    level=level,
                    extra_data=extra_data
                )
                
                return result
                
            except Exception as e:
                # Log function error
                execution_time = time.time() - start_time
                extra_data = {"execution_time": execution_time, "error": str(e)}
                
                logger.error(
                    f"Error in function {func.__name__}",
                    category=category,
                    extra_data=extra_data,
                    exception=e
                )
                
                raise
        
        return wrapper
    return decorator


# Example usage and demonstration
async def main():
    """Demonstrate the advanced logging system."""
    print("📝 HeyGen AI - Advanced Logging System Demo")
    print("=" * 70)
    
    # Initialize logging system
    logger = AdvancedLoggingSystem(
        name="heygen_ai_demo",
        log_dir="logs",
        level=LogLevel.DEBUG,
        format_type="json",
        enable_performance_monitoring=True,
        enable_request_tracking=True
    )
    
    try:
        # Set request context
        logger.set_request_context(
            request_id=str(uuid.uuid4()),
            user_id="user123",
            session_id="session456"
        )
        
        # Log different types of messages
        print("\n📋 Logging Different Message Types...")
        
        logger.info("System initialized successfully", category=LogCategory.SYSTEM)
        logger.debug("Debug information", category=LogCategory.DEBUG)
        logger.warning("This is a warning message", category=LogCategory.SYSTEM)
        logger.error("This is an error message", category=LogCategory.SYSTEM)
        logger.critical("This is a critical message", category=LogCategory.SYSTEM)
        
        # Log security events
        print("\n🔐 Logging Security Events...")
        logger.security("User login attempt", level=LogLevel.INFO, extra_data={"ip": "192.168.1.1"})
        logger.security("Failed authentication", level=LogLevel.WARNING, extra_data={"user": "admin"})
        logger.security("Suspicious activity detected", level=LogLevel.ERROR, extra_data={"activity": "brute_force"})
        
        # Log performance metrics
        print("\n⚡ Logging Performance Metrics...")
        logger.performance("High CPU usage detected", level=LogLevel.WARNING, extra_data={"cpu_percent": 85.5})
        logger.performance("Memory usage normal", level=LogLevel.INFO, extra_data={"memory_percent": 45.2})
        logger.performance("Database query slow", level=LogLevel.WARNING, extra_data={"query_time": 2.5})
        
        # Log business events
        print("\n💼 Logging Business Events...")
        logger.business("Order created", level=LogLevel.INFO, extra_data={"order_id": "12345", "amount": 99.99})
        logger.business("Payment processed", level=LogLevel.INFO, extra_data={"payment_id": "pay_67890"})
        logger.business("Refund issued", level=LogLevel.INFO, extra_data={"refund_id": "ref_11111"})
        
        # Log API events
        print("\n🌐 Logging API Events...")
        logger.api("API request received", level=LogLevel.INFO, extra_data={"method": "POST", "endpoint": "/api/v1/models"})
        logger.api("API response sent", level=LogLevel.INFO, extra_data={"status_code": 200, "response_time": 0.5})
        logger.api("API rate limit exceeded", level=LogLevel.WARNING, extra_data={"client_ip": "192.168.1.100"})
        
        # Log database events
        print("\n🗄️ Logging Database Events...")
        logger.database("Database connection established", level=LogLevel.INFO)
        logger.database("Query executed", level=LogLevel.DEBUG, extra_data={"query": "SELECT * FROM models", "execution_time": 0.1})
        logger.database("Database error", level=LogLevel.ERROR, extra_data={"error": "Connection timeout"})
        
        # Log cache events
        print("\n💾 Logging Cache Events...")
        logger.cache("Cache hit", level=LogLevel.DEBUG, extra_data={"key": "model_123", "hit_rate": 0.85})
        logger.cache("Cache miss", level=LogLevel.DEBUG, extra_data={"key": "model_456"})
        logger.cache("Cache eviction", level=LogLevel.INFO, extra_data={"evicted_keys": 10})
        
        # Log external service events
        print("\n🔗 Logging External Service Events...")
        logger.external("External API call", level=LogLevel.INFO, extra_data={"service": "openai", "endpoint": "/v1/chat"})
        logger.external("External service error", level=LogLevel.ERROR, extra_data={"service": "stripe", "error": "Payment failed"})
        logger.external("External service timeout", level=LogLevel.WARNING, extra_data={"service": "aws", "timeout": 30})
        
        # Log with tags
        print("\n🏷️ Logging with Tags...")
        logger.info("Model training started", category=LogCategory.BUSINESS, tags=["training", "model", "ai"])
        logger.info("Model training completed", category=LogCategory.BUSINESS, tags=["training", "model", "ai", "success"])
        logger.warning("Model training failed", category=LogCategory.BUSINESS, tags=["training", "model", "ai", "error"])
        
        # Log with exception
        print("\n❌ Logging with Exception...")
        try:
            raise ValueError("This is a test exception")
        except Exception as e:
            logger.error("An error occurred", category=LogCategory.SYSTEM, exception=e)
        
        # Get logging statistics
        print("\n📊 Logging Statistics:")
        stats = logger.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Get recent logs
        print("\n📋 Recent Logs:")
        recent_logs = logger.get_recent_logs(5)
        for log in recent_logs:
            print(f"  {log.timestamp} | {log.level.value} | {log.category.value} | {log.message}")
        
        # Clear request context
        logger.clear_request_context()
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error("Demo failed", exception=e)
    
    finally:
        # Cleanup
        logger.cleanup()
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())

