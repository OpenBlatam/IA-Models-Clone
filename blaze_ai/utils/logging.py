"""
Logging utilities for the Blaze AI module.

This module provides advanced logging capabilities including structured logging,
log rotation, multiple output formats, and performance monitoring.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
from contextlib import contextmanager
import functools

# =============================================================================
# Logging Configuration
# =============================================================================

@dataclass
class LogConfig:
    """Logging configuration."""
    
    # Basic settings
    level: str = "INFO"
    format: str = "json"  # json, text, structured
    output: str = "both"  # console, file, both
    
    # File settings
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    log_directory: str = "./logs"
    
    # Performance settings
    enable_performance_logging: bool = True
    enable_slow_query_logging: bool = True
    slow_query_threshold: float = 1.0  # seconds
    
    # Structured logging
    enable_structured_logging: bool = True
    include_timestamp: bool = True
    include_level: bool = True
    include_module: bool = True
    include_function: bool = True
    include_line_number: bool = True
    include_process_id: bool = True
    include_thread_id: bool = True
    
    # Advanced features
    enable_log_compression: bool = True
    enable_log_encryption: bool = False
    enable_log_rotation: bool = True
    enable_log_aggregation: bool = False
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            errors.append(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        
        valid_formats = ["json", "text", "structured"]
        if self.format not in valid_formats:
            errors.append(f"Invalid log format: {self.format}. Must be one of {valid_formats}")
        
        valid_outputs = ["console", "file", "both"]
        if self.output not in valid_outputs:
            errors.append(f"Invalid output: {self.output}. Must be one of {valid_outputs}")
        
        if self.max_file_size <= 0:
            errors.append("max_file_size must be positive")
        
        if self.backup_count < 0:
            errors.append("backup_count must be non-negative")
        
        if self.slow_query_threshold <= 0:
            errors.append("slow_query_threshold must be positive")
        
        return errors

# =============================================================================
# Structured Log Record
# =============================================================================

@dataclass
class StructuredLogRecord:
    """Structured log record with metadata."""
    
    # Core fields
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line_number: int
    
    # Context fields
    process_id: int
    thread_id: int
    thread_name: str
    
    # Additional fields
    extra: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    performance_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        
        # Filter out None values
        result = {k: v for k, v in result.items() if v is not None}
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

# =============================================================================
# Performance Logger
# =============================================================================

class PerformanceLogger:
    """Performance logging and monitoring."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.logger = logging.getLogger("performance")
        self.slow_queries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    @contextmanager
    def log_operation(self, operation_name: str, **context):
        """Context manager for logging operation performance."""
        if not self.config.enable_performance_logging:
            yield
            return
        
        start_time = time.time()
        start_cpu = time.process_time()
        
        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_cpu = time.process_time()
            
            duration = end_time - start_time
            cpu_time = end_cpu - start_cpu
            
            self._log_performance(operation_name, duration, cpu_time, success, error, context)
    
    def _log_performance(self, operation_name: str, duration: float, cpu_time: float, 
                        success: bool, error: Optional[str], context: Dict[str, Any]):
        """Log performance information."""
        performance_data = {
            "operation": operation_name,
            "duration": duration,
            "cpu_time": cpu_time,
            "success": success,
            "timestamp": time.time(),
            **context
        }
        
        if error:
            performance_data["error"] = error
        
        # Log to performance logger
        if success:
            self.logger.info(f"Operation completed: {operation_name}", extra={
                "performance_data": performance_data
            })
        else:
            self.logger.error(f"Operation failed: {operation_name}", extra={
                "performance_data": performance_data
            })
        
        # Track slow queries
        if duration > self.config.slow_query_threshold:
            with self._lock:
                self.slow_queries.append(performance_data)
                
                # Keep only recent slow queries
                if len(self.slow_queries) > 100:
                    self.slow_queries = self.slow_queries[-100:]
        
        # Log slow queries separately
        if duration > self.config.slow_query_threshold and self.config.enable_slow_query_logging:
            self.logger.warning(f"Slow operation detected: {operation_name} took {duration:.2f}s", extra={
                "performance_data": performance_data
            })
    
    def get_slow_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent slow queries."""
        with self._lock:
            return sorted(
                self.slow_queries[-limit:],
                key=lambda x: x["duration"],
                reverse=True
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if not self.slow_queries:
                return {"total_operations": 0, "slow_queries": 0}
            
            durations = [q["duration"] for q in self.slow_queries]
            return {
                "total_operations": len(self.slow_queries),
                "slow_queries": len(durations),
                "average_duration": sum(durations) / len(durations),
                "max_duration": max(durations),
                "min_duration": min(durations)
            }

# =============================================================================
# Structured Formatter
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """Structured log formatter."""
    
    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        # Extract extra fields
        extra = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                extra[key] = value
        
        # Create structured record
        structured_record = StructuredLogRecord(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            process_id=record.process,
            thread_id=record.thread,
            thread_name=record.threadName,
            extra=extra
        )
        
        # Add exception information
        if record.exc_info:
            structured_record.exception = ''.join(traceback.format_exception(*record.exc_info))
        
        # Return formatted output
        if self.config.format == "json":
            return structured_record.to_json()
        else:
            return self._format_text(structured_record)
    
    def _format_text(self, record: StructuredLogRecord) -> str:
        """Format as human-readable text."""
        parts = [
            f"[{record.timestamp}]",
            f"[{record.level}]",
            f"[{record.module}:{record.function}:{record.line_number}]",
            f"[PID:{record.process_id} TID:{record.thread_id}]",
            record.message
        ]
        
        if record.exception:
            parts.append(f"\nException: {record.exception}")
        
        if record.extra:
            parts.append(f"\nExtra: {json.dumps(record.extra, default=str)}")
        
        return " ".join(parts)

# =============================================================================
# Log Manager
# =============================================================================

class LogManager:
    """Centralized log management."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.loggers: Dict[str, logging.Logger] = {}
        self.performance_logger = PerformanceLogger(config)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid log configuration: {errors}")
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create log directory if needed
        if self.config.output in ["file", "both"]:
            log_dir = Path(self.config.log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        if self.config.output in ["console", "both"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(StructuredFormatter(self.config))
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if self.config.output in ["file", "both"]:
            if self.config.log_file:
                log_file_path = Path(self.config.log_directory) / self.config.log_file
            else:
                log_file_path = Path(self.config.log_directory) / "blaze_ai.log"
            
            if self.config.enable_log_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file_path)
            
            file_handler.setFormatter(StructuredFormatter(self.config))
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name."""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            
            # Set level
            logger.setLevel(getattr(logging, self.config.level.upper()))
            
            # Store reference
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def get_performance_logger(self) -> PerformanceLogger:
        """Get the performance logger."""
        return self.performance_logger
    
    def update_config(self, new_config: LogConfig):
        """Update logging configuration."""
        errors = new_config.validate()
        if errors:
            raise ValueError(f"Invalid log configuration: {errors}")
        
        self.config = new_config
        self._setup_logging()
    
    def get_config(self) -> LogConfig:
        """Get current logging configuration."""
        return self.config
    
    def get_logger_stats(self) -> Dict[str, Any]:
        """Get statistics about loggers."""
        stats = {
            "total_loggers": len(self.loggers),
            "loggers": list(self.loggers.keys()),
            "performance_stats": self.performance_logger.get_performance_stats()
        }
        
        # Add handler information
        root_logger = logging.getLogger()
        stats["handlers"] = [
            {
                "type": type(handler).__name__,
                "level": handler.level,
                "formatter": type(handler.formatter).__name__ if handler.formatter else None
            }
            for handler in root_logger.handlers
        ]
        
        return stats

# =============================================================================
# Global Instance and Convenience Functions
# =============================================================================

_default_log_manager: Optional[LogManager] = None

def get_log_manager(config: Optional[LogConfig] = None) -> LogManager:
    """Get the global log manager instance."""
    global _default_log_manager
    if _default_log_manager is None:
        if config is None:
            config = LogConfig()
        _default_log_manager = LogManager(config)
    return _default_log_manager

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return get_log_manager().get_logger(name)

def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger."""
    return get_log_manager().get_performance_logger()

def setup_logging(config: LogConfig):
    """Setup global logging configuration."""
    global _default_log_manager
    _default_log_manager = LogManager(config)

def log_performance(operation_name: str, **context):
    """Decorator for logging function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            performance_logger = get_performance_logger()
            with performance_logger.log_operation(operation_name, **context):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# =============================================================================
# Default Configuration
# =============================================================================

def create_default_log_config() -> LogConfig:
    """Create default logging configuration."""
    return LogConfig()

def create_production_log_config() -> LogConfig:
    """Create production logging configuration."""
    config = LogConfig()
    config.level = "WARNING"
    config.output = "both"
    config.log_file = "blaze_ai_production.log"
    config.enable_performance_logging = True
    config.enable_slow_query_logging = True
    config.slow_query_threshold = 0.5
    config.enable_log_rotation = True
    config.max_file_size = 50 * 1024 * 1024  # 50MB
    config.backup_count = 10
    return config

def create_development_log_config() -> LogConfig:
    """Create development logging configuration."""
    config = LogConfig()
    config.level = "DEBUG"
    config.output = "both"
    config.log_file = "blaze_ai_development.log"
    config.enable_performance_logging = True
    config.enable_slow_query_logging = True
    config.slow_query_threshold = 0.1
    config.enable_log_rotation = True
    config.max_file_size = 10 * 1024 * 1024  # 10MB
    config.backup_count = 5
    return config


