"""
Advanced Logging System
=======================

Comprehensive logging system for the Bulk TruthGPT system.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from structlog.stdlib import LoggerFactory

class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogContext:
    """Log context information."""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self, context: LogContext):
        super().__init__()
        self.context = context
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to log record."""
        record.component = self.context.component
        record.operation = self.context.operation
        
        if self.context.user_id:
            record.user_id = self.context.user_id
        if self.context.session_id:
            record.session_id = self.context.session_id
        if self.context.request_id:
            record.request_id = self.context.request_id
        if self.context.task_id:
            record.task_id = self.context.task_id
        if self.context.metadata:
            record.metadata = self.context.metadata
        
        return True

class StructuredLogger:
    """
    Structured logger with context support.
    
    Provides structured logging with context information,
    performance metrics, and error tracking.
    """
    
    def __init__(self, name: str, context: Optional[LogContext] = None):
        self.name = name
        self.context = context or LogContext(component=name, operation="unknown")
        self.logger = structlog.get_logger(name)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _log(self, level: str, message: str, **kwargs) -> None:
        """Internal logging method."""
        log_data = {
            'component': self.context.component,
            'operation': self.context.operation,
            'message': message,
            **kwargs
        }
        
        if self.context.user_id:
            log_data['user_id'] = self.context.user_id
        if self.context.session_id:
            log_data['session_id'] = self.context.session_id
        if self.context.request_id:
            log_data['request_id'] = self.context.request_id
        if self.context.task_id:
            log_data['task_id'] = self.context.task_id
        if self.context.metadata:
            log_data['metadata'] = self.context.metadata
        
        getattr(self.logger, level.lower())(message, **log_data)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log('critical', message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log('error', message, exc_info=True, **kwargs)
    
    def performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics."""
        self._log('info', f"Performance: {operation}", 
                 operation=operation, duration=duration, **kwargs)
    
    def metric(self, name: str, value: Union[int, float], **kwargs) -> None:
        """Log metric."""
        self._log('info', f"Metric: {name}", metric_name=name, metric_value=value, **kwargs)
    
    def audit(self, action: str, resource: str, **kwargs) -> None:
        """Log audit event."""
        self._log('info', f"Audit: {action}", 
                 audit_action=action, audit_resource=resource, **kwargs)
    
    def set_context(self, context: LogContext) -> None:
        """Set logging context."""
        self.context = context
    
    def update_context(self, **kwargs) -> None:
        """Update logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

class LoggingManager:
    """
    Centralized logging management.
    
    Manages loggers, handlers, and configuration for the entire system.
    """
    
    def __init__(self):
        self.loggers = {}
        self.handlers = {}
        self.formatters = {}
        self.filters = {}
        
    def setup_logging(self, 
                      log_level: str = "INFO",
                      log_format: str = "json",
                      log_file: Optional[str] = None,
                      log_max_size: str = "100MB",
                      log_backup_count: int = 5) -> None:
        """Setup logging configuration."""
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if log_format == "json":
            console_formatter = JSONFormatter()
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Parse max size
            max_size = self._parse_size(log_max_size)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=log_backup_count
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(console_formatter)
            root_logger.addHandler(file_handler)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self, name: str, context: Optional[LogContext] = None) -> StructuredLogger:
        """Get a structured logger."""
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(name, context)
        return self.loggers[name]
    
    def add_handler(self, name: str, handler: logging.Handler) -> None:
        """Add a handler to all loggers."""
        self.handlers[name] = handler
        for logger in self.loggers.values():
            logger.logger.addHandler(handler)
    
    def remove_handler(self, name: str) -> None:
        """Remove a handler from all loggers."""
        if name in self.handlers:
            handler = self.handlers[name]
            for logger in self.loggers.values():
                logger.logger.removeHandler(handler)
            del self.handlers[name]
    
    def set_level(self, level: str) -> None:
        """Set logging level for all loggers."""
        for logger in self.loggers.values():
            logger.logger.setLevel(getattr(logging, level.upper()))

# Global logging manager
logging_manager = LoggingManager()

def setup_logger(name: str, context: Optional[LogContext] = None) -> StructuredLogger:
    """Setup a logger with the given name and context."""
    return logging_manager.get_logger(name, context)

def get_logger(name: str) -> StructuredLogger:
    """Get an existing logger."""
    return logging_manager.get_logger(name)











