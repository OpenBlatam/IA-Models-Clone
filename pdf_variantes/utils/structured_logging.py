"""
PDF Variantes - Structured Logging
Structured logging with context and correlation IDs
"""

import logging
import json
import sys
from typing import Any, Optional, Dict
from datetime import datetime
from contextvars import ContextVar
from uuid import uuid4

# Context variables for request tracking
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
_service_name: ContextVar[Optional[str]] = ContextVar('service_name', default='pdf-variantes-api')


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add context variables
        request_id = _request_id.get()
        if request_id:
            log_data["request_id"] = request_id
        
        user_id = _user_id.get()
        if user_id:
            log_data["user_id"] = user_id
        
        service_name = _service_name.get()
        if service_name:
            log_data["service"] = service_name
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs', 'message',
                          'pathname', 'process', 'processName', 'relativeCreated', 'thread',
                          'threadName', 'exc_info', 'exc_text', 'stack_info', 'extra_fields']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class StructuredLogger(logging.Logger):
    """Enhanced logger with structured logging capabilities"""
    
    def _log_with_context(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info=None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log with context variables"""
        extra = extra or {}
        
        # Add context to extra
        request_id = _request_id.get()
        if request_id:
            extra['request_id'] = request_id
        
        user_id = _user_id.get()
        if user_id:
            extra['user_id'] = user_id
        
        service_name = _service_name.get()
        if service_name:
            extra['service'] = service_name
        
        # Merge kwargs into extra
        if kwargs:
            extra.update(kwargs)
        
        if extra:
            extra['extra_fields'] = extra.copy()
        
        super()._log(level, msg, args, exc_info=exc_info, extra=extra)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, args, **kwargs)
    
    def error(self, msg: str, *args, exc_info=None, **kwargs):
        self._log_with_context(logging.ERROR, msg, args, exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, *args, exc_info=None, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, args, exc_info=exc_info, **kwargs)


def setup_structured_logging(
    level: str = "INFO",
    output_format: str = "json",
    service_name: str = "pdf-variantes-api"
) -> None:
    """Setup structured logging"""
    # Set service name
    _service_name.set(service_name)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter
    if output_format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Replace logger class
    logging.setLoggerClass(StructuredLogger)


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance"""
    logger = logging.getLogger(name)
    # Ensure it's a StructuredLogger
    if not isinstance(logger, StructuredLogger):
        logger.__class__ = StructuredLogger
    return logger


def set_request_id(request_id: str) -> None:
    """Set request ID in context"""
    _request_id.set(request_id)


def set_user_id(user_id: str) -> None:
    """Set user ID in context"""
    _user_id.set(user_id)


def get_request_id() -> Optional[str]:
    """Get current request ID from context"""
    return _request_id.get()


def get_user_id() -> Optional[str]:
    """Get current user ID from context"""
    return _user_id.get()


def clear_context() -> None:
    """Clear context variables"""
    _request_id.set(None)
    _user_id.set(None)


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Set multiple context variables at once"""
    if request_id:
        _request_id.set(request_id)
    if user_id:
        _user_id.set(user_id)
    # Store correlation ID separately if needed
    if correlation_id:
        # You could add a correlation_id context var if needed
        pass


def clear_request_context() -> None:
    """Clear all request context variables"""
    clear_context()


def log_performance(
    operation: str,
    duration: float,
    logger: logging.Logger,
    status_code: Optional[int] = None,
    **kwargs
) -> None:
    """Log performance metrics"""
    extra = {
        "operation": operation,
        "duration": duration,
        "duration_ms": duration * 1000,
    }
    
    if status_code:
        extra["status_code"] = status_code
    
    extra.update(kwargs)
    
    if duration > 1.0:
        logger.warning(f"Slow operation: {operation} took {duration:.3f}s", extra=extra)
    else:
        logger.info(f"Operation: {operation} took {duration:.3f}s", extra=extra)
