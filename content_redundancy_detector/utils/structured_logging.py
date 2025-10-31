"""
Structured Logging with Context Tracking
Enhanced logging for microservices and production environments
"""

import logging
import json
import sys
from typing import Any, Dict, Optional
from datetime import datetime
from contextvars import ContextVar

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
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
        request_id = request_id_var.get(None)
        user_id = user_id_var.get(None)
        correlation_id = correlation_id_var.get(None)
        
        if request_id:
            log_data["request_id"] = request_id
        if user_id:
            log_data["user_id"] = user_id
        if correlation_id:
            log_data["correlation_id"] = correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data, default=str)


def setup_structured_logging(level: str = "INFO", use_json: bool = False):
    """Setup structured logging"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger with context awareness"""
    logger = logging.getLogger(name)
    
    # Add context to log records
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id_var.get(None)
        record.user_id = user_id_var.get(None)
        record.correlation_id = correlation_id_var.get(None)
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    return logger


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    correlation_id: Optional[str] = None
):
    """Set request context variables"""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if correlation_id:
        correlation_id_var.set(correlation_id)


def clear_request_context():
    """Clear request context variables"""
    request_id_var.set(None)
    user_id_var.set(None)
    correlation_id_var.set(None)


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **kwargs
):
    """Log with context variables"""
    extra = {
        "request_id": request_id_var.get(None),
        "user_id": user_id_var.get(None),
        "correlation_id": correlation_id_var.get(None),
        **kwargs
    }
    
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message, extra=extra)


def log_performance(
    operation: str,
    duration: float,
    logger: Optional[logging.Logger] = None,
    **metadata
):
    """Log performance metrics"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    log_data = {
        "operation": operation,
        "duration_seconds": round(duration, 3),
        "request_id": request_id_var.get(None),
        "user_id": user_id_var.get(None),
        **metadata
    }
    
    # Classify performance
    if duration > 5.0:
        level = "error"
    elif duration > 2.0:
        level = "warning"
    elif duration > 1.0:
        level = "info"
    else:
        level = "debug"
    
    log_method = getattr(logger, level, logger.info)
    log_method(
        f"Performance: {operation} took {duration:.3f}s",
        extra=log_data
    )






