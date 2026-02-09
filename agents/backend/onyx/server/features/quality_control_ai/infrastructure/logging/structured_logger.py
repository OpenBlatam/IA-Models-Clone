"""
Structured Logger

Logger with structured output and context management.
"""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
from contextvars import ContextVar

# Context variable for request context
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


class StructuredLogger:
    """
    Structured logger with context support.
    
    Provides JSON-formatted logs with request context.
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Log level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.name = name
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current request context."""
        return request_context.get()
    
    def _format_message(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format log message as structured data.
        
        Args:
            level: Log level
            message: Log message
            extra: Additional fields
        
        Returns:
            Structured log dictionary
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **self._get_context(),
        }
        
        if extra:
            log_data.update(extra)
        
        return log_data
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        log_data = self._format_message("INFO", message, kwargs)
        self.logger.info(json.dumps(log_data))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        log_data = self._format_message("ERROR", message, kwargs)
        self.logger.error(json.dumps(log_data))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        log_data = self._format_message("WARNING", message, kwargs)
        self.logger.warning(json.dumps(log_data))
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        log_data = self._format_message("DEBUG", message, kwargs)
        self.logger.debug(json.dumps(log_data))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        log_data = self._format_message("CRITICAL", message, kwargs)
        self.logger.critical(json.dumps(log_data))


def get_logger(name: str) -> StructuredLogger:
    """
    Get structured logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


def setup_logging(
    level: str = "INFO",
    format: str = "json",
    file: Optional[str] = None
):
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format ('json' or 'text')
        file: Optional log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    if format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    root_logger.addHandler(handler)
    
    # Add file handler if specified
    if file:
        file_handler = logging.FileHandler(file)
        if format == "json":
            file_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        root_logger.addHandler(file_handler)


def set_request_context(**kwargs):
    """
    Set request context for logging.
    
    Args:
        **kwargs: Context variables
    """
    context = request_context.get({})
    context.update(kwargs)
    request_context.set(context)


def clear_request_context():
    """Clear request context."""
    request_context.set({})



