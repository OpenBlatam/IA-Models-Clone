"""
Structured Logging
==================

Advanced structured logging.
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """Structured logger with JSON output."""
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        json_output: bool = True,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_service: bool = True,
        service_name: Optional[str] = None
    ):
        self.name = name
        self.json_output = json_output
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_service = include_service
        self.service_name = service_name
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Add console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.value))
        
        if json_output:
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        self.logger.addHandler(handler)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ):
        """Log with structured data."""
        log_data = {
            "message": message,
            **kwargs
        }
        
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat()
        
        if self.include_level:
            log_data["level"] = level.value
        
        if self.include_service and self.service_name:
            log_data["service"] = self.service_name
        
        log_method = getattr(self.logger, level.value.lower())
        log_method(json.dumps(log_data) if self.json_output else message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)















