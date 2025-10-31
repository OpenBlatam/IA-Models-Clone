"""
Logging Service for HeyGen AI
============================

Provides centralized logging functionality with structured logging,
log rotation, and multiple output formats.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Configuration for logging service."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    file_enabled: bool = True
    file_path: str = "logs/heygen_ai.log"
    file_max_size: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5
    
    # Console logging
    console_enabled: bool = True
    console_level: str = "INFO"
    
    # Structured logging
    structured: bool = True
    include_timestamp: bool = True
    include_level: bool = True
    include_logger: bool = True
    
    # Performance logging
    performance_logging: bool = True
    slow_query_threshold: float = 1.0  # seconds
    
    # Log rotation
    rotation_enabled: bool = True
    rotation_when: str = "midnight"
    rotation_interval: int = 1
    rotation_backup_count: int = 7


class StructuredFormatter(logging.Formatter):
    """Structured log formatter."""
    
    def __init__(self, config: LoggingConfig):
        super().__init__()
        self.config = config
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        if not self.config.structured:
            return super().format(record)
        
        log_data = {
            "message": record.getMessage(),
            "level": record.levelname,
            "logger": record.name,
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR:
            import traceback
            log_data["stack_trace"] = traceback.format_stack()
        
        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """Performance logging utility."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = logging.getLogger("performance")
        self.slow_query_threshold = config.slow_query_threshold
    
    def log_operation(self, operation: str, duration: float, **kwargs):
        """Log operation performance."""
        if duration > self.slow_query_threshold:
            self.logger.warning(
                f"Slow operation detected: {operation}",
                extra={
                    "operation": operation,
                    "duration": duration,
                    "threshold": self.slow_query_threshold,
                    **kwargs
                }
            )
        else:
            self.logger.info(
                f"Operation completed: {operation}",
                extra={
                    "operation": operation,
                    "duration": duration,
                    **kwargs
                }
            )
    
    def log_function_call(self, func_name: str, duration: float, **kwargs):
        """Log function call performance."""
        self.log_operation(f"function:{func_name}", duration, **kwargs)
    
    def log_api_call(self, endpoint: str, method: str, duration: float, status_code: int, **kwargs):
        """Log API call performance."""
        self.log_operation(
            f"api:{method}:{endpoint}",
            duration,
            status_code=status_code,
            **kwargs
        )


class LoggingService:
    """Centralized logging service."""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig()
        self.loggers: Dict[str, logging.Logger] = {}
        self.performance_logger = PerformanceLogger(self.config)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        if self.config.file_enabled:
            log_path = Path(self.config.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.console_level.upper()))
            
            if self.config.structured:
                console_formatter = StructuredFormatter(self.config)
            else:
                console_formatter = logging.Formatter(
                    self.config.format,
                    datefmt=self.config.date_format
                )
            
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if self.config.file_enabled:
            if self.config.rotation_enabled:
                file_handler = logging.handlers.TimedRotatingFileHandler(
                    self.config.file_path,
                    when=self.config.rotation_when,
                    interval=self.config.rotation_interval,
                    backupCount=self.config.rotation_backup_count
                )
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    self.config.file_path,
                    maxBytes=self.config.file_max_size,
                    backupCount=self.config.file_backup_count
                )
            
            file_handler.setLevel(getattr(logging, self.config.level.upper()))
            
            if self.config.structured:
                file_formatter = StructuredFormatter(self.config)
            else:
                file_formatter = logging.Formatter(
                    self.config.format,
                    datefmt=self.config.date_format
                )
            
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Set up performance logging
        if self.config.performance_logging:
            perf_logger = logging.getLogger("performance")
            perf_logger.setLevel(logging.INFO)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        
        return self.loggers[name]
    
    def set_level(self, logger_name: str, level: str):
        """Set log level for a specific logger."""
        logger = self.get_logger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
    
    def add_handler(self, logger_name: str, handler: logging.Handler):
        """Add a handler to a specific logger."""
        logger = self.get_logger(logger_name)
        logger.addHandler(handler)
    
    def remove_handler(self, logger_name: str, handler: logging.Handler):
        """Remove a handler from a specific logger."""
        logger = self.get_logger(logger_name)
        logger.removeHandler(handler)
    
    def log_event(self, event: str, level: str = "info", logger_name: str = "events", **kwargs):
        """Log an event with structured data."""
        logger = self.get_logger(logger_name)
        
        log_data = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        if level == "debug":
            logger.debug(f"Event: {event}", extra={"extra_fields": log_data})
        elif level == "info":
            logger.info(f"Event: {event}", extra={"extra_fields": log_data})
        elif level == "warning":
            logger.warning(f"Event: {event}", extra={"extra_fields": log_data})
        elif level == "error":
            logger.error(f"Event: {event}", extra={"extra_fields": log_data})
        elif level == "critical":
            logger.critical(f"Event: {event}", extra={"extra_fields": log_data})
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.performance_logger.log_operation(operation, duration, **kwargs)
    
    def log_function_performance(self, func_name: str, duration: float, **kwargs):
        """Log function performance."""
        self.performance_logger.log_function_call(func_name, duration, **kwargs)
    
    def log_api_performance(self, endpoint: str, method: str, duration: float, status_code: int, **kwargs):
        """Log API performance."""
        self.performance_logger.log_api_call(endpoint, method, duration, status_code, **kwargs)
    
    def get_logger_info(self) -> Dict[str, Any]:
        """Get information about all loggers."""
        return {
            "config": asdict(self.config),
            "loggers": {
                name: {
                    "level": logger.level,
                    "handlers": len(logger.handlers),
                    "propagate": logger.propagate
                }
                for name, logger in self.loggers.items()
            }
        }
    
    def reload_config(self, new_config: LoggingConfig):
        """Reload logging configuration."""
        self.config = new_config
        self._setup_logging()
        logger.info("Logging configuration reloaded")


# Global logging service instance
_logging_service: Optional[LoggingService] = None


def get_logging_service(config: Optional[LoggingConfig] = None) -> LoggingService:
    """Get the global logging service instance."""
    global _logging_service
    
    if _logging_service is None:
        _logging_service = LoggingService(config)
    
    return _logging_service


def setup_logging(config: Optional[LoggingConfig] = None) -> LoggingService:
    """Setup the global logging service."""
    global _logging_service
    
    if _logging_service is not None:
        _logging_service.reload_config(config or LoggingConfig())
    else:
        _logging_service = LoggingService(config)
    
    return _logging_service


def get_logger(name: str) -> logging.Logger:
    """Get a logger from the global logging service."""
    service = get_logging_service()
    return service.get_logger(name)


def log_event(event: str, level: str = "info", logger_name: str = "events", **kwargs):
    """Log an event using the global logging service."""
    service = get_logging_service()
    service.log_event(event, level, logger_name, **kwargs)


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance using the global logging service."""
    service = get_logging_service()
    service.log_performance(operation, duration, **kwargs)
