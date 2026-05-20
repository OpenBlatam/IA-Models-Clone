from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import logging
import logging.handlers
import sys
import os
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pathlib import Path
import json
from ..core.exceptions import ConfigurationError
from typing import Any, List, Dict, Optional
import asyncio
"""
Onyx AI Video System - Logger Utility

Logging utilities for the Onyx AI Video system with integration
with Onyx's logging patterns and structured logging support.
"""




class OnyxLogger:
    """
    Onyx-compatible logger for AI Video system.
    
    Provides structured logging with Onyx integration, request tracking,
    and performance monitoring.
    """
    
    def __init__(
        self,
        name: str = "ai_video",
        level: str = "INFO",
        use_onyx_logging: bool = True,
        log_file: Optional[str] = None,
        max_size: int = 10,
        backup_count: int = 5
    ):
        
    """__init__ function."""
self.name = name
        self.level = getattr(logging, level.upper())
        self.use_onyx_logging = use_onyx_logging
        self.log_file = log_file
        self.max_size = max_size * 1024 * 1024  # Convert to bytes
        self.backup_count = backup_count
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Request tracking
        self._request_id = None
        self._user_id = None
        self._session_id = None
        
        # Performance tracking
        self._start_times = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with appropriate handlers and formatters."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        if self.use_onyx_logging:
            formatter = self._create_onyx_formatter()
        else:
            formatter = self._create_standard_formatter()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            file_handler = self._create_file_handler()
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def _create_onyx_formatter(self) -> logging.Formatter:
        """Create Onyx-compatible formatter."""
        return logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(user_id)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _create_standard_formatter(self) -> logging.Formatter:
        """Create standard formatter."""
        return logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _create_file_handler(self) -> logging.handlers.RotatingFileHandler:
        """Create rotating file handler."""
        # Ensure log directory exists
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        return logging.handlers.RotatingFileHandler(
            filename=self.log_file,
            maxBytes=self.max_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
    
    def set_request_context(self, request_id: str, user_id: str, session_id: Optional[str] = None):
        """Set request context for logging."""
        self._request_id = request_id
        self._user_id = user_id
        self._session_id = session_id
    
    async def clear_request_context(self) -> Any:
        """Clear request context."""
        self._request_id = None
        self._user_id = None
        self._session_id = None
    
    def _get_extra(self, **kwargs) -> Dict[str, Any]:
        """Get extra logging parameters."""
        extra = {
            'request_id': self._request_id or 'N/A',
            'user_id': self._user_id or 'N/A',
            'session_id': self._session_id or 'N/A',
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        return extra
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        extra = self._get_extra(**kwargs)
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        extra = self._get_extra(**kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        extra = self._get_extra(**kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        extra = self._get_extra(**kwargs)
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        extra = self._get_extra(**kwargs)
        self.logger.critical(message, extra=extra)
    
    def exception(self, message: str, **kwargs):
        """Log exception message."""
        extra = self._get_extra(**kwargs)
        self.logger.exception(message, extra=extra)
    
    def log_request(self, request_data: Dict[str, Any], **kwargs):
        """Log request data."""
        extra = self._get_extra(**kwargs)
        self.logger.info(f"Request received: {json.dumps(request_data, default=str)}", extra=extra)
    
    def log_response(self, response_data: Dict[str, Any], **kwargs):
        """Log response data."""
        extra = self._get_extra(**kwargs)
        self.logger.info(f"Response sent: {json.dumps(response_data, default=str)}", extra=extra)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        extra = self._get_extra(operation=operation, duration=duration, **kwargs)
        self.logger.info(f"Performance: {operation} took {duration:.3f}s", extra=extra)
    
    def log_plugin_execution(self, plugin_name: str, status: str, duration: float, **kwargs):
        """Log plugin execution."""
        extra = self._get_extra(plugin_name=plugin_name, status=status, duration=duration, **kwargs)
        self.logger.info(f"Plugin {plugin_name}: {status} in {duration:.3f}s", extra=extra)
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self._start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self._start_times:
            return 0.0
        
        start_time = self._start_times[operation]
        duration = (datetime.now() - start_time).total_seconds()
        del self._start_times[operation]
        
        self.log_performance(operation, duration)
        return duration
    
    def log_structured(self, event: str, data: Dict[str, Any], level: str = "INFO", **kwargs):
        """Log structured data."""
        log_data = {
            "event": event,
            "data": data,
            **kwargs
        }
        
        extra = self._get_extra(**log_data)
        message = f"Structured log: {json.dumps(log_data, default=str)}"
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, extra=extra)


class OnyxLoggingAdapter:
    """
    Adapter for Onyx logging system.
    
    Provides compatibility layer between AI Video logging and Onyx logging.
    """
    
    def __init__(self, logger: OnyxLogger):
        
    """__init__ function."""
self.logger = logger
    
    def log_function_call(self, function_name: str, args: tuple, kwargs: dict, result: Any, duration: float):
        """Log function call with Onyx format."""
        self.logger.log_structured(
            event="function_call",
            data={
                "function": function_name,
                "args": str(args),
                "kwargs": kwargs,
                "result": str(result),
                "duration": duration
            },
            level="DEBUG"
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with Onyx context."""
        self.logger.error(
            f"Error occurred: {str(error)}",
            error_type=error.__class__.__name__,
            error_message=str(error),
            context=context
        )
    
    def log_onyx_integration(self, component: str, status: str, details: Dict[str, Any]):
        """Log Onyx integration events."""
        self.logger.log_structured(
            event="onyx_integration",
            data={
                "component": component,
                "status": status,
                "details": details
            },
            level="INFO"
        )


class PerformanceLogger:
    """
    Performance logging utility.
    
    Provides detailed performance logging and metrics collection.
    """
    
    def __init__(self, logger: OnyxLogger):
        
    """__init__ function."""
self.logger = logger
        self.metrics = {}
    
    def start_operation(self, operation: str, **kwargs):
        """Start timing an operation."""
        self.logger.start_timer(operation)
        self.metrics[operation] = {
            "start_time": datetime.now(),
            "kwargs": kwargs
        }
    
    def end_operation(self, operation: str, result: Any = None, error: Exception = None):
        """End timing an operation."""
        if operation not in self.metrics:
            return
        
        start_time = self.metrics[operation]["start_time"]
        duration = (datetime.now() - start_time).total_seconds()
        
        # Update metrics
        if operation not in self.metrics:
            self.metrics[operation] = {"count": 0, "total_time": 0, "errors": 0}
        
        self.metrics[operation]["count"] += 1
        self.metrics[operation]["total_time"] += duration
        
        if error:
            self.metrics[operation]["errors"] += 1
            self.logger.error(
                f"Operation {operation} failed after {duration:.3f}s",
                operation=operation,
                duration=duration,
                error=str(error)
            )
        else:
            self.logger.log_performance(operation, duration, result=str(result))
        
        # Clean up
        del self.metrics[operation]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()


class RequestLogger:
    """
    Request logging utility.
    
    Provides request-specific logging with correlation IDs.
    """
    
    def __init__(self, logger: OnyxLogger):
        
    """__init__ function."""
self.logger = logger
    
    def log_request_start(self, request_id: str, user_id: str, method: str, path: str, **kwargs):
        """Log request start."""
        self.logger.set_request_context(request_id, user_id)
        self.logger.info(
            f"Request started: {method} {path}",
            method=method,
            path=path,
            **kwargs
        )
    
    def log_request_end(self, request_id: str, status_code: int, duration: float, **kwargs):
        """Log request end."""
        self.logger.info(
            f"Request completed: {status_code} in {duration:.3f}s",
            status_code=status_code,
            duration=duration,
            **kwargs
        )
        self.logger.clear_request_context()
    
    def log_request_error(self, request_id: str, error: Exception, **kwargs):
        """Log request error."""
        self.logger.error(
            f"Request failed: {str(error)}",
            error_type=error.__class__.__name__,
            error_message=str(error),
            **kwargs
        )
        self.logger.clear_request_context()


# Global logger instances
_global_logger: Optional[OnyxLogger] = None
_performance_logger: Optional[PerformanceLogger] = None
_request_logger: Optional[RequestLogger] = None


def setup_logger(
    name: str = "ai_video",
    level: str = "INFO",
    use_onyx_logging: bool = True,
    log_file: Optional[str] = None,
    **kwargs
) -> OnyxLogger:
    """Setup global logger."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = OnyxLogger(
            name=name,
            level=level,
            use_onyx_logging=use_onyx_logging,
            log_file=log_file,
            **kwargs
        )
    
    return _global_logger


def get_logger() -> OnyxLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger(get_logger())
    return _performance_logger


async def get_request_logger() -> RequestLogger:
    """Get request logger instance."""
    global _request_logger
    if _request_logger is None:
        _request_logger = RequestLogger(get_logger())
    return _request_logger


def configure_logging_from_config(config: Dict[str, Any]) -> OnyxLogger:
    """Configure logging from configuration dictionary."""
    logging_config = config.get("logging", {})
    
    return setup_logger(
        name=logging_config.get("name", "ai_video"),
        level=logging_config.get("level", "INFO"),
        use_onyx_logging=logging_config.get("use_onyx_logging", True),
        log_file=logging_config.get("file_path"),
        max_size=logging_config.get("max_size", 10),
        backup_count=logging_config.get("backup_count", 5)
    )


# Logging decorators
def log_function_call(logger: Optional[OnyxLogger] = None):
    """Decorator to log function calls."""
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            log = logger or get_logger()
            log.start_timer(func.__name__)
            
            try:
                result = func(*args, **kwargs)
                log.end_timer(func.__name__)
                return result
            except Exception as e:
                log.error(f"Function {func.__name__} failed: {e}")
                raise
        
        return wrapper
    return decorator


def log_performance(operation_name: str, logger: Optional[OnyxLogger] = None):
    """Decorator to log performance metrics."""
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            log = logger or get_logger()
            log.start_timer(operation_name)
            
            try:
                result = func(*args, **kwargs)
                log.end_timer(operation_name)
                return result
            except Exception as e:
                log.error(f"Operation {operation_name} failed: {e}")
                raise
        
        return wrapper
    return decorator


# Utility functions
def log_system_startup(config: Dict[str, Any]):
    """Log system startup information."""
    logger = get_logger()
    logger.info("AI Video system starting up", config_summary=config)


def log_system_shutdown():
    """Log system shutdown."""
    logger = get_logger()
    logger.info("AI Video system shutting down")


def log_configuration_loaded(config_path: str, config: Dict[str, Any]):
    """Log configuration loading."""
    logger = get_logger()
    logger.info(f"Configuration loaded from {config_path}", config_keys=list(config.keys()))


def log_plugin_loaded(plugin_name: str, plugin_info: Dict[str, Any]):
    """Log plugin loading."""
    logger = get_logger()
    logger.info(f"Plugin loaded: {plugin_name}", plugin_info=plugin_info)


def log_plugin_error(plugin_name: str, error: Exception):
    """Log plugin error."""
    logger = get_logger()
    logger.error(f"Plugin error in {plugin_name}: {error}", plugin_name=plugin_name, error=str(error)) 