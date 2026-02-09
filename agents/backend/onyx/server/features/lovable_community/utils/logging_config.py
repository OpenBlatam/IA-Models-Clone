"""
Advanced logging configuration with structlog

Production-ready structured logging with better performance and observability.
Supports JSON output for log aggregation systems.
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import structlog
    from structlog.stdlib import LoggerFactory
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None
    LoggerFactory = None


def setup_logging(
    level: str = "INFO",
    use_structlog: bool = True,
    json_output: bool = False,
    include_timestamp: bool = True,
    include_module: bool = True,
    include_line: bool = False
) -> None:
    """
    Configure logging system with structlog support.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_structlog: Use structlog for structured logging
        json_output: Output logs as JSON (for log aggregation)
        include_timestamp: Include timestamp in logs
        include_module: Include module name
        include_line: Include line number
        
    Raises:
        ValueError: If level is not a valid logging level
    """
    if not level or not isinstance(level, str):
        raise ValueError(f"level must be a non-empty string, got {type(level).__name__}")
    
    level_upper = level.upper().strip()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    if level_upper not in valid_levels:
        raise ValueError(
            f"Invalid logging level '{level}'. Must be one of: {', '.join(valid_levels)}"
        )
    
    log_level = getattr(logging, level_upper, logging.INFO)
    
    if use_structlog and STRUCTLOG_AVAILABLE:
        setup_structlog(
            level=log_level,
            json_output=json_output
        )
    else:
        setup_standard_logging(
            level=log_level,
            include_timestamp=include_timestamp,
            include_module=include_module,
            include_line=include_line
        )


def setup_structlog(
    level: int = logging.INFO,
    json_output: bool = False
) -> None:
    """
    Setup structlog for structured logging.
    
    Args:
        level: Logging level
        json_output: Output as JSON
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )


def setup_standard_logging(
    level: int = logging.INFO,
    include_timestamp: bool = True,
    include_module: bool = True,
    include_line: bool = False
) -> None:
    """
    Setup standard logging (fallback when structlog is not available).
    
    Args:
        level: Logging level
        include_timestamp: Include timestamp
        include_module: Include module name
        include_line: Include line number
    """
    parts = []
    if include_timestamp:
        parts.append("%(asctime)s")
    parts.append("%(levelname)s")
    if include_module:
        parts.append("%(name)s")
    if include_line:
        parts.append("%(filename)s:%(lineno)d")
    parts.append("%(message)s")
    format_string = " - ".join(parts)
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (optional, must be valid level)
        
    Returns:
        Configured logger
        
    Raises:
        ValueError: If name is None or empty, or level is invalid
    """
    if not name or not isinstance(name, str) or not name.strip():
        raise ValueError(f"Logger name must be a non-empty string, got {type(name).__name__}")
    
    logger = logging.getLogger(name.strip())
    
    if level:
        if not isinstance(level, str) or not level.strip():
            raise ValueError(f"Logging level must be a non-empty string, got {type(level).__name__}")
        
        level_upper = level.upper().strip()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        if level_upper not in valid_levels:
            raise ValueError(
                f"Invalid logging level '{level}'. Must be one of: {', '.join(valid_levels)}"
            )
        
        logger.setLevel(getattr(logging, level_upper, logging.INFO))
    
    return logger


def get_structlogger(name: str) -> Any:
    """
    Get a structlog logger (if available).
    
    Args:
        name: Logger name
        
    Returns:
        Structlog logger or standard logger
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return get_logger(name)


class StructuredLogger:
    """
    Structured logger wrapper for better observability.
    
    Provides consistent logging interface with context support.
    """
    
    def __init__(self, name: str):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            
        Raises:
            ValueError: If name is None or empty
        """
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"Logger name must be a non-empty string, got {type(name).__name__}")
        
        self.name = name.strip()
        if STRUCTLOG_AVAILABLE:
            self.logger = structlog.get_logger(self.name)
        else:
            self.logger = logging.getLogger(self.name)
    
    def _log(
        self,
        level: str,
        message: str,
        **kwargs: Any
    ) -> None:
        """
        Log a message with structured data.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            **kwargs: Additional structured data
            
        Raises:
            ValueError: If level is invalid or message is None/empty
        """
        if not message or not isinstance(message, str) or not message.strip():
            raise ValueError(f"Log message must be a non-empty string, got {type(message).__name__}")
        
        if not level or not isinstance(level, str):
            raise ValueError(f"Log level must be a non-empty string, got {type(level).__name__}")
        
        level_upper = level.upper().strip()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        if level_upper not in valid_levels:
            raise ValueError(
                f"Invalid log level '{level}'. Must be one of: {', '.join(valid_levels)}"
            )
        
        if STRUCTLOG_AVAILABLE:
            log_method = getattr(self.logger, level_upper.lower())
            log_method(message.strip(), **kwargs)
        else:
            log_level = getattr(logging, level_upper, logging.INFO)
            if kwargs:
                message = f"{message.strip()} | {', '.join(f'{k}={v}' for k, v in kwargs.items())}"
            self.logger.log(log_level, message)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        if STRUCTLOG_AVAILABLE:
            self.logger.exception(message, **kwargs)
        else:
            self.logger.exception(message, extra=kwargs)
    
    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """
        Bind context variables to logger.
        
        Args:
            **kwargs: Context variables
            
        Returns:
            New logger instance with bound context
        """
        if STRUCTLOG_AVAILABLE:
            return StructuredLogger(self.name).logger.bind(**kwargs)
        return self


class PerformanceLogger:
    """
    Performance logger for metrics and timing.
    
    Optimized for production use with structured logging.
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize performance logger.
        
        Args:
            logger: Structured logger instance
        """
        self.logger = logger or StructuredLogger("performance")
    
    def log_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an operation with duration.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
            metadata: Additional metadata
        """
        log_data = {
            "operation": operation,
            "duration_seconds": duration,
            "success": success,
        }
        
        if metadata:
            log_data.update(metadata)
        
        if success:
            self.logger.info(
                f"{operation} completed",
                **log_data
            )
        else:
            self.logger.warning(
                f"{operation} failed",
                **log_data
            )
    
    def log_query(
        self,
        query_type: str,
        duration: float,
        row_count: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Log a database query.
        
        Args:
            query_type: Query type (SELECT, INSERT, UPDATE, DELETE)
            duration: Duration in seconds
            row_count: Number of rows affected
            **kwargs: Additional query metadata
        """
        log_data = {
            "query_type": query_type,
            "duration_seconds": duration,
            **kwargs
        }
        
        if row_count is not None:
            log_data["row_count"] = row_count
        
        self.logger.debug(
            f"Database query: {query_type}",
            **log_data
        )
    
    def log_cache(
        self,
        operation: str,
        key: str,
        hit: bool = True,
        duration: Optional[float] = None
    ) -> None:
        """
        Log a cache operation.
        
        Args:
            operation: Operation (get, set, delete)
            key: Cache key (truncated for privacy)
            hit: Whether it was a hit (for get operations)
            duration: Operation duration
        """
        log_data = {
            "cache_operation": operation,
            "cache_key": key[:50] + "..." if len(key) > 50 else key,
            "cache_hit": hit,
        }
        
        if duration is not None:
            log_data["duration_seconds"] = duration
        
        self.logger.debug(
            f"Cache {operation}",
            **log_data
        )
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        **kwargs: Any
    ) -> None:
        """
        Log an HTTP request.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration: Request duration in seconds
            **kwargs: Additional request metadata
        """
        log_data = {
            "http_method": method,
            "http_path": path,
            "http_status": status_code,
            "duration_seconds": duration,
            **kwargs
        }
        
        level = "info" if status_code < 400 else "warning" if status_code < 500 else "error"
        getattr(self.logger, level)(
            f"{method} {path}",
            **log_data
        )


# Global performance logger
performance_logger = PerformanceLogger()
