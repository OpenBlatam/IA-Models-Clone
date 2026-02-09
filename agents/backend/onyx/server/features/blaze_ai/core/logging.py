"""
Logging configuration for Enhanced Blaze AI.

This module provides centralized logging configuration with structured logging,
file rotation, and different log levels for different environments.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """
    Structured logging formatter that outputs JSON logs.
    
    This formatter creates structured logs that are easier to parse
    and analyze in log aggregation systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add request context if present
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        
        if hasattr(record, 'ip_address'):
            log_entry["ip_address"] = record.ip_address
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development environments.
    
    This formatter adds colors to console output for better readability
    during development and debugging.
    """
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format log message
        formatted = f"{color}[{timestamp}] {record.levelname:8s}{reset} "
        formatted += f"{color}{record.name}{reset} - {record.getMessage()}"
        
        # Add exception information if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class LoggingConfig:
    """Configuration class for logging setup."""
    
    def __init__(
        self,
        level: str = "INFO",
        format_type: str = "structured",
        log_file: Optional[str] = None,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = False,
        enable_structured: bool = True
    ):
        """
        Initialize logging configuration.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_type: Log format type (structured, colored, standard)
            log_file: Path to log file
            max_size: Maximum log file size in bytes
            backup_count: Number of log file backups
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_structured: Enable structured logging
        """
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.format_type = format_type
        self.log_file = log_file
        self.max_size = max_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_structured = enable_structured
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'LoggingConfig':
        """Create LoggingConfig from dictionary."""
        return cls(
            level=config.get('level', 'INFO'),
            format_type=config.get('format_type', 'structured'),
            log_file=config.get('file'),
            max_size=config.get('max_size', 10 * 1024 * 1024),
            backup_count=config.get('backup_count', 5),
            enable_console=config.get('enable_console', True),
            enable_file=config.get('enable_file', False),
            enable_structured=config.get('enable_structured', True)
        )


def setup_logging(
    config: Optional[LoggingConfig] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = False
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        config: Logging configuration object
        level: Logging level override
        log_file: Log file path override
        enable_console: Enable console logging
        enable_file: Enable file logging
    """
    if config is None:
        config = LoggingConfig()
    
    # Override config with parameters if provided
    if level:
        config.level = getattr(logging, level.upper(), logging.INFO)
    if log_file:
        config.log_file = log_file
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(config.level)
    
    # Create formatters
    if config.format_type == "structured" and config.enable_structured:
        formatter = StructuredFormatter()
        console_formatter = ColoredFormatter() if config.enable_console else formatter
    elif config.format_type == "colored":
        formatter = ColoredFormatter()
        console_formatter = formatter
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = formatter
    
    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.enable_file and config.log_file:
        # Ensure log directory exists
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(config.level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    _set_logger_levels()
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized", extra={
        'extra_fields': {
            'level': logging.getLevelName(config.level),
            'format_type': config.format_type,
            'console_enabled': config.enable_console,
            'file_enabled': config.enable_file,
            'structured_enabled': config.enable_structured
        }
    })


def _set_logger_levels() -> None:
    """Set specific logger levels for different modules."""
    # Set lower levels for verbose modules
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    # Set higher levels for noisy third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('redis').setLevel(logging.WARNING)
    
    # Set appropriate levels for AI model libraries
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('anthropic').setLevel(logging.INFO)
    logging.getLogger('stability_sdk').setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def get_request_logger(name: str, request_id: str, user_id: Optional[str] = None, ip_address: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with request context.
    
    Args:
        name: Logger name
        request_id: Unique request identifier
        user_id: User identifier (optional)
        ip_address: Client IP address (optional)
    
    Returns:
        Logger with request context
    """
    logger = logging.getLogger(name)
    
    # Add request context to logger
    logger.request_id = request_id
    if user_id:
        logger.user_id = user_id
    if ip_address:
        logger.ip_address = ip_address
    
    return logger


def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None, level: str = "DEBUG") -> None:
    """
    Log function call information.
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
        level: Log level for the message
    """
    logger = logging.getLogger(__name__)
    
    # Prepare log message
    message = f"Function call: {func_name}"
    extra_fields = {"function": func_name}
    
    if args:
        extra_fields["args"] = str(args)
    if kwargs:
        # Filter out sensitive information
        safe_kwargs = {k: v for k, v in kwargs.items() if not _is_sensitive_key(k)}
        extra_fields["kwargs"] = str(safe_kwargs)
    
    # Log with appropriate level
    log_func = getattr(logger, level.lower(), logger.debug)
    log_func(message, extra={'extra_fields': extra_fields})


def log_function_result(func_name: str, result: Any = None, execution_time: float = None, level: str = "DEBUG") -> None:
    """
    Log function execution result.
    
    Args:
        func_name: Name of the function
        result: Function result
        execution_time: Function execution time in seconds
        level: Log level for the message
    """
    logger = logging.getLogger(__name__)
    
    # Prepare log message
    message = f"Function result: {func_name}"
    extra_fields = {"function": func_name}
    
    if result is not None:
        extra_fields["result"] = str(result)
    if execution_time is not None:
        extra_fields["execution_time"] = f"{execution_time:.4f}s"
    
    # Log with appropriate level
    log_func = getattr(logger, level.lower(), logger.debug)
    log_func(message, extra={'extra_fields': extra_fields})


def log_error(error: Exception, context: Dict[str, Any] = None, level: str = "ERROR") -> None:
    """
    Log error information with context.
    
    Args:
        error: Exception that occurred
        context: Additional context information
        level: Log level for the message
    """
    logger = logging.getLogger(__name__)
    
    # Prepare log message
    message = f"Error occurred: {type(error).__name__}: {str(error)}"
    extra_fields = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_module": getattr(error, '__module__', 'unknown')
    }
    
    if context:
        extra_fields.update(context)
    
    # Log with appropriate level
    log_func = getattr(logger, level.lower(), logger.error)
    log_func(message, extra={'extra_fields': extra_fields}, exc_info=True)


def _is_sensitive_key(key: str) -> bool:
    """
    Check if a key contains sensitive information.
    
    Args:
        key: Key to check
    
    Returns:
        True if the key contains sensitive information
    """
    sensitive_patterns = [
        'password', 'secret', 'key', 'token', 'auth', 'credential',
        'api_key', 'private', 'sensitive', 'confidential'
    ]
    
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in sensitive_patterns)


def set_log_level(logger_name: str, level: str) -> None:
    """
    Set log level for a specific logger.
    
    Args:
        logger_name: Name of the logger
        level: Log level to set
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def enable_debug_logging() -> None:
    """Enable debug logging for development."""
    setup_logging(level="DEBUG", enable_console=True, enable_file=False)


def enable_production_logging(log_file: str = "logs/app.log") -> None:
    """Enable production logging with file output."""
    setup_logging(
        level="INFO",
        log_file=log_file,
        enable_console=False,
        enable_file=True
    )


def enable_structured_logging() -> None:
    """Enable structured logging for production environments."""
    config = LoggingConfig(
        level="INFO",
        format_type="structured",
        enable_structured=True,
        enable_console=True,
        enable_file=True
    )
    setup_logging(config)


# Convenience functions for common logging patterns
def log_startup(service_name: str, version: str, config_summary: Dict[str, Any] = None) -> None:
    """Log service startup information."""
    logger = logging.getLogger(__name__)
    message = f"🚀 {service_name} v{version} starting up"
    extra_fields = {"service": service_name, "version": version, "event": "startup"}
    
    if config_summary:
        extra_fields["config"] = config_summary
    
    logger.info(message, extra={'extra_fields': extra_fields})


def log_shutdown(service_name: str, version: str) -> None:
    """Log service shutdown information."""
    logger = logging.getLogger(__name__)
    message = f"🛑 {service_name} v{version} shutting down"
    extra_fields = {"service": service_name, "version": version, "event": "shutdown"}
    
    logger.info(message, extra={'extra_fields': extra_fields})


def log_health_check(service_name: str, status: str, details: Dict[str, Any] = None) -> None:
    """Log health check information."""
    logger = logging.getLogger(__name__)
    message = f"🏥 {service_name} health check: {status}"
    extra_fields = {"service": service_name, "status": status, "event": "health_check"}
    
    if details:
        extra_fields["details"] = details
    
    logger.info(message, extra={'extra_fields': extra_fields})


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "WARNING") -> None:
    """Log security-related events."""
    logger = logging.getLogger(__name__)
    message = f"🔒 Security event: {event_type}"
    extra_fields = {"event_type": event_type, "severity": severity, "event": "security"}
    
    if details:
        extra_fields["details"] = details
    
    # Log with appropriate level
    log_func = getattr(logger, severity.lower(), logger.warning)
    log_func(message, extra={'extra_fields': extra_fields})


def log_performance_metric(metric_name: str, value: Any, unit: str = None, tags: Dict[str, str] = None) -> None:
    """Log performance metrics."""
    logger = logging.getLogger(__name__)
    message = f"📊 Performance metric: {metric_name} = {value}"
    extra_fields = {"metric_name": metric_name, "value": value, "event": "performance_metric"}
    
    if unit:
        extra_fields["unit"] = unit
    if tags:
        extra_fields["tags"] = tags
    
    logger.info(message, extra={'extra_fields': extra_fields})
