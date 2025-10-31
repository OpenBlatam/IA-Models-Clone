from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
import threading
import traceback
import functools
    import structlog
from typing import Any, List, Dict, Optional
import asyncio
"""
AI Video System - Logging Configuration

Production-ready logging configuration with structured logging,
log rotation, multiple handlers, and advanced logging features.
"""


try:
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LogConfig:
    """Logging configuration settings."""
    # General settings
    level: str = "INFO"
    format: str = "json"  # json, text, structured
    include_timestamp: bool = True
    include_level: bool = True
    include_module: bool = True
    include_function: bool = True
    include_line: bool = True
    
    # File logging
    log_file: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    log_directory: str = "logs"
    
    # Console logging
    console_enabled: bool = True
    console_level: str = "INFO"
    
    # Structured logging
    structured_logging: bool = True
    include_process_id: bool = True
    include_thread_id: bool = True
    include_hostname: bool = True
    
    # Performance logging
    log_performance: bool = True
    slow_query_threshold: float = 1.0  # seconds
    
    # Error logging
    log_exceptions: bool = True
    include_stack_trace: bool = True
    error_notification: bool = False
    
    # Security logging
    log_security_events: bool = True
    mask_sensitive_data: bool = True
    sensitive_fields: List[str] = field(default_factory=lambda: [
        'password', 'token', 'api_key', 'secret', 'key'
    ])


class StructuredFormatter(logging.Formatter):
    """
    Structured log formatter with JSON output.
    
    Features:
    - JSON structured logging
    - Custom field inclusion
    - Sensitive data masking
    - Performance metrics
    """
    
    def __init__(self, config: LogConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.sensitive_fields = set(config.sensitive_fields)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add process and thread information
        if self.config.include_process_id:
            log_entry['process_id'] = record.process
        
        if self.config.include_thread_id:
            log_entry['thread_id'] = record.thread
        
        # Add hostname
        if self.config.include_hostname:
            log_entry['hostname'] = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info) if self.config.include_stack_trace else None
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Mask sensitive data
        if self.config.mask_sensitive_data:
            log_entry = self._mask_sensitive_data(log_entry)
        
        # Add performance metrics
        if self.config.log_performance and hasattr(record, 'duration'):
            log_entry['performance'] = {
                'duration_ms': record.duration * 1000,
                'slow_query': record.duration > self.config.slow_query_threshold
            }
        
        return json.dumps(log_entry, default=str)
    
    def _mask_sensitive_data(self, data: Any) -> Any:
        """Recursively mask sensitive data in log entries."""
        if isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                if key.lower() in self.sensitive_fields:
                    masked_data[key] = '***MASKED***'
                else:
                    masked_data[key] = self._mask_sensitive_data(value)
            return masked_data
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        else:
            return data


class PerformanceLogger:
    """
    Performance logging utilities.
    
    Features:
    - Function timing
    - Query timing
    - Performance metrics
    - Slow operation detection
    """
    
    def __init__(self, logger: logging.Logger, threshold: float = 1.0):
        
    """__init__ function."""
self.logger = logger
        self.threshold = threshold
    
    def time_function(self, func_name: Optional[str] = None):
        """Decorator to time function execution."""
        def decorator(func) -> Any:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                start_time = datetime.now()
                try:
                    result = await func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    if duration > self.threshold:
                        self.logger.warning(
                            f"Slow function execution: {func_name or func.__name__}",
                            extra={
                                'extra_fields': {
                                    'function_name': func_name or func.__name__,
                                    'duration': duration,
                                    'slow_query': True
                                }
                            }
                        )
                    else:
                        self.logger.debug(
                            f"Function execution: {func_name or func.__name__}",
                            extra={
                                'extra_fields': {
                                    'function_name': func_name or func.__name__,
                                    'duration': duration,
                                    'slow_query': False
                                }
                            }
                        )
                    
                    return result
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.error(
                        f"Function execution failed: {func_name or func.__name__}",
                        extra={
                            'extra_fields': {
                                'function_name': func_name or func.__name__,
                                'duration': duration,
                                'error': str(e)
                            }
                        },
                        exc_info=True
                    )
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start_time = datetime.now()
                try:
                    result = func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    if duration > self.threshold:
                        self.logger.warning(
                            f"Slow function execution: {func_name or func.__name__}",
                            extra={
                                'extra_fields': {
                                    'function_name': func_name or func.__name__,
                                    'duration': duration,
                                    'slow_query': True
                                }
                            }
                        )
                    else:
                        self.logger.debug(
                            f"Function execution: {func_name or func.__name__}",
                            extra={
                                'extra_fields': {
                                    'function_name': func_name or func.__name__,
                                    'duration': duration,
                                    'slow_query': False
                                }
                            }
                        )
                    
                    return result
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.error(
                        f"Function execution failed: {func_name or func.__name__}",
                        extra={
                            'extra_fields': {
                                'function_name': func_name or func.__name__,
                                'duration': duration,
                                'error': str(e)
                            }
                        },
                        exc_info=True
                    )
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def log_query_performance(self, query: str, duration: float, success: bool = True):
        """Log database query performance."""
        if duration > self.threshold:
            self.logger.warning(
                "Slow database query detected",
                extra={
                    'extra_fields': {
                        'query_type': 'database',
                        'query': query[:200] + '...' if len(query) > 200 else query,
                        'duration': duration,
                        'success': success,
                        'slow_query': True
                    }
                }
            )
        else:
            self.logger.debug(
                "Database query executed",
                extra={
                    'extra_fields': {
                        'query_type': 'database',
                        'query': query[:200] + '...' if len(query) > 200 else query,
                        'duration': duration,
                        'success': success,
                        'slow_query': False
                    }
                }
            )


class SecurityLogger:
    """
    Security event logging.
    
    Features:
    - Security event logging
    - Authentication logging
    - Authorization logging
    - Threat detection logging
    """
    
    def __init__(self, logger: logging.Logger):
        
    """__init__ function."""
self.logger = logger
    
    def log_login_attempt(self, user_id: str, success: bool, ip_address: str, user_agent: str):
        """Log login attempt."""
        self.logger.info(
            f"Login attempt: {'success' if success else 'failed'}",
            extra={
                'extra_fields': {
                    'event_type': 'login_attempt',
                    'user_id': user_id,
                    'success': success,
                    'ip_address': ip_address,
                    'user_agent': user_agent
                }
            }
        )
    
    def log_authorization_failure(self, user_id: str, resource: str, action: str, reason: str):
        """Log authorization failure."""
        self.logger.warning(
            "Authorization failure",
            extra={
                'extra_fields': {
                    'event_type': 'authorization_failure',
                    'user_id': user_id,
                    'resource': resource,
                    'action': action,
                    'reason': reason
                }
            }
        )
    
    def log_security_threat(self, threat_type: str, details: Dict[str, Any], severity: str = "medium"):
        """Log security threat."""
        log_method = getattr(self.logger, severity.lower(), self.logger.warning)
        log_method(
            f"Security threat detected: {threat_type}",
            extra={
                'extra_fields': {
                    'event_type': 'security_threat',
                    'threat_type': threat_type,
                    'severity': severity,
                    'details': details
                }
            }
        )
    
    def log_data_access(self, user_id: str, data_type: str, action: str, resource_id: str):
        """Log data access."""
        self.logger.info(
            f"Data access: {action}",
            extra={
                'extra_fields': {
                    'event_type': 'data_access',
                    'user_id': user_id,
                    'data_type': data_type,
                    'action': action,
                    'resource_id': resource_id
                }
            }
        )


class LogManager:
    """
    Centralized log management.
    
    Features:
    - Logger configuration
    - Handler management
    - Log rotation
    - Multiple output formats
    """
    
    def __init__(self, config: LogConfig):
        
    """__init__ function."""
self.config = config
        self.loggers: Dict[str, logging.Logger] = {}
        self.performance_loggers: Dict[str, PerformanceLogger] = {}
        self.security_loggers: Dict[str, SecurityLogger] = {}
        
        # Create log directory
        if config.log_directory:
            Path(config.log_directory).mkdir(parents=True, exist_ok=True)
    
    def setup_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """Setup a logger with the specified configuration."""
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level or self.config.level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.console_level))
            
            if self.config.format == "json":
                console_handler.setFormatter(StructuredFormatter(self.config))
            else:
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_file:
            log_file_path = Path(self.config.log_directory) / self.config.log_file
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.level))
            
            if self.config.format == "json":
                file_handler.setFormatter(StructuredFormatter(self.config))
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            logger.addHandler(file_handler)
        
        # Error file handler
        error_log_file = f"error_{self.config.log_file}" if self.config.log_file else "error.log"
        error_log_path = Path(self.config.log_directory) / error_log_file
        
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        
        if self.config.format == "json":
            error_handler.setFormatter(StructuredFormatter(self.config))
        else:
            error_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        logger.addHandler(error_handler)
        
        # Security file handler
        if self.config.log_security_events:
            security_log_file = f"security_{self.config.log_file}" if self.config.log_file else "security.log"
            security_log_path = Path(self.config.log_directory) / security_log_file
            
            security_handler = logging.handlers.RotatingFileHandler(
                security_log_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            security_handler.setLevel(logging.INFO)
            
            if self.config.format == "json":
                security_handler.setFormatter(StructuredFormatter(self.config))
            else:
                security_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            logger.addHandler(security_handler)
        
        self.loggers[name] = logger
        return logger
    
    def get_performance_logger(self, name: str) -> PerformanceLogger:
        """Get a performance logger for the specified name."""
        if name not in self.performance_loggers:
            logger = self.setup_logger(f"{name}.performance")
            self.performance_loggers[name] = PerformanceLogger(
                logger, self.config.slow_query_threshold
            )
        
        return self.performance_loggers[name]
    
    def get_security_logger(self, name: str) -> SecurityLogger:
        """Get a security logger for the specified name."""
        if name not in self.security_loggers:
            logger = self.setup_logger(f"{name}.security")
            self.security_loggers[name] = SecurityLogger(logger)
        
        return self.security_loggers[name]
    
    def setup_structlog(self) -> None:
        """Setup structured logging with structlog."""
        if not STRUCTLOG_AVAILABLE:
            logger.warning("structlog not available, using standard logging")
            return
        
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
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def rotate_logs(self) -> None:
        """Manually rotate log files."""
        for logger_instance in self.loggers.values():
            for handler in logger_instance.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """Clean up old log files."""
        if not self.config.log_directory:
            return 0
        
        log_dir = Path(self.config.log_directory)
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        removed_count = 0
        
        for log_file in log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove old log file {log_file}: {e}")
        
        return removed_count


# Global log manager instance
log_config = LogConfig()
log_manager = LogManager(log_config)

# Setup main logger
main_logger = log_manager.setup_logger("ai_video")
performance_logger = log_manager.get_performance_logger("ai_video")
security_logger = log_manager.get_security_logger("ai_video")

# Setup structlog if available
if STRUCTLOG_AVAILABLE:
    log_manager.setup_structlog()


# Logging decorators
def log_function_call(logger_name: str = "ai_video"):
    """Decorator to log function calls."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = log_manager.setup_logger(f"{logger_name}.function")
            logger.info(
                f"Function called: {func.__name__}",
                extra={
                    'extra_fields': {
                        'function_name': func.__name__,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                logger.info(
                    f"Function completed: {func.__name__}",
                    extra={
                        'extra_fields': {
                            'function_name': func.__name__,
                            'status': 'success'
                        }
                    }
                )
                return result
            except Exception as e:
                logger.error(
                    f"Function failed: {func.__name__}",
                    extra={
                        'extra_fields': {
                            'function_name': func.__name__,
                            'status': 'error',
                            'error': str(e)
                        }
                    },
                    exc_info=True
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = log_manager.setup_logger(f"{logger_name}.function")
            logger.info(
                f"Function called: {func.__name__}",
                extra={
                    'extra_fields': {
                        'function_name': func.__name__,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                }
            )
            
            try:
                result = func(*args, **kwargs)
                logger.info(
                    f"Function completed: {func.__name__}",
                    extra={
                        'extra_fields': {
                            'function_name': func.__name__,
                            'status': 'success'
                        }
                    }
                )
                return result
            except Exception as e:
                logger.error(
                    f"Function failed: {func.__name__}",
                    extra={
                        'extra_fields': {
                            'function_name': func.__name__,
                            'status': 'error',
                            'error': str(e)
                        }
                    },
                    exc_info=True
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_security_event(event_type: str, severity: str = "info"):
    """Decorator to log security events."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                result = await func(*args, **kwargs)
                security_logger.log_security_threat(
                    event_type, {'status': 'success'}, severity
                )
                return result
            except Exception as e:
                security_logger.log_security_threat(
                    event_type, {'status': 'error', 'error': str(e)}, severity
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                security_logger.log_security_threat(
                    event_type, {'status': 'success'}, severity
                )
                return result
            except Exception as e:
                security_logger.log_security_threat(
                    event_type, {'status': 'error', 'error': str(e)}, severity
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Utility functions
def setup_logging(config: Optional[LogConfig] = None) -> None:
    """Setup logging with the specified configuration."""
    global log_config, log_manager
    
    if config:
        log_config = config
    
    log_manager = LogManager(log_config)
    
    # Setup main loggers
    global main_logger, performance_logger, security_logger
    main_logger = log_manager.setup_logger("ai_video")
    performance_logger = log_manager.get_performance_logger("ai_video")
    security_logger = log_manager.get_security_logger("ai_video")
    
    # Setup structlog if available
    if STRUCTLOG_AVAILABLE:
        log_manager.setup_structlog()
    
    main_logger.info("Logging system initialized", extra={
        'extra_fields': {
            'config': {
                'level': log_config.level,
                'format': log_config.format,
                'log_file': log_config.log_file,
                'console_enabled': log_config.console_enabled
            }
        }
    })


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return log_manager.setup_logger(name)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger with the specified name."""
    return log_manager.get_performance_logger(name)


def get_security_logger(name: str) -> SecurityLogger:
    """Get a security logger with the specified name."""
    return log_manager.get_security_logger(name)


async def cleanup_logging_resources() -> None:
    """Cleanup logging resources."""
    # Rotate logs
    log_manager.rotate_logs()
    
    # Cleanup old logs
    removed_count = log_manager.cleanup_old_logs()
    
    main_logger.info(f"Logging cleanup completed: {removed_count} old log files removed") 