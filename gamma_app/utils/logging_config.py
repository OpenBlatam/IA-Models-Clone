"""
Gamma App - Advanced Logging Configuration
Comprehensive logging setup with structured logging and monitoring
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
import json
from datetime import datetime
import traceback

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class ContextFilter(logging.Filter):
    """Filter to add context information to log records"""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        # Add context information to the record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True

class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def filter(self, record):
        # Add performance context if available
        if hasattr(record, 'request_id'):
            # This would be set by middleware
            pass
        return True

class SecurityFilter(logging.Filter):
    """Filter to handle security-sensitive log records"""
    
    def filter(self, record):
        # Mask sensitive information
        if hasattr(record, 'message'):
            message = record.message
            # Mask passwords, tokens, etc.
            import re
            message = re.sub(r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+', 
                           'password="***"', message, flags=re.IGNORECASE)
            message = re.sub(r'token["\']?\s*[:=]\s*["\']?[^"\'\s]+', 
                           'token="***"', message, flags=re.IGNORECASE)
            record.message = message
        return True

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup advanced logging configuration"""
    
    # Create logs directory
    log_dir = Path(config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
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
    
    # Logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": JSONFormatter,
            },
            "performance": {
                "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s - %(request_id)s - %(response_time)sms",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "filters": {
            "context": {
                "()": ContextFilter,
                "context": {
                    "service": "gamma_app",
                    "version": "1.0.0"
                }
            },
            "performance": {
                "()": PerformanceFilter,
            },
            "security": {
                "()": SecurityFilter,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": config.get('console_level', 'INFO'),
                "formatter": "standard",
                "stream": sys.stdout,
                "filters": ["context", "security"]
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": config.get('file_level', 'INFO'),
                "formatter": "detailed",
                "filename": str(log_dir / "gamma_app.log"),
                "maxBytes": config.get('max_file_size', 10 * 1024 * 1024),  # 10MB
                "backupCount": config.get('backup_count', 5),
                "filters": ["context", "security"]
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(log_dir / "gamma_app_errors.log"),
                "maxBytes": config.get('max_file_size', 10 * 1024 * 1024),
                "backupCount": config.get('backup_count', 5),
                "filters": ["context", "security"]
            },
            "json_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": config.get('json_level', 'INFO'),
                "formatter": "json",
                "filename": str(log_dir / "gamma_app_structured.log"),
                "maxBytes": config.get('max_file_size', 10 * 1024 * 1024),
                "backupCount": config.get('backup_count', 5),
                "filters": ["context"]
            },
            "performance_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "performance",
                "filename": str(log_dir / "gamma_app_performance.log"),
                "maxBytes": config.get('max_file_size', 10 * 1024 * 1024),
                "backupCount": config.get('backup_count', 5),
                "filters": ["context", "performance"]
            },
            "security_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": "detailed",
                "filename": str(log_dir / "gamma_app_security.log"),
                "maxBytes": config.get('max_file_size', 10 * 1024 * 1024),
                "backupCount": config.get('backup_count', 10),  # Keep more security logs
                "filters": ["context", "security"]
            }
        },
        "loggers": {
            "gamma_app": {
                "level": config.get('app_level', 'INFO'),
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "gamma_app.api": {
                "level": config.get('api_level', 'INFO'),
                "handlers": ["console", "file", "performance_file"],
                "propagate": False
            },
            "gamma_app.security": {
                "level": "WARNING",
                "handlers": ["console", "security_file"],
                "propagate": False
            },
            "gamma_app.performance": {
                "level": "INFO",
                "handlers": ["performance_file"],
                "propagate": False
            },
            "gamma_app.ai": {
                "level": config.get('ai_level', 'INFO'),
                "handlers": ["console", "file"],
                "propagate": False
            },
            "gamma_app.cache": {
                "level": config.get('cache_level', 'INFO'),
                "handlers": ["console", "file"],
                "propagate": False
            },
            "gamma_app.database": {
                "level": config.get('db_level', 'WARNING'),
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "performance_file"],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy.pool": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": {
            "level": config.get('root_level', 'WARNING'),
            "handlers": ["console", "file"]
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Set up structured logging for specific modules
    setup_structured_logging()

def setup_structured_logging():
    """Setup structured logging for specific modules"""
    
    # API request/response logging
    api_logger = structlog.get_logger("gamma_app.api")
    
    # Security event logging
    security_logger = structlog.get_logger("gamma_app.security")
    
    # Performance logging
    performance_logger = structlog.get_logger("gamma_app.performance")
    
    # AI operation logging
    ai_logger = structlog.get_logger("gamma_app.ai")
    
    # Cache operation logging
    cache_logger = structlog.get_logger("gamma_app.cache")

class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self):
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = structlog.get_logger(self.__class__.__module__)
        return self._logger
    
    def log_info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(message, **kwargs)
    
    def log_exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)

def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger by name"""
    return structlog.get_logger(name)

def log_api_request(method: str, path: str, user_id: Optional[str] = None, 
                   request_id: Optional[str] = None, **kwargs):
    """Log API request"""
    logger = get_logger("gamma_app.api")
    logger.info(
        "API request",
        method=method,
        path=path,
        user_id=user_id,
        request_id=request_id,
        **kwargs
    )

def log_api_response(method: str, path: str, status_code: int, 
                    response_time: float, user_id: Optional[str] = None,
                    request_id: Optional[str] = None, **kwargs):
    """Log API response"""
    logger = get_logger("gamma_app.api")
    logger.info(
        "API response",
        method=method,
        path=path,
        status_code=status_code,
        response_time=response_time,
        user_id=user_id,
        request_id=request_id,
        **kwargs
    )

def log_security_event(event_type: str, severity: str, source_ip: str,
                      user_id: Optional[str] = None, **kwargs):
    """Log security event"""
    logger = get_logger("gamma_app.security")
    logger.warning(
        "Security event",
        event_type=event_type,
        severity=severity,
        source_ip=source_ip,
        user_id=user_id,
        **kwargs
    )

def log_performance_metric(metric_name: str, value: float, 
                          metric_type: str, **kwargs):
    """Log performance metric"""
    logger = get_logger("gamma_app.performance")
    logger.info(
        "Performance metric",
        metric_name=metric_name,
        value=value,
        metric_type=metric_type,
        **kwargs
    )

def log_ai_operation(operation: str, model: str, tokens_used: int,
                    generation_time: float, **kwargs):
    """Log AI operation"""
    logger = get_logger("gamma_app.ai")
    logger.info(
        "AI operation",
        operation=operation,
        model=model,
        tokens_used=tokens_used,
        generation_time=generation_time,
        **kwargs
    )

def log_cache_operation(operation: str, key: str, hit: bool,
                       response_time: float, **kwargs):
    """Log cache operation"""
    logger = get_logger("gamma_app.cache")
    logger.info(
        "Cache operation",
        operation=operation,
        key=key,
        hit=hit,
        response_time=response_time,
        **kwargs
    )

# Global logger instances
api_logger = get_logger("gamma_app.api")
security_logger = get_logger("gamma_app.security")
performance_logger = get_logger("gamma_app.performance")
ai_logger = get_logger("gamma_app.ai")
cache_logger = get_logger("gamma_app.cache")
