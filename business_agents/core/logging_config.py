"""
Logging Configuration
=====================

Comprehensive logging configuration for the Business Agents System.
"""

import logging
import logging.config
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import traceback

from ..config import config

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
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
            if key not in [
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "getMessage", "exc_info",
                "exc_text", "stack_info"
            ]:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class BusinessAgentsFilter(logging.Filter):
    """Custom filter for Business Agents logs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records."""
        # Add request ID if available
        if hasattr(record, 'request_id'):
            record.request_id = getattr(record, 'request_id', 'unknown')
        
        # Add service context
        record.service = "business_agents"
        
        return True

class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
    
    def log_execution_time(self, operation: str, duration: float, **kwargs):
        """Log operation execution time."""
        self.logger.info(
            f"Operation {operation} completed",
            extra={
                "operation": operation,
                "duration_seconds": duration,
                "performance_metric": True,
                **kwargs
            }
        )
    
    def log_agent_execution(self, agent_id: str, capability: str, duration: float, success: bool):
        """Log agent execution metrics."""
        self.logger.info(
            f"Agent {agent_id} executed {capability}",
            extra={
                "agent_id": agent_id,
                "capability": capability,
                "duration_seconds": duration,
                "success": success,
                "performance_metric": True
            }
        )
    
    def log_workflow_execution(self, workflow_id: str, duration: float, steps_count: int, success: bool):
        """Log workflow execution metrics."""
        self.logger.info(
            f"Workflow {workflow_id} executed",
            extra={
                "workflow_id": workflow_id,
                "duration_seconds": duration,
                "steps_count": steps_count,
                "success": success,
                "performance_metric": True
            }
        )

class SecurityLogger:
    """Logger for security events."""
    
    def __init__(self):
        self.logger = logging.getLogger("security")
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str = None):
        """Log authentication attempts."""
        self.logger.warning(
            f"Authentication attempt for user {user_id}",
            extra={
                "user_id": user_id,
                "success": success,
                "ip_address": ip_address,
                "security_event": True
            }
        )
    
    def log_authorization_failure(self, user_id: str, resource: str, action: str):
        """Log authorization failures."""
        self.logger.warning(
            f"Authorization failure for user {user_id}",
            extra={
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "security_event": True
            }
        )
    
    def log_suspicious_activity(self, activity: str, details: Dict[str, Any]):
        """Log suspicious activities."""
        self.logger.error(
            f"Suspicious activity detected: {activity}",
            extra={
                "activity": activity,
                "details": details,
                "security_event": True
            }
        )

def setup_logging() -> None:
    """Setup comprehensive logging configuration."""
    
    # Determine log level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Create logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": config.log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": JSONFormatter
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "filters": {
            "business_agents": {
                "()": BusinessAgentsFilter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout,
                "filters": ["business_agents"]
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "json",
                "filename": config.log_file or "business_agents.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "filters": ["business_agents"]
            },
            "performance": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": logging.INFO,
                "formatter": "json",
                "filename": "performance.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "filters": ["business_agents"]
            },
            "security": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": logging.WARNING,
                "formatter": "json",
                "filename": "security.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "filters": ["business_agents"]
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "business_agents": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "performance": {
                "level": logging.INFO,
                "handlers": ["performance"],
                "propagate": False
            },
            "security": {
                "level": logging.WARNING,
                "handlers": ["security"],
                "propagate": False
            },
            "uvicorn": {
                "level": logging.INFO,
                "handlers": ["console"],
                "propagate": False
            },
            "fastapi": {
                "level": logging.INFO,
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Create specialized loggers
    performance_logger = PerformanceLogger()
    security_logger = SecurityLogger()
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized", extra={
        "log_level": config.log_level,
        "log_file": config.log_file,
        "environment": config.environment.value
    })

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"business_agents.{name}")

def log_function_call(func_name: str, **kwargs):
    """Log function call with parameters."""
    logger = get_logger("function_calls")
    logger.debug(f"Function {func_name} called", extra={
        "function": func_name,
        "parameters": kwargs
    })

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log error with context."""
    logger = get_logger("errors")
    logger.error(f"Error occurred: {str(error)}", extra={
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }, exc_info=True)
