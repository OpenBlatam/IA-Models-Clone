"""
Logging Configuration for Improved Video-OpusClip API

Comprehensive logging system with:
- Structured logging with JSON format
- Request/response logging
- Performance logging
- Error tracking
- Security event logging
- Log rotation and management
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import logging
import logging.config
import structlog
import sys
from pathlib import Path
from datetime import datetime
import json

from ..config import settings

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class LoggingConfig:
    """Configuration for structured logging system."""
    
    def __init__(self):
        self.log_level = settings.log_level.value.upper()
        self.log_format = settings.log_format
        self.log_file = settings.log_file
        self.enable_structured_logging = settings.enable_structured_logging
        self.log_request_id = settings.log_request_id
        self.log_response_time = settings.log_response_time
        
        # Log file paths
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.app_log_file = self.log_dir / "app.log"
        self.error_log_file = self.log_dir / "error.log"
        self.access_log_file = self.log_dir / "access.log"
        self.performance_log_file = self.log_dir / "performance.log"
        self.security_log_file = self.log_dir / "security.log"
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get comprehensive logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.log_format,
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "structured": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processor": structlog.dev.ConsoleRenderer(colors=True),
                },
                "json": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processor": structlog.processors.JSONRenderer(),
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.log_level,
                    "formatter": "structured" if self.enable_structured_logging else "default",
                    "stream": "ext://sys.stdout",
                },
                "app_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json" if self.enable_structured_logging else "detailed",
                    "filename": str(self.app_log_file),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "json" if self.enable_structured_logging else "detailed",
                    "filename": str(self.error_log_file),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                },
                "access_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json" if self.enable_structured_logging else "detailed",
                    "filename": str(self.access_log_file),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                },
                "performance_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json" if self.enable_structured_logging else "detailed",
                    "filename": str(self.performance_log_file),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                },
                "security_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "WARNING",
                    "formatter": "json" if self.enable_structured_logging else "detailed",
                    "filename": str(self.security_log_file),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                }
            },
            "loggers": {
                "": {  # Root logger
                    "level": self.log_level,
                    "handlers": ["console", "app_file"],
                    "propagate": False,
                },
                "app": {
                    "level": "INFO",
                    "handlers": ["console", "app_file"],
                    "propagate": False,
                },
                "error": {
                    "level": "ERROR",
                    "handlers": ["console", "error_file"],
                    "propagate": False,
                },
                "access": {
                    "level": "INFO",
                    "handlers": ["access_file"],
                    "propagate": False,
                },
                "performance": {
                    "level": "INFO",
                    "handlers": ["performance_file"],
                    "propagate": False,
                },
                "security": {
                    "level": "WARNING",
                    "handlers": ["console", "security_file"],
                    "propagate": False,
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["access_file"],
                    "propagate": False,
                }
            }
        }

# =============================================================================
# STRUCTURED LOGGING SETUP
# =============================================================================

def setup_structured_logging():
    """Setup structured logging with structlog."""
    
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
            structlog.processors.JSONRenderer() if settings.enable_structured_logging else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# =============================================================================
# CUSTOM LOGGING PROCESSORS
# =============================================================================

class RequestIDProcessor:
    """Processor to add request ID to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add request ID to event dictionary."""
        # This would be populated by middleware
        if hasattr(event_dict, 'request_id'):
            event_dict['request_id'] = event_dict.get('request_id')
        return event_dict

class PerformanceProcessor:
    """Processor to add performance metrics to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add performance metrics to event dictionary."""
        if 'response_time' in event_dict:
            event_dict['performance'] = {
                'response_time': event_dict['response_time'],
                'timestamp': datetime.utcnow().isoformat()
            }
        return event_dict

class SecurityProcessor:
    """Processor to add security context to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add security context to event dictionary."""
        if 'security_event' in event_dict:
            event_dict['security'] = {
                'event_type': event_dict.get('security_event'),
                'severity': event_dict.get('severity', 'medium'),
                'timestamp': datetime.utcnow().isoformat()
            }
        return event_dict

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class LoggerManager:
    """Manager for different types of loggers."""
    
    def __init__(self):
        self._loggers = {}
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup different types of loggers."""
        self._loggers = {
            'app': structlog.get_logger('app'),
            'error': structlog.get_logger('error'),
            'access': structlog.get_logger('access'),
            'performance': structlog.get_logger('performance'),
            'security': structlog.get_logger('security')
        }
    
    def get_logger(self, logger_type: str = 'app') -> structlog.BoundLogger:
        """Get logger by type."""
        return self._loggers.get(logger_type, self._loggers['app'])
    
    def log_request(self, method: str, url: str, status_code: int, 
                   response_time: float, request_id: Optional[str] = None):
        """Log HTTP request."""
        self._loggers['access'].info(
            "HTTP request",
            method=method,
            url=url,
            status_code=status_code,
            response_time=response_time,
            request_id=request_id
        )
    
    def log_error(self, error: Exception, context: Optional[Dict] = None, 
                 request_id: Optional[str] = None):
        """Log error with context."""
        self._loggers['error'].error(
            "Error occurred",
            error=str(error),
            error_type=type(error).__name__,
            context=context or {},
            request_id=request_id
        )
    
    def log_performance(self, operation: str, duration: float, 
                       metrics: Optional[Dict] = None, request_id: Optional[str] = None):
        """Log performance metrics."""
        self._loggers['performance'].info(
            "Performance metric",
            operation=operation,
            duration=duration,
            metrics=metrics or {},
            request_id=request_id
        )
    
    def log_security_event(self, event_type: str, severity: str, 
                          details: Optional[Dict] = None, request_id: Optional[str] = None):
        """Log security event."""
        self._loggers['security'].warning(
            "Security event",
            security_event=event_type,
            severity=severity,
            details=details or {},
            request_id=request_id
        )

# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================

class LoggingMiddleware:
    """Middleware for request/response logging."""
    
    def __init__(self, logger_manager: LoggerManager):
        self.logger_manager = logger_manager
    
    async def log_request(self, request, response, response_time: float):
        """Log HTTP request and response."""
        request_id = getattr(request.state, 'request_id', None)
        
        self.logger_manager.log_request(
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            response_time=response_time,
            request_id=request_id
        )
    
    async def log_error(self, request, error: Exception):
        """Log request error."""
        request_id = getattr(request.state, 'request_id', None)
        
        context = {
            'method': request.method,
            'url': str(request.url),
            'client_ip': request.client.host if request.client else None,
            'user_agent': request.headers.get('User-Agent')
        }
        
        self.logger_manager.log_error(
            error=error,
            context=context,
            request_id=request_id
        )

# =============================================================================
# LOG ANALYSIS UTILITIES
# =============================================================================

class LogAnalyzer:
    """Utility for analyzing log files."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
    
    def analyze_error_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze error logs for the specified time period."""
        error_log_file = self.log_dir / "error.log"
        
        if not error_log_file.exists():
            return {"error": "Error log file not found"}
        
        errors = []
        error_counts = {}
        
        try:
            with open(error_log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        errors.append(log_entry)
                        
                        error_type = log_entry.get('error_type', 'Unknown')
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    except json.JSONDecodeError:
                        continue
            
            return {
                'total_errors': len(errors),
                'error_types': error_counts,
                'recent_errors': errors[-10:] if errors else []
            }
        except Exception as e:
            return {"error": f"Failed to analyze error logs: {str(e)}"}
    
    def analyze_performance_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance logs for the specified time period."""
        performance_log_file = self.log_dir / "performance.log"
        
        if not performance_log_file.exists():
            return {"error": "Performance log file not found"}
        
        performance_metrics = []
        
        try:
            with open(performance_log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        performance_metrics.append(log_entry)
                    except json.JSONDecodeError:
                        continue
            
            if not performance_metrics:
                return {"error": "No performance metrics found"}
            
            # Calculate statistics
            durations = [m.get('duration', 0) for m in performance_metrics]
            operations = [m.get('operation', 'unknown') for m in performance_metrics]
            
            operation_counts = {}
            for op in operations:
                operation_counts[op] = operation_counts.get(op, 0) + 1
            
            return {
                'total_operations': len(performance_metrics),
                'average_duration': sum(durations) / len(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'operation_counts': operation_counts,
                'recent_metrics': performance_metrics[-10:] if performance_metrics else []
            }
        except Exception as e:
            return {"error": f"Failed to analyze performance logs: {str(e)}"}
    
    def analyze_security_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze security logs for the specified time period."""
        security_log_file = self.log_dir / "security.log"
        
        if not security_log_file.exists():
            return {"error": "Security log file not found"}
        
        security_events = []
        event_counts = {}
        severity_counts = {}
        
        try:
            with open(security_log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        security_events.append(log_entry)
                        
                        event_type = log_entry.get('security_event', 'Unknown')
                        severity = log_entry.get('severity', 'medium')
                        
                        event_counts[event_type] = event_counts.get(event_type, 0) + 1
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    except json.JSONDecodeError:
                        continue
            
            return {
                'total_events': len(security_events),
                'event_types': event_counts,
                'severity_counts': severity_counts,
                'recent_events': security_events[-10:] if security_events else []
            }
        except Exception as e:
            return {"error": f"Failed to analyze security logs: {str(e)}"}

# =============================================================================
# LOGGING INITIALIZATION
# =============================================================================

def initialize_logging():
    """Initialize the complete logging system."""
    
    # Setup structured logging
    setup_structured_logging()
    
    # Get logging configuration
    config = LoggingConfig()
    logging_config = config.get_logging_config()
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Create logger manager
    logger_manager = LoggerManager()
    
    # Create log analyzer
    log_analyzer = LogAnalyzer(config.log_dir)
    
    return logger_manager, log_analyzer

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'LoggingConfig',
    'setup_structured_logging',
    'RequestIDProcessor',
    'PerformanceProcessor',
    'SecurityProcessor',
    'LoggerManager',
    'LoggingMiddleware',
    'LogAnalyzer',
    'initialize_logging'
]






























