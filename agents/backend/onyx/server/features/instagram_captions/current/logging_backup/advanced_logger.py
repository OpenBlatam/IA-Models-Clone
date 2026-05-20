"""
Advanced Logger for Instagram Captions API v10.0

Structured logging with multiple handlers and advanced formatting.
"""

import logging
import json
import time
import traceback
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class LogContext:
    """Logging context information."""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    context: Dict[str, Any]
    extra_data: Dict[str, Any]
    exception_info: Optional[Dict[str, Any]] = None

class AdvancedLogger:
    """Advanced logging system with structured logging capabilities."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.context = LogContext()
        
        # Configure logger
        self._configure_logger()
        
        # Initialize formatters
        self.formatters = {
            'json': self._create_json_formatter(),
            'text': self._create_text_formatter(),
            'structured': self._create_structured_formatter()
        }
        
        # Set default formatter
        self.current_formatter = 'structured'
    
    def _configure_logger(self):
        """Configure the underlying logger."""
        # Set log level
        log_level = self.config.get('log_level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add console handler
        if self.config.get('console_logging', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            self.logger.addHandler(console_handler)
        
        # Add file handler
        log_file = self.config.get('log_file')
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            self.logger.addHandler(file_handler)
        
        # Add rotating file handler
        rotating_config = self.config.get('rotating_file')
        if rotating_config:
            from .log_rotator import LogRotator
            rotating_handler = LogRotator.create_handler(
                filename=rotating_config['filename'],
                max_bytes=rotating_config.get('max_bytes', 10 * 1024 * 1024),  # 10MB
                backup_count=rotating_config.get('backup_count', 5)
            )
            rotating_handler.setLevel(getattr(logging, log_level.upper()))
            self.logger.addHandler(rotating_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _create_json_formatter(self):
        """Create JSON formatter."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields
                if hasattr(record, 'context'):
                    log_entry['context'] = record.context
                if hasattr(record, 'extra_data'):
                    log_entry['extra_data'] = record.extra_data
                if hasattr(record, 'exception_info'):
                    log_entry['exception_info'] = record.exception_info
                
                return json.dumps(log_entry, ensure_ascii=False)
        
        return JSONFormatter()
    
    def _create_text_formatter(self):
        """Create human-readable text formatter."""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _create_structured_formatter(self):
        """Create structured text formatter."""
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                # Base format
                formatted = f"[{record.levelname:8}] {record.name}: {record.getMessage()}"
                
                # Add context if available
                if hasattr(record, 'context') and record.context:
                    context_str = ' | '.join([f"{k}={v}" for k, v in record.context.items() if v])
                    if context_str:
                        formatted += f" | Context: {context_str}"
                
                # Add extra data if available
                if hasattr(record, 'extra_data') and record.extra_data:
                    extra_str = ' | '.join([f"{k}={v}" for k, v in record.extra_data.items() if v])
                    if extra_str:
                        formatted += f" | Extra: {extra_str}"
                
                return formatted
        
        return StructuredFormatter()
    
    def set_context(self, **kwargs):
        """Set logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def clear_context(self):
        """Clear logging context."""
        self.context = LogContext()
    
    def _log_with_context(self, level: int, message: str, extra_data: Optional[Dict[str, Any]] = None,
                         exception: Optional[Exception] = None):
        """Internal method to log with context."""
        # Create log record
        record = self.logger.makeRecord(
            name=self.name,
            level=level,
            fn='',
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Add context and extra data
        record.context = asdict(self.context)
        record.extra_data = extra_data or {}
        
        # Add exception info if provided
        if exception:
            record.exception_info = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        # Log the record
        self.logger.handle(record)
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, extra_data)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self._log_with_context(logging.INFO, message, extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, extra_data)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
              exception: Optional[Exception] = None):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, extra_data, exception)
    
    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
                exception: Optional[Exception] = None):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, extra_data, exception)
    
    def exception(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log exception with traceback."""
        self._log_with_context(logging.ERROR, message, extra_data, sys.exc_info())
    
    def log_request(self, method: str, endpoint: str, status_code: int, 
                   response_time: float, user_id: Optional[str] = None,
                   ip_address: Optional[str] = None):
        """Log HTTP request information."""
        self.info(
            f"HTTP {method} {endpoint} - {status_code} ({response_time:.3f}s)",
            extra_data={
                'request_method': method,
                'request_endpoint': endpoint,
                'response_status': status_code,
                'response_time_ms': round(response_time * 1000, 2),
                'user_id': user_id,
                'ip_address': ip_address
            }
        )
    
    def log_security_event(self, event_type: str, description: str, 
                          severity: str = 'medium', user_id: Optional[str] = None,
                          ip_address: Optional[str] = None):
        """Log security-related events."""
        level = logging.WARNING if severity == 'high' else logging.INFO
        self._log_with_context(
            level,
            f"Security Event: {event_type} - {description}",
            extra_data={
                'event_type': event_type,
                'description': description,
                'severity': severity,
                'user_id': user_id,
                'ip_address': ip_address,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_performance_metric(self, metric_name: str, value: Union[int, float],
                              unit: str = '', tags: Optional[Dict[str, str]] = None):
        """Log performance metrics."""
        self.info(
            f"Performance Metric: {metric_name} = {value}{unit}",
            extra_data={
                'metric_name': metric_name,
                'metric_value': value,
                'metric_unit': unit,
                'metric_tags': tags or {},
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_business_event(self, event_type: str, description: str,
                          user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log business-related events."""
        self.info(
            f"Business Event: {event_type} - {description}",
            extra_data={
                'event_type': event_type,
                'description': description,
                'user_id': user_id,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def set_formatter(self, formatter_name: str):
        """Set the logging formatter."""
        if formatter_name not in self.formatters:
            raise ValueError(f"Unknown formatter: {formatter_name}")
        
        self.current_formatter = formatter_name
        
        # Update all handlers with new formatter
        for handler in self.logger.handlers:
            handler.setFormatter(self.formatters[formatter_name])
    
    def get_logger_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        stats = {
            'name': self.name,
            'level': self.logger.level,
            'handlers_count': len(self.logger.handlers),
            'formatter': self.current_formatter,
            'context': asdict(self.context)
        }
        
        # Add handler information
        handler_info = []
        for handler in self.logger.handlers:
            handler_info.append({
                'type': type(handler).__name__,
                'level': handler.level,
                'formatter': type(handler.formatter).__name__ if handler.formatter else None
            })
        stats['handlers'] = handler_info
        
        return stats
    
    def add_custom_handler(self, handler: logging.Handler):
        """Add a custom logging handler."""
        self.logger.addHandler(handler)
    
    def remove_handler(self, handler: logging.Handler):
        """Remove a logging handler."""
        if handler in self.logger.handlers:
            self.logger.removeHandler(handler)
    
    def flush(self):
        """Flush all handlers."""
        for handler in self.logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

# Import sys for exception handling
import sys

