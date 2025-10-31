from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
import json
import traceback
import inspect
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import sys
import os
from typing import Any, List, Dict, Optional
import asyncio
"""
Structured Logger Module
=======================

Comprehensive structured logging with context capture for:
- Module, function, and parameter tracking
- Error context and stack traces
- Performance metrics
- Security events
- Audit trails
"""


class StructuredLogger:
    """
    Structured logger with comprehensive context capture.
    
    Features:
    - Module, function, and parameter tracking
    - Error context with stack traces
    - Performance metrics
    - Structured JSON output
    - Configurable log levels
    - File and console output
    """
    
    def __init__(self, 
                 name: str = "cybersecurity_toolkit",
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_file: bool = True):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            enable_console: Enable console output
            enable_file: Enable file output
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.enable_console = enable_console
        self.enable_file = enable_file
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.json_formatter = self._create_json_formatter()
        self.console_formatter = self._create_console_formatter()
        
        # Setup handlers
        if enable_console:
            self._setup_console_handler()
        
        if enable_file and log_file:
            self._setup_file_handler()
    
    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""
        class JSONFormatter(logging.Formatter):
            def format(self, record) -> Any:
                # Create structured log entry
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "process_id": record.process,
                    "thread_id": record.thread,
                }
                
                # Add extra fields if present
                if hasattr(record, 'context'):
                    log_entry['context'] = record.context
                
                if hasattr(record, 'parameters'):
                    log_entry['parameters'] = record.parameters
                
                if hasattr(record, 'performance'):
                    log_entry['performance'] = record.performance
                
                if hasattr(record, 'error_context'):
                    log_entry['error_context'] = record.error_context
                
                if hasattr(record, 'security_event'):
                    log_entry['security_event'] = record.security_event
                
                return json.dumps(log_entry, default=str)
        
        return JSONFormatter()
    
    def _create_console_formatter(self) -> logging.Formatter:
        """Create console formatter for human-readable output."""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        )
    
    def _setup_console_handler(self) -> Any:
        """Setup console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self) -> Any:
        """Setup file handler."""
        # Ensure log directory exists
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(file_handler)
    
    def _get_calling_context(self, depth: int = 2) -> Dict[str, Any]:
        """
        Get calling context information.
        
        Args:
            depth: Stack depth to analyze
            
        Returns:
            Context information dictionary
        """
        try:
            frame = inspect.currentframe()
            for _ in range(depth):
                if frame:
                    frame = frame.f_back
            
            if frame:
                return {
                    "module": frame.f_globals.get('__name__', 'unknown'),
                    "function": frame.f_code.co_name,
                    "line": frame.f_lineno,
                    "filename": frame.f_code.co_filename
                }
        except Exception:
            pass
        
        return {
            "module": "unknown",
            "function": "unknown",
            "line": 0,
            "filename": "unknown"
        }
    
    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for logging (remove sensitive data).
        
        Args:
            parameters: Parameters to sanitize
            
        Returns:
            Sanitized parameters
        """
        if not parameters:
            return {}
        
        sensitive_keys = {
            'password', 'token', 'key', 'secret', 'auth', 'credential',
            'private_key', 'api_key', 'access_token', 'refresh_token'
        }
        
        sanitized = {}
        for key, value in parameters.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_parameters(value)
            elif isinstance(value, list) and len(value) > 10:
                sanitized[key] = f"[{len(value)} items]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def log_function_entry(self, 
                          function_name: str,
                          parameters: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None):
        """
        Log function entry with context.
        
        Args:
            function_name: Name of the function being entered
            parameters: Function parameters
            context: Additional context information
        """
        calling_context = self._get_calling_context()
        
        extra = {
            'context': {
                'event_type': 'function_entry',
                'function_name': function_name,
                'calling_context': calling_context,
                **(context or {})
            },
            'parameters': self._sanitize_parameters(parameters or {})
        }
        
        self.logger.info(f"Entering function: {function_name}", extra=extra)
    
    def log_function_exit(self,
                         function_name: str,
                         return_value: Optional[Any] = None,
                         execution_time: Optional[float] = None,
                         context: Optional[Dict[str, Any]] = None):
        """
        Log function exit with context.
        
        Args:
            function_name: Name of the function being exited
            return_value: Function return value
            execution_time: Function execution time in seconds
            context: Additional context information
        """
        calling_context = self._get_calling_context()
        
        extra = {
            'context': {
                'event_type': 'function_exit',
                'function_name': function_name,
                'calling_context': calling_context,
                **(context or {})
            },
            'performance': {
                'execution_time': execution_time
            } if execution_time else None
        }
        
        message = f"Exiting function: {function_name}"
        if execution_time:
            message += f" (execution_time: {execution_time:.4f}s)"
        
        self.logger.info(message, extra=extra)
    
    def log_error(self,
                  error: Exception,
                  function_name: str,
                  parameters: Optional[Dict[str, Any]] = None,
                  context: Optional[Dict[str, Any]] = None,
                  include_traceback: bool = True):
        """
        Log error with comprehensive context.
        
        Args:
            error: Exception that occurred
            function_name: Name of the function where error occurred
            parameters: Function parameters
            context: Additional context information
            include_traceback: Include full traceback
        """
        calling_context = self._get_calling_context()
        
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'function_name': function_name,
            'calling_context': calling_context,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error_id': str(uuid.uuid4())
        }
        
        if include_traceback:
            error_context['traceback'] = traceback.format_exc()
        
        extra = {
            'context': {
                'event_type': 'error',
                **(context or {})
            },
            'parameters': self._sanitize_parameters(parameters or {}),
            'error_context': error_context
        }
        
        self.logger.error(
            f"Error in {function_name}: {type(error).__name__}: {str(error)}",
            extra=extra
        )
    
    def log_security_event(self,
                          event_type: str,
                          event_data: Dict[str, Any],
                          severity: str = "INFO",
                          context: Optional[Dict[str, Any]] = None):
        """
        Log security event with context.
        
        Args:
            event_type: Type of security event
            event_data: Event data
            severity: Event severity
            context: Additional context information
        """
        calling_context = self._get_calling_context()
        
        security_event = {
            'event_type': event_type,
            'event_data': event_data,
            'severity': severity,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_id': str(uuid.uuid4()),
            'calling_context': calling_context
        }
        
        extra = {
            'context': {
                'event_type': 'security_event',
                **(context or {})
            },
            'security_event': security_event
        }
        
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(f"Security event: {event_type}", extra=extra)
    
    def log_performance(self,
                       operation: str,
                       execution_time: float,
                       metrics: Optional[Dict[str, Any]] = None,
                       context: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            execution_time: Execution time in seconds
            metrics: Additional performance metrics
            context: Additional context information
        """
        calling_context = self._get_calling_context()
        
        performance_data = {
            'operation': operation,
            'execution_time': execution_time,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'calling_context': calling_context,
            **(metrics or {})
        }
        
        extra = {
            'context': {
                'event_type': 'performance',
                **(context or {})
            },
            'performance': performance_data
        }
        
        self.logger.info(f"Performance: {operation} took {execution_time:.4f}s", extra=extra)
    
    def log_validation_error(self,
                           validation_type: str,
                           field_name: str,
                           field_value: Any,
                           error_message: str,
                           context: Optional[Dict[str, Any]] = None):
        """
        Log validation error with context.
        
        Args:
            validation_type: Type of validation that failed
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            error_message: Validation error message
            context: Additional context information
        """
        calling_context = self._get_calling_context()
        
        validation_context = {
            'validation_type': validation_type,
            'field_name': field_name,
            'field_value': str(field_value)[:100],  # Truncate long values
            'error_message': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'calling_context': calling_context
        }
        
        extra = {
            'context': {
                'event_type': 'validation_error',
                **(context or {})
            },
            'error_context': validation_context
        }
        
        self.logger.warning(
            f"Validation error: {validation_type} failed for {field_name}: {error_message}",
            extra=extra
        )

# Global logger instance
_global_logger: Optional[StructuredLogger] = None

def get_logger(name: str = "cybersecurity_toolkit") -> StructuredLogger:
    """
    Get or create global logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        log_file = logs_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        _global_logger = StructuredLogger(
            name=name,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=str(log_file),
            enable_console=True,
            enable_file=True
        )
    
    return _global_logger

def log_function_call(func) -> Any:
    """
    Decorator to automatically log function calls with context.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger()
        
        # Get function parameters
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        parameters = dict(bound_args.arguments)
        
        # Log function entry
        logger.log_function_entry(func.__name__, parameters)
        
        start_time = time.time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log successful exit
            execution_time = time.time() - start_time
            logger.log_function_exit(func.__name__, result, execution_time)
            
            return result
            
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            logger.log_error(e, func.__name__, parameters)
            logger.log_function_exit(func.__name__, None, execution_time)
            raise
    
    return wrapper

def log_async_function_call(func) -> Any:
    """
    Decorator to automatically log async function calls with context.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    async def wrapper(*args, **kwargs) -> Any:
        logger = get_logger()
        
        # Get function parameters
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        parameters = dict(bound_args.arguments)
        
        # Log function entry
        logger.log_function_entry(func.__name__, parameters)
        
        start_time = time.time()
        
        try:
            # Execute async function
            result = await func(*args, **kwargs)
            
            # Log successful exit
            execution_time = time.time() - start_time
            logger.log_function_exit(func.__name__, result, execution_time)
            
            return result
            
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            logger.log_error(e, func.__name__, parameters)
            logger.log_function_exit(func.__name__, None, execution_time)
            raise
    
    return wrapper

# --- Named Exports ---

__all__ = [
    'StructuredLogger',
    'get_logger',
    'log_function_call',
    'log_async_function_call'
] 