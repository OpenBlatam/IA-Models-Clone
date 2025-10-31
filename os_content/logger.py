#!/usr/bin/env python3
"""
Advanced logging system for OS Content System
Supports structured logging, multiple handlers, and performance monitoring
"""

import logging
import logging.handlers
import sys
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import structlog
from structlog import get_logger, configure
from structlog.stdlib import LoggerFactory
from structlog.processors import TimeStamper, JSONRenderer, add_log_level
from structlog.dev import ConsoleRenderer
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored output
colorama.init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        # Add color to the message for errors
        if record.levelno >= logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)

class PerformanceLogger:
    """Performance monitoring logger"""
    
    def __init__(self, logger):
        self.logger = logger
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str, extra_data: Optional[Dict[str, Any]] = None):
        """End timing an operation and log the duration"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.logger.info(
                f"Operation completed: {operation}",
                extra={
                    'operation': operation,
                    'duration_seconds': duration,
                    'extra_data': extra_data or {}
                }
            )
            del self.start_times[operation]
        else:
            self.logger.warning(f"Timer not found for operation: {operation}")

class OSContentLogger:
    """Main logger class for OS Content System"""
    
    def __init__(self, name: str = "os_content", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = None
        self.performance_logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with all handlers and formatters"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create the logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / f"{self.name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / f"{self.name}_error.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s\n'
            'Exception: %(exc_info)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
        
        # JSON file handler for structured logging
        json_handler = logging.handlers.RotatingFileHandler(
            logs_dir / f"{self.name}_structured.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        json_handler.setLevel(logging.INFO)
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)
        
        # Performance logger
        self.performance_logger = PerformanceLogger(self.logger)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance"""
        if name:
            return logging.getLogger(f"{self.name}.{name}")
        return self.logger
    
    def get_performance_logger(self) -> PerformanceLogger:
        """Get the performance logger"""
        return self.performance_logger
    
    def log_with_context(self, level: str, message: str, **context):
        """Log a message with additional context"""
        extra_fields = {'extra_fields': context}
        
        if level.upper() == 'DEBUG':
            self.logger.debug(message, extra=extra_fields)
        elif level.upper() == 'INFO':
            self.logger.info(message, extra=extra_fields)
        elif level.upper() == 'WARNING':
            self.logger.warning(message, extra=extra_fields)
        elif level.upper() == 'ERROR':
            self.logger.error(message, extra=extra_fields)
        elif level.upper() == 'CRITICAL':
            self.logger.critical(message, extra=extra_fields)
    
    def log_api_request(self, method: str, path: str, status_code: int, duration: float, **kwargs):
        """Log API request details"""
        self.logger.info(
            f"API Request: {method} {path} - {status_code} ({duration:.3f}s)",
            extra={
                'extra_fields': {
                    'type': 'api_request',
                    'method': method,
                    'path': path,
                    'status_code': status_code,
                    'duration': duration,
                    **kwargs
                }
            }
        )
    
    def log_database_operation(self, operation: str, table: str, duration: float, **kwargs):
        """Log database operation details"""
        self.logger.info(
            f"Database: {operation} on {table} ({duration:.3f}s)",
            extra={
                'extra_fields': {
                    'type': 'database_operation',
                    'operation': operation,
                    'table': table,
                    'duration': duration,
                    **kwargs
                }
            }
        )
    
    def log_ml_operation(self, operation: str, model: str, duration: float, **kwargs):
        """Log machine learning operation details"""
        self.logger.info(
            f"ML Operation: {operation} with {model} ({duration:.3f}s)",
            extra={
                'extra_fields': {
                    'type': 'ml_operation',
                    'operation': operation,
                    'model': model,
                    'duration': duration,
                    **kwargs
                }
            }
        )

# Setup structlog for advanced logging
configure(
    processors=[
        add_log_level,
        TimeStamper(fmt="iso"),
        JSONRenderer() if not sys.stdout.isatty() else ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Global logger instance
logger = OSContentLogger()

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance"""
    return logger.get_logger(name)

def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger"""
    return logger.get_performance_logger()

def log_with_context(level: str, message: str, **context):
    """Log a message with additional context"""
    logger.log_with_context(level, message, **context)

def log_api_request(method: str, path: str, status_code: int, duration: float, **kwargs):
    """Log API request details"""
    logger.log_api_request(method, path, status_code, duration, **kwargs)

def log_database_operation(operation: str, table: str, duration: float, **kwargs):
    """Log database operation details"""
    logger.log_database_operation(operation, table, duration, **kwargs)

def log_ml_operation(operation: str, model: str, duration: float, **kwargs):
    """Log machine learning operation details"""
    logger.log_ml_operation(operation, model, duration, **kwargs)
