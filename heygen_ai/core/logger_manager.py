"""
Centralized Logging Management System
====================================

Manages all logging configuration for the HeyGen AI system:
- Multiple log levels and handlers
- Structured logging with JSON format
- Log rotation and compression
- Performance monitoring
- Centralized log configuration
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import traceback
import os

from .config_manager import get_config


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
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
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        level = f"{color}{record.levelname:8}{reset}"
        logger = f"{color}{record.name}{reset}"
        message = record.getMessage()
        
        return f"{timestamp} | {level} | {logger} | {message}"


class LoggerManager:
    """Centralized logging manager"""
    
    def __init__(self):
        self.config = get_config()
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        
        # Create logs directory
        self.logs_dir = Path(self.config.system.log_file).parent
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.monitoring.log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        self._add_console_handler()
        
        # Add file handler
        self._add_file_handler()
        
        # Add error file handler
        self._add_error_file_handler()
        
        # Add performance handler
        if self.config.monitoring.enable_metrics:
            self._add_performance_handler()
        
        # Set logging level for third-party libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    def _add_console_handler(self):
        """Add console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.config.system.debug:
            # Use colored formatter for development
            formatter = ColoredFormatter()
        else:
            # Use structured formatter for production
            formatter = StructuredFormatter()
        
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
        self.handlers['console'] = console_handler
    
    def _add_file_handler(self):
        """Add file handler with rotation"""
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.system.log_file,
            maxBytes=self.config.system.max_log_size,
            backupCount=self.config.system.backup_logs,
            encoding='utf-8'
        )
        
        formatter = StructuredFormatter()
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        self.handlers['file'] = file_handler
    
    def _add_error_file_handler(self):
        """Add error-only file handler"""
        error_log_file = self.logs_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_file,
            maxBytes=self.config.system.max_log_size,
            backupCount=self.config.system.backup_logs,
            encoding='utf-8'
        )
        
        formatter = StructuredFormatter()
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(error_handler)
        self.handlers['error_file'] = error_handler
    
    def _add_performance_handler(self):
        """Add performance logging handler"""
        perf_log_file = self.logs_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=perf_log_file,
            maxBytes=self.config.system.max_log_size,
            backupCount=self.config.system.backup_logs,
            encoding='utf-8'
        )
        
        formatter = StructuredFormatter()
        perf_handler.setFormatter(formatter)
        perf_handler.setLevel(logging.INFO)
        
        # Create performance logger
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
        
        self.loggers['performance'] = perf_logger
        self.handlers['performance'] = perf_handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def set_level(self, logger_name: str, level: Union[str, int]):
        """Set logging level for a specific logger"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        logger = self.get_logger(logger_name)
        logger.setLevel(level)
    
    def add_extra_fields(self, logger_name: str, extra_fields: Dict[str, Any]):
        """Add extra fields to log records for a specific logger"""
        logger = self.get_logger(logger_name)
        
        # Create a custom filter to add extra fields
        class ExtraFieldsFilter(logging.Filter):
            def __init__(self, fields: Dict[str, Any]):
                super().__init__()
                self.fields = fields
            
            def filter(self, record: logging.LogRecord) -> bool:
                record.extra_fields = self.fields
                return True
        
        # Remove existing filters
        for handler in logger.handlers:
            for filter_obj in list(handler.filters):
                if isinstance(filter_obj, ExtraFieldsFilter):
                    handler.removeFilter(filter_obj)
        
        # Add new filter
        for handler in logger.handlers:
            handler.addFilter(ExtraFieldsFilter(extra_fields))
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        if 'performance' in self.loggers:
            perf_logger = self.loggers['performance']
            extra_fields = {
                "operation": operation,
                "duration_ms": round(duration * 1000, 2),
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
            
            perf_logger.info(f"Performance: {operation} took {duration:.3f}s", 
                           extra={'extra_fields': extra_fields})
    
    def log_api_request(self, method: str, path: str, status_code: int, 
                       duration: float, user_id: Optional[str] = None):
        """Log API request details"""
        logger = self.get_logger("api")
        
        extra_fields = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        level = logging.INFO if status_code < 400 else logging.WARNING
        logger.log(level, f"API Request: {method} {path} -> {status_code} ({duration:.3f}s)",
                  extra={'extra_fields': extra_fields})
    
    def log_security_event(self, event_type: str, description: str, 
                          user_id: Optional[str] = None, ip_address: Optional[str] = None):
        """Log security-related events"""
        logger = self.get_logger("security")
        
        extra_fields = {
            "event_type": event_type,
            "description": description,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.warning(f"Security Event: {event_type} - {description}",
                      extra={'extra_fields': extra_fields})
    
    def log_database_operation(self, operation: str, table: str, duration: float, 
                              rows_affected: Optional[int] = None):
        """Log database operations"""
        logger = self.get_logger("database")
        
        extra_fields = {
            "operation": operation,
            "table": table,
            "duration_ms": round(duration * 1000, 2),
            "rows_affected": rows_affected,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Database: {operation} on {table} took {duration:.3f}s",
                   extra={'extra_fields': extra_fields})
    
    def log_cache_operation(self, operation: str, key: str, hit: bool, duration: float):
        """Log cache operations"""
        logger = self.get_logger("cache")
        
        extra_fields = {
            "operation": operation,
            "key": key,
            "hit": hit,
            "duration_ms": round(duration * 1000, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.debug(f"Cache: {operation} {key} {'HIT' if hit else 'MISS'} ({duration:.3f}s)",
                    extra={'extra_fields': extra_fields})
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any], 
                             logger_name: str = "error"):
        """Log error with additional context"""
        logger = self.get_logger(logger_name)
        
        extra_fields = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.error(f"Error: {type(error).__name__}: {str(error)}",
                    extra={'extra_fields': extra_fields}, exc_info=True)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "total_loggers": len(self.loggers),
            "total_handlers": len(self.handlers),
            "log_level": self.config.monitoring.log_level,
            "log_file": self.config.system.log_file,
            "max_log_size": self.config.system.max_log_size,
            "backup_logs": self.config.system.backup_logs,
            "environment": self.config.system.environment.value,
            "debug_mode": self.config.system.debug
        }
        
        # Add file sizes if they exist
        log_files = [
            self.config.system.log_file,
            self.logs_dir / "errors.log",
            self.logs_dir / "performance.log"
        ]
        
        file_sizes = {}
        for log_file in log_files:
            if Path(log_file).exists():
                file_sizes[Path(log_file).name] = Path(log_file).stat().st_size
        
        stats["file_sizes"] = file_sizes
        
        return stats
    
    def cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            # Get all log files
            log_files = list(self.logs_dir.glob("*.log*"))
            
            # Sort by modification time
            log_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Keep only the most recent files
            files_to_keep = self.config.system.backup_logs + 1  # +1 for current log
            
            if len(log_files) > files_to_keep:
                files_to_remove = log_files[:-files_to_keep]
                
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        logging.info(f"Removed old log file: {file_path}")
                    except Exception as e:
                        logging.warning(f"Failed to remove old log file {file_path}: {e}")
        
        except Exception as e:
            logging.error(f"Error during log cleanup: {e}")


# Global logger manager instance
logger_manager = LoggerManager()


def get_logger(name: str) -> logging.Logger:
    """Get logger from global manager"""
    return logger_manager.get_logger(name)


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics using global manager"""
    logger_manager.log_performance(operation, duration, **kwargs)


def log_api_request(method: str, path: str, status_code: int, 
                   duration: float, user_id: Optional[str] = None):
    """Log API request using global manager"""
    logger_manager.log_api_request(method, path, status_code, duration, user_id)


def log_security_event(event_type: str, description: str, 
                      user_id: Optional[str] = None, ip_address: Optional[str] = None):
    """Log security event using global manager"""
    logger_manager.log_security_event(event_type, description, user_id, ip_address)


def log_database_operation(operation: str, table: str, duration: float, 
                          rows_affected: Optional[int] = None):
    """Log database operation using global manager"""
    logger_manager.log_database_operation(operation, table, duration, rows_affected)


def log_cache_operation(operation: str, key: str, hit: bool, duration: float):
    """Log cache operation using global manager"""
    logger_manager.log_cache_operation(operation, key, hit, duration)


def log_error_with_context(error: Exception, context: Dict[str, Any], 
                         logger_name: str = "error"):
    """Log error with context using global manager"""
    logger_manager.log_error_with_context(error, context, logger_name)


if __name__ == "__main__":
    # Example usage
    logger = get_logger("example")
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Log performance
    log_performance("database_query", 0.125, table="users", rows=100)
    
    # Log API request
    log_api_request("GET", "/api/users", 200, 0.045, user_id="123")
    
    # Log security event
    log_security_event("login_failed", "Invalid credentials", ip_address="192.168.1.1")
    
    # Get stats
    stats = logger_manager.get_log_stats()
    print("Logging Statistics:")
    print(json.dumps(stats, indent=2, default=str))
