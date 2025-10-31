"""
Modern Logging System for BUL
=============================

Using loguru for better logging with structured output and performance.
"""

import sys
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from loguru import logger
from ..config.modern_config import get_config, LogLevel

class StructuredFormatter:
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: Dict[str, Any]) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "message": record["message"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"]
        }
        
        # Add extra fields if present
        if "extra" in record:
            log_data.update(record["extra"])
        
        # Add exception info if present
        if record["exception"]:
            log_data["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_data, ensure_ascii=False)

class ModernLogger:
    """Modern logging system using loguru"""
    
    def __init__(self):
        self.config = get_config()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Remove default handler
        logger.remove()
        
        # Console handler
        if self.config.logging.json_logs:
            logger.add(
                sys.stdout,
                format=StructuredFormatter().format,
                level=self.config.logging.level.value,
                colorize=False,
                serialize=True
            )
        else:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=self.config.logging.level.value,
                colorize=True
            )
        
        # File handler
        if self.config.logging.file_path:
            log_file = Path(self.config.logging.file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.logging.json_logs:
                logger.add(
                    log_file,
                    format=StructuredFormatter().format,
                    level=self.config.logging.level.value,
                    rotation=f"{self.config.logging.max_file_size} bytes",
                    retention=self.config.logging.backup_count,
                    compression="gz",
                    serialize=True
                )
            else:
                logger.add(
                    log_file,
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                    level=self.config.logging.level.value,
                    rotation=f"{self.config.logging.max_file_size} bytes",
                    retention=self.config.logging.backup_count,
                    compression="gz"
                )
        
        # Error file handler (separate file for errors)
        if self.config.logging.file_path:
            error_file = Path(self.config.logging.file_path).with_suffix('.error.log')
            logger.add(
                error_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="ERROR",
                rotation=f"{self.config.logging.max_file_size} bytes",
                retention=self.config.logging.backup_count,
                compression="gz"
            )
    
    def get_logger(self, name: str = None) -> Any:
        """Get a logger instance"""
        if name:
            return logger.bind(logger_name=name)
        return logger
    
    def add_context(self, **kwargs):
        """Add context to all subsequent log messages"""
        return logger.bind(**kwargs)
    
    def log_performance(self, operation: str, duration: float, **extra):
        """Log performance metrics"""
        logger.info(
            f"Performance: {operation}",
            operation=operation,
            duration=duration,
            **extra
        )
    
    def log_api_call(self, method: str, url: str, status_code: int, duration: float, **extra):
        """Log API calls"""
        logger.info(
            f"API Call: {method} {url} - {status_code}",
            method=method,
            url=url,
            status_code=status_code,
            duration=duration,
            **extra
        )
    
    def log_document_generation(self, document_id: str, agent: str, duration: float, **extra):
        """Log document generation"""
        logger.info(
            f"Document Generated: {document_id}",
            document_id=document_id,
            agent=agent,
            duration=duration,
            **extra
        )
    
    def log_security_event(self, event_type: str, details: str, **extra):
        """Log security events"""
        logger.warning(
            f"Security Event: {event_type}",
            event_type=event_type,
            details=details,
            **extra
        )

# Global logger instance
_modern_logger: Optional[ModernLogger] = None

def get_modern_logger() -> ModernLogger:
    """Get the global modern logger instance"""
    global _modern_logger
    if _modern_logger is None:
        _modern_logger = ModernLogger()
    return _modern_logger

def get_logger(name: str = None):
    """Get a logger instance"""
    return get_modern_logger().get_logger(name)

def log_performance(operation: str, duration: float, **extra):
    """Log performance metrics"""
    get_modern_logger().log_performance(operation, duration, **extra)

def log_api_call(method: str, url: str, status_code: int, duration: float, **extra):
    """Log API calls"""
    get_modern_logger().log_api_call(method, url, status_code, duration, **extra)

def log_document_generation(document_id: str, agent: str, duration: float, **extra):
    """Log document generation"""
    get_modern_logger().log_document_generation(document_id, agent, duration, **extra)

def log_security_event(event_type: str, details: str, **extra):
    """Log security events"""
    get_modern_logger().log_security_event(event_type, details, **extra)

# Context manager for logging
class LogContext:
    """Context manager for adding logging context"""
    
    def __init__(self, **context):
        self.context = context
        self.old_context = {}
    
    def __enter__(self):
        # Store old context and add new context
        self.old_context = logger.contextualize(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old context
        logger.contextualize(**self.old_context)

# Decorator for automatic logging
def log_function_calls(log_args: bool = False, log_result: bool = False):
    """Decorator to automatically log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            # Log function entry
            if log_args:
                logger.info(f"Calling {func.__name__}", args=args, kwargs=kwargs)
            else:
                logger.info(f"Calling {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                
                # Log function exit
                duration = (datetime.now() - start_time).total_seconds()
                if log_result:
                    logger.info(f"Completed {func.__name__}", duration=duration, result=result)
                else:
                    logger.info(f"Completed {func.__name__}", duration=duration)
                
                return result
            except Exception as e:
                # Log function error
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Error in {func.__name__}", duration=duration, error=str(e))
                raise
        
        return wrapper
    return decorator

# Async version of the decorator
def log_async_function_calls(log_args: bool = False, log_result: bool = False):
    """Decorator to automatically log async function calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            # Log function entry
            if log_args:
                logger.info(f"Calling async {func.__name__}", args=args, kwargs=kwargs)
            else:
                logger.info(f"Calling async {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                
                # Log function exit
                duration = (datetime.now() - start_time).total_seconds()
                if log_result:
                    logger.info(f"Completed async {func.__name__}", duration=duration, result=result)
                else:
                    logger.info(f"Completed async {func.__name__}", duration=duration)
                
                return result
            except Exception as e:
                # Log function error
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Error in async {func.__name__}", duration=duration, error=str(e))
                raise
        
        return wrapper
    return decorator




