"""
Logging utilities
Advanced logging functions
"""

from typing import Optional, Dict, Any
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    file_path: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with configuration
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        file_path: Optional log file path
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls
    
    Args:
        logger: Logger instance
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_performance(logger: logging.Logger):
    """
    Decorator to log function performance
    
    Args:
        logger: Logger instance
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"{func.__name__} took {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {str(e)}")
                raise
        return wrapper
    return decorator


def log_error(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log error with context
    
    Args:
        logger: Logger instance
        error: Exception to log
        context: Optional context dictionary
    """
    message = f"Error: {str(error)}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message += f" | Context: {context_str}"
    
    logger.error(message, exc_info=True)


def log_info(logger: logging.Logger, message: str, **kwargs):
    """
    Log info with additional data
    
    Args:
        logger: Logger instance
        message: Log message
        **kwargs: Additional data to log
    """
    if kwargs:
        data_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"{message} | {data_str}")
    else:
        logger.info(message)


def log_warning(logger: logging.Logger, message: str, **kwargs):
    """
    Log warning with additional data
    
    Args:
        logger: Logger instance
        message: Log message
        **kwargs: Additional data to log
    """
    if kwargs:
        data_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.warning(f"{message} | {data_str}")
    else:
        logger.warning(message)


def log_debug(logger: logging.Logger, message: str, **kwargs):
    """
    Log debug with additional data
    
    Args:
        logger: Logger instance
        message: Log message
        **kwargs: Additional data to log
    """
    if kwargs:
        data_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"{message} | {data_str}")
    else:
        logger.debug(message)


def create_log_context(**kwargs) -> Dict[str, Any]:
    """
    Create log context dictionary
    
    Args:
        **kwargs: Context key-value pairs
    
    Returns:
        Context dictionary
    """
    return kwargs


def format_log_message(level: str, message: str, **kwargs) -> str:
    """
    Format log message
    
    Args:
        level: Log level
        message: Log message
        **kwargs: Additional data
    
    Returns:
        Formatted message
    """
    if kwargs:
        data_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"[{level}] {message} | {data_str}"
    
    return f"[{level}] {message}"

