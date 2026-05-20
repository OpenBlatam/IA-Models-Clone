"""
Logging Decorator - Log function calls
"""

from typing import Callable
import functools
import logging

from .decorator import BaseDecorator

logger = logging.getLogger(__name__)


class LoggingDecorator(BaseDecorator):
    """
    Decorator that logs function calls
    """
    
    def __init__(self, log_level: int = logging.INFO, log_args: bool = False):
        super().__init__("LoggingDecorator")
        self.log_level = log_level
        self.log_args = log_args
    
    def decorate(self, func: Callable) -> Callable:
        """Decorate function with logging"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(self.log_level, f"Calling {func.__name__}")
            
            if self.log_args:
                logger.debug(f"Args: {args}, Kwargs: {kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.log(self.log_level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {str(e)}", exc_info=True)
                raise
        
        return wrapper


def log_calls(log_level: int = logging.INFO, log_args: bool = False):
    """Function decorator for logging"""
    decorator = LoggingDecorator(log_level=log_level, log_args=log_args)
    return decorator.decorate








