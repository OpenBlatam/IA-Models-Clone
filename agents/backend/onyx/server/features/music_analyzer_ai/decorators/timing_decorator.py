"""
Timing Decorator - Measure execution time
"""

from typing import Callable
import functools
import time
import logging

from .decorator import BaseDecorator

logger = logging.getLogger(__name__)


class TimingDecorator(BaseDecorator):
    """
    Decorator that measures execution time
    """
    
    def __init__(self, log_threshold: float = 0.0):
        super().__init__("TimingDecorator")
        self.log_threshold = log_threshold
    
    def decorate(self, func: Callable) -> Callable:
        """Decorate function with timing"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if elapsed >= self.log_threshold:
                logger.info(f"{func.__name__} took {elapsed:.3f}s")
            
            return result
        
        return wrapper


def time_execution(log_threshold: float = 0.0):
    """Function decorator for timing"""
    decorator = TimingDecorator(log_threshold=log_threshold)
    return decorator.decorate








