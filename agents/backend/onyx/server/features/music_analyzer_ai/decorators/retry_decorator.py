"""
Retry Decorator - Retry failed operations
"""

from typing import Callable, Type, Tuple
import functools
import time
import logging

from .decorator import BaseDecorator

logger = logging.getLogger(__name__)


class RetryDecorator(BaseDecorator):
    """
    Decorator that retries failed operations
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        super().__init__("RetryDecorator")
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
    
    def decorate(self, func: Callable) -> Callable:
        """Decorate function with retry logic"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = self.delay
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    if attempt == self.max_retries:
                        logger.error(f"{func.__name__} failed after {self.max_retries} retries: {str(e)}")
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= self.backoff
        
        return wrapper


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Function decorator for retry"""
    decorator = RetryDecorator(
        max_retries=max_retries,
        delay=delay,
        backoff=backoff,
        exceptions=exceptions
    )
    return decorator.decorate








