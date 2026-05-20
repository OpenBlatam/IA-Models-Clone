"""
Retry Handler with Exponential Backoff
Implements retry logic with exponential backoff for resilient service communication
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Type, Tuple, Optional
import random

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Add random jitter to delay
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Last attempt failed, raise exception
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts: {str(e)}"
                        )
                        raise
                    
                    # Calculate delay with jitter
                    if jitter:
                        jitter_value = random.uniform(0, delay * 0.1)
                        actual_delay = min(delay + jitter_value, max_delay)
                    else:
                        actual_delay = min(delay, max_delay)
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                        f"Retrying in {actual_delay:.2f} seconds..."
                    )
                    
                    time.sleep(actual_delay)
                    delay *= backoff_factor
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class RetryHandler:
    """
    Retry handler class for more complex retry scenarios
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    def execute(
        self,
        func: Callable,
        *args,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            *args: Function arguments
            exceptions: Exceptions to catch and retry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        delay = self.initial_delay
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(
                        f"Function {func.__name__} failed after {self.max_retries + 1} attempts: {str(e)}"
                    )
                    raise
                
                # Calculate delay with jitter
                if self.jitter:
                    jitter_value = random.uniform(0, delay * 0.1)
                    actual_delay = min(delay + jitter_value, self.max_delay)
                else:
                    actual_delay = min(delay, self.max_delay)
                
                logger.warning(
                    f"Function {func.__name__} failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}. "
                    f"Retrying in {actual_delay:.2f} seconds..."
                )
                
                time.sleep(actual_delay)
                delay *= self.backoff_factor
        
        if last_exception:
            raise last_exception















