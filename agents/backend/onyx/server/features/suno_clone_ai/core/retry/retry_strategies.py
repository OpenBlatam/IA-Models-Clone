"""
Retry Strategies

Advanced retry strategies with different backoff algorithms.
"""

import logging
import time
from typing import Callable, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RetryStrategy(ABC):
    """Base class for retry strategies."""
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """
        Get delay for attempt.
        
        Args:
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        pass


class ExponentialBackoff(RetryStrategy):
    """Exponential backoff strategy."""
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0
    ):
        """
        Initialize exponential backoff.
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Backoff multiplier
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
    
    def get_delay(self, attempt: int) -> float:
        """Get exponential delay."""
        delay = self.initial_delay * (self.multiplier ** attempt)
        return min(delay, self.max_delay)


class LinearBackoff(RetryStrategy):
    """Linear backoff strategy."""
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        increment: float = 1.0
    ):
        """
        Initialize linear backoff.
        
        Args:
            initial_delay: Initial delay
            max_delay: Maximum delay
            increment: Delay increment per attempt
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.increment = increment
    
    def get_delay(self, attempt: int) -> float:
        """Get linear delay."""
        delay = self.initial_delay + (self.increment * attempt)
        return min(delay, self.max_delay)


class FixedDelay(RetryStrategy):
    """Fixed delay strategy."""
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize fixed delay.
        
        Args:
            delay: Fixed delay in seconds
        """
        self.delay = delay
    
    def get_delay(self, attempt: int) -> float:
        """Get fixed delay."""
        return self.delay


def retry_with_strategy(
    strategy: RetryStrategy,
    max_attempts: int = 3,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with strategy.
    
    Args:
        strategy: Retry strategy
        max_attempts: Maximum retry attempts
        exceptions: Exceptions to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = strategy.get_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    
    return decorator



