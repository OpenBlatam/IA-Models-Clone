"""
Retry Manager for Color Grading AI
===================================

Advanced retry management with exponential backoff and jitter.
"""

import logging
import asyncio
import random
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    retryable_exceptions: tuple = (Exception,)


class RetryManager:
    """
    Advanced retry manager.
    
    Features:
    - Multiple retry strategies
    - Exponential backoff
    - Jitter for distributed systems
    - Custom retry logic
    - Retry statistics
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Optional retry configuration
        """
        self.config = config or RetryConfig()
        self._retry_stats: Dict[str, Dict[str, Any]] = {}
    
    async def execute_with_retry(
        self,
        func: Callable,
        operation_name: str = "operation",
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            operation_name: Operation name for tracking
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        attempt = 0
        
        while attempt < self.config.max_attempts:
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success
                self._record_success(operation_name, attempt + 1)
                return result
            
            except self.config.retryable_exceptions as e:
                last_exception = e
                attempt += 1
                
                if attempt >= self.config.max_attempts:
                    # All retries exhausted
                    self._record_failure(operation_name, attempt)
                    logger.error(f"Operation {operation_name} failed after {attempt} attempts")
                    raise
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Operation {operation_name} failed (attempt {attempt}/{self.config.max_attempts}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                await asyncio.sleep(delay)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.initial_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.initial_delay * attempt
        else:
            delay = self.config.initial_delay
        
        # Apply max delay
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)
        
        return delay
    
    def _record_success(self, operation_name: str, attempts: int):
        """Record successful retry."""
        if operation_name not in self._retry_stats:
            self._retry_stats[operation_name] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "total_attempts": 0,
            }
        
        stats = self._retry_stats[operation_name]
        stats["total"] += 1
        stats["successful"] += 1
        stats["total_attempts"] += attempts
    
    def _record_failure(self, operation_name: str, attempts: int):
        """Record failed retry."""
        if operation_name not in self._retry_stats:
            self._retry_stats[operation_name] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "total_attempts": 0,
            }
        
        stats = self._retry_stats[operation_name]
        stats["total"] += 1
        stats["failed"] += 1
        stats["total_attempts"] += attempts
    
    def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get retry statistics.
        
        Args:
            operation_name: Optional operation name filter
            
        Returns:
            Statistics dictionary
        """
        if operation_name:
            return self._retry_stats.get(operation_name, {})
        return self._retry_stats.copy()
    
    def reset_stats(self, operation_name: Optional[str] = None):
        """Reset statistics."""
        if operation_name:
            if operation_name in self._retry_stats:
                del self._retry_stats[operation_name]
        else:
            self._retry_stats.clear()


def retry_on_failure(config: Optional[RetryConfig] = None):
    """
    Decorator for automatic retry.
    
    Args:
        config: Optional retry configuration
    """
    retry_manager = RetryManager(config)
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_manager.execute_with_retry(
                func,
                operation_name=func.__name__,
                *args,
                **kwargs
            )
        return wrapper
    return decorator

