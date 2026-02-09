"""
Retry Utilities - Enhanced retry mechanisms.

Provides:
- Retry decorators
- Retry context managers
- Retry strategies
- Exponential backoff
- Jitter support
- Retry configuration
- Retry statistics
"""

import time
import logging
import random
from typing import Callable, Any, Optional, TypeVar, Tuple, List, Type, Dict
from functools import wraps
from contextlib import contextmanager
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(str, Enum):
    """Retry strategies."""
    FIXED = "fixed"  # Fixed delay
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    CUSTOM = "custom"  # Custom function


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, message: str, attempts: int, last_exception: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    retry_condition: Optional[Callable[[Any], bool]] = None
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for attempt.
        
        Args:
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if attempt == 0:
            delay = self.initial_delay
        elif self.strategy == RetryStrategy.FIXED:
            delay = self.initial_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay * (1 + attempt)
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.multiplier ** attempt)
        else:
            delay = self.initial_delay
        
        # Apply max delay
        delay = min(delay, self.max_delay)
        
        # Apply jitter
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)
        
        return delay


@dataclass
class RetryResult:
    """Retry execution result."""
    success: bool
    attempts: int
    total_time: float
    last_exception: Optional[Exception] = None
    results: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "total_time": self.total_time,
            "last_exception": str(self.last_exception) if self.last_exception else None,
        }


def calculate_delay(
    attempt: int,
    base_delay: float,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    max_delay: Optional[float] = None,
    multiplier: float = 2.0,
    jitter: bool = False,
    jitter_range: float = 0.1,
    custom_func: Optional[Callable[[int], float]] = None,
) -> float:
    """
    Calculate delay for retry attempt.
    
    Args:
        attempt: Attempt number (0-indexed)
        base_delay: Base delay in seconds
        strategy: Retry strategy
        max_delay: Maximum delay (None = no limit)
        multiplier: Multiplier for exponential/linear
        jitter: Whether to add jitter
        jitter_range: Jitter range (0.0-1.0)
        custom_func: Custom delay function
    
    Returns:
        Delay in seconds
    """
    if strategy == RetryStrategy.FIXED:
        delay = base_delay
    elif strategy == RetryStrategy.EXPONENTIAL:
        delay = base_delay * (multiplier ** attempt)
    elif strategy == RetryStrategy.LINEAR:
        delay = base_delay * (1 + attempt * multiplier)
    elif strategy == RetryStrategy.CUSTOM:
        if custom_func is None:
            raise ValueError("Custom strategy requires custom_func")
        delay = custom_func(attempt)
    else:
        delay = base_delay
    
    if max_delay is not None:
        delay = min(delay, max_delay)
    
    # Apply jitter
    if jitter:
        jitter_amount = delay * jitter_range
        delay += random.uniform(-jitter_amount, jitter_amount)
        delay = max(0, delay)
    
    return delay


@contextmanager
def retry_context(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    max_delay: Optional[float] = None,
    multiplier: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    reraise: bool = True,
):
    """
    Context manager for retry logic.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        strategy: Retry strategy
        max_delay: Maximum delay
        multiplier: Multiplier for backoff
        exceptions: Exceptions to catch
        on_retry: Optional callback on retry
        reraise: Whether to reraise on final failure
    
    Example:
        >>> with retry_context(max_attempts=3, base_delay=1.0) as ctx:
        >>>     result = risky_operation()
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            yield
            return
        except exceptions as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                delay = calculate_delay(
                    attempt,
                    base_delay,
                    strategy,
                    max_delay,
                    multiplier,
                )
                
                if on_retry:
                    on_retry(e, attempt + 1)
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                
                time.sleep(delay)
            else:
                logger.error(f"All {max_attempts} attempts failed")
    
    if reraise and last_exception:
        raise RetryExhausted(
            f"Operation failed after {max_attempts} attempts",
            max_attempts,
            last_exception,
        ) from last_exception


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    max_delay: Optional[float] = None,
    multiplier: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    reraise: bool = True,
):
    """
    Decorator for retry logic.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        strategy: Retry strategy
        max_delay: Maximum delay
        multiplier: Multiplier for backoff
        exceptions: Exceptions to catch
        on_retry: Optional callback on retry
        reraise: Whether to reraise on final failure
    
    Example:
        >>> @retry(max_attempts=5, base_delay=2.0)
        >>> def unreliable_function():
        >>>     # May fail
        >>>     pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt,
                            base_delay,
                            strategy,
                            max_delay,
                            multiplier,
                        )
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        else:
                            logger.warning(
                                f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                                f"Retrying in {delay:.1f}s..."
                            )
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            if reraise and last_exception:
                raise RetryExhausted(
                    f"{func.__name__} failed after {max_attempts} attempts",
                    max_attempts,
                    last_exception,
                ) from last_exception
            
            return None  # Should not reach here if reraise=True
        return wrapper
    return decorator


def execute_with_retry(
    func: Callable[..., T],
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    max_delay: Optional[float] = None,
    multiplier: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    default_return: Any = None,
    **kwargs,
) -> T:
    """
    Execute function with retry logic.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        strategy: Retry strategy
        max_delay: Maximum delay
        multiplier: Multiplier for backoff
        exceptions: Exceptions to catch
        on_retry: Optional callback on retry
        default_return: Return value on final failure
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return on final failure
    
    Example:
        >>> result = execute_with_retry(
        >>>     unreliable_function,
        >>>     arg1, arg2,
        >>>     max_attempts=5,
        >>>     base_delay=2.0
        >>> )
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                delay = calculate_delay(
                    attempt,
                    base_delay,
                    strategy,
                    max_delay,
                    multiplier,
                )
                
                if on_retry:
                    on_retry(e, attempt + 1)
                else:
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                
                time.sleep(delay)
            else:
                logger.error(f"{func.__name__} failed after {max_attempts} attempts")
    
    if default_return is not None:
        return default_return
    
    if last_exception:
        raise RetryExhausted(
            f"{func.__name__} failed after {max_attempts} attempts",
            max_attempts,
            last_exception,
        ) from last_exception
    
    return None


class RetryManager:
    """
    Manager for retry operations.
    
    Provides centralized retry configuration and management.
    """
    
    def __init__(
        self,
        default_max_attempts: int = 3,
        default_base_delay: float = 1.0,
        default_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    ):
        """
        Initialize retry manager.
        
        Args:
            default_max_attempts: Default max attempts
            default_base_delay: Default base delay
            default_strategy: Default retry strategy
        """
        self.default_max_attempts = default_max_attempts
        self.default_base_delay = default_base_delay
        self.default_strategy = default_strategy
    
    def execute(
        self,
        func: Callable[..., T],
        *args,
        max_attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
        strategy: Optional[RetryStrategy] = None,
        **kwargs,
    ) -> T:
        """
        Execute function with retry using manager defaults.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            max_attempts: Max attempts (uses default if None)
            base_delay: Base delay (uses default if None)
            strategy: Retry strategy (uses default if None)
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        """
        return execute_with_retry(
            func,
            *args,
            max_attempts=max_attempts or self.default_max_attempts,
            base_delay=base_delay or self.default_base_delay,
            strategy=strategy or self.default_strategy,
            **kwargs,
        )


# Global retry manager instance
_default_retry_manager = RetryManager()


def get_retry_manager() -> RetryManager:
    """Get default retry manager."""
    return _default_retry_manager


class RetryExecutor:
    """Retry executor with policy support."""
    
    def __init__(self, policy: Optional[RetryPolicy] = None):
        """
        Initialize retry executor.
        
        Args:
            policy: Optional retry policy
        """
        self.policy = policy or RetryPolicy()
        self.stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
        }
    
    def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> RetryResult:
        """
        Execute function with retry policy.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            RetryResult
        """
        start_time = time.time()
        last_exception = None
        results = []
        
        for attempt in range(self.policy.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                # Check retry condition if provided
                if self.policy.retry_condition and not self.policy.retry_condition(result):
                    return RetryResult(
                        success=True,
                        attempts=attempt + 1,
                        total_time=time.time() - start_time,
                        results=[result],
                    )
                
                results.append(result)
                self.stats["total_attempts"] += attempt + 1
                self.stats["successful_retries"] += 1
                
                return RetryResult(
                    success=True,
                    attempts=attempt + 1,
                    total_time=time.time() - start_time,
                    results=results,
                )
            except tuple(self.policy.retry_on) as e:
                last_exception = e
                self.stats["total_attempts"] += attempt + 1
                
                if attempt < self.policy.max_attempts - 1:
                    delay = self.policy.calculate_delay(attempt)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{self.policy.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.stats["failed_retries"] += 1
                    logger.error(f"{func.__name__} failed after {self.policy.max_attempts} attempts")
        
        return RetryResult(
            success=False,
            attempts=self.policy.max_attempts,
            total_time=time.time() - start_time,
            last_exception=last_exception,
            results=results,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return self.stats.copy()


__all__ = [
    "RetryStrategy",
    "RetryExhausted",
    "RetryPolicy",
    "RetryResult",
    "RetryExecutor",
    "calculate_delay",
    "retry_context",
    "retry",
    "execute_with_retry",
    "RetryManager",
    "get_retry_manager",
]

