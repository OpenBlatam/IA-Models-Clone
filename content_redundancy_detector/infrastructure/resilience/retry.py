"""
Advanced Retry Mechanism - Exponential backoff and jitter
Production-ready retry system with multiple strategies
"""

import asyncio
import time
import random
from typing import Any, Callable, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED = "fixed"           # Fixed delay
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"         # Linear backoff
    RANDOM = "random"        # Random jitter

@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    retryable_exceptions: tuple = (Exception,)
    timeout: Optional[float] = None

class ExponentialBackoff:
    """Exponential backoff calculator"""
    
    def __init__(self, initial_delay: float = 1.0, multiplier: float = 2.0, max_delay: float = 60.0):
        self.initial_delay = initial_delay
        self.multiplier = multiplier
        self.max_delay = max_delay

    def calculate(self, attempt: int, jitter: float = 0.0) -> float:
        """Calculate delay for attempt number"""
        delay = min(
            self.initial_delay * (self.multiplier ** (attempt - 1)),
            self.max_delay
        )
        
        # Add jitter
        if jitter > 0:
            jitter_value = delay * jitter * (2 * random.random() - 1)
            delay = max(0, delay + jitter_value)
        
        return delay

async def retry_async(
    func: Callable,
    *args,
    policy: RetryPolicy = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs
) -> Any:
    """Retry async function with policy"""
    policy = policy or RetryPolicy()
    last_exception = None
    
    for attempt in range(1, policy.max_attempts + 1):
        try:
            # Check timeout
            if policy.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=policy.timeout
                )
            else:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            
            return result
            
        except policy.retryable_exceptions as e:
            last_exception = e
            
            # Don't retry on last attempt
            if attempt >= policy.max_attempts:
                logger.error(
                    f"Retry failed after {attempt} attempts: {e}"
                )
                raise
            
            # Calculate delay
            delay = _calculate_delay(attempt, policy)
            
            # Callback
            if on_retry:
                try:
                    on_retry(attempt, e)
                except Exception as callback_error:
                    logger.warning(f"Retry callback error: {callback_error}")
            
            logger.warning(
                f"Retry attempt {attempt}/{policy.max_attempts} after {delay:.2f}s: {e}"
            )
            
            await asyncio.sleep(delay)
        
        except Exception as e:
            # Non-retryable exception
            logger.error(f"Non-retryable exception: {e}")
            raise
    
    # Should not reach here
    if last_exception:
        raise last_exception
    raise Exception("Retry exhausted without result")

def retry_sync(
    func: Callable,
    *args,
    policy: RetryPolicy = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs
) -> Any:
    """Retry sync function with policy"""
    policy = policy or RetryPolicy()
    last_exception = None
    
    for attempt in range(1, policy.max_attempts + 1):
        try:
            result = func(*args, **kwargs)
            return result
            
        except policy.retryable_exceptions as e:
            last_exception = e
            
            # Don't retry on last attempt
            if attempt >= policy.max_attempts:
                logger.error(
                    f"Retry failed after {attempt} attempts: {e}"
                )
                raise
            
            # Calculate delay
            delay = _calculate_delay(attempt, policy)
            
            # Callback
            if on_retry:
                try:
                    on_retry(attempt, e)
                except Exception as callback_error:
                    logger.warning(f"Retry callback error: {callback_error}")
            
            logger.warning(
                f"Retry attempt {attempt}/{policy.max_attempts} after {delay:.2f}s: {e}"
            )
            
            time.sleep(delay)
        
        except Exception as e:
            # Non-retryable exception
            logger.error(f"Non-retryable exception: {e}")
            raise
    
    # Should not reach here
    if last_exception:
        raise last_exception
    raise Exception("Retry exhausted without result")

def _calculate_delay(attempt: int, policy: RetryPolicy) -> float:
    """Calculate delay based on strategy"""
    jitter = policy.jitter_range if policy.jitter else 0.0
    
    if policy.strategy == RetryStrategy.FIXED:
        delay = policy.initial_delay
    
    elif policy.strategy == RetryStrategy.EXPONENTIAL:
        backoff = ExponentialBackoff(
            policy.initial_delay,
            policy.multiplier,
            policy.max_delay
        )
        delay = backoff.calculate(attempt, jitter)
    
    elif policy.strategy == RetryStrategy.LINEAR:
        delay = min(
            policy.initial_delay * attempt,
            policy.max_delay
        )
        if jitter > 0:
            jitter_value = delay * jitter * (2 * random.random() - 1)
            delay = max(0, delay + jitter_value)
    
    elif policy.strategy == RetryStrategy.RANDOM:
        delay = random.uniform(
            policy.initial_delay,
            min(policy.initial_delay * attempt, policy.max_delay)
        )
    
    else:
        delay = policy.initial_delay
    
    return delay






