"""
PDF Variantes - Retry Pattern
Exponential backoff retry mechanism
"""

import asyncio
import time
from typing import Callable, Optional, Type, Tuple, Any, List
from dataclasses import dataclass
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt"""
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±20%)
            import random
            jitter_amount = delay * 0.2 * (random.random() * 2 - 1)
            delay += jitter_amount
        
        return max(0, delay)


async def retry(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable] = None,
    **kwargs
) -> Any:
    """Retry function call with exponential backoff
    
    Args:
        func: Function to retry (can be async or sync)
        *args: Positional arguments
        config: Retry configuration
        on_retry: Callback called on each retry
        **kwargs: Keyword arguments
    
    Returns:
        Result of function call
    
    Raises:
        Last exception if all attempts fail
    """
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts:
                delay = config.get_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt}/{config.max_attempts} after {delay:.2f}s. "
                    f"Error: {str(e)}"
                )
                
                if on_retry:
                    try:
                        if asyncio.iscoroutinefunction(on_retry):
                            await on_retry(attempt, e, delay)
                        else:
                            on_retry(attempt, e, delay)
                    except Exception as retry_error:
                        logger.error(f"Error in on_retry callback: {retry_error}")
                
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} retry attempts failed")
        
        except Exception as e:
            # Non-retryable exception
            logger.error(f"Non-retryable exception: {e}")
            raise
    
    # All attempts failed
    if last_exception:
        raise last_exception
    raise Exception("Retry failed without exception")


def retry_sync(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable] = None,
    **kwargs
) -> Any:
    """Synchronous retry function call with exponential backoff"""
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return func(*args, **kwargs)
        
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts:
                delay = config.get_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt}/{config.max_attempts} after {delay:.2f}s. "
                    f"Error: {str(e)}"
                )
                
                if on_retry:
                    try:
                        on_retry(attempt, e, delay)
                    except Exception as retry_error:
                        logger.error(f"Error in on_retry callback: {retry_error}")
                
                time.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} retry attempts failed")
        
        except Exception as e:
            # Non-retryable exception
            logger.error(f"Non-retryable exception: {e}")
            raise
    
    # All attempts failed
    if last_exception:
        raise last_exception
    raise Exception("Retry failed without exception")






