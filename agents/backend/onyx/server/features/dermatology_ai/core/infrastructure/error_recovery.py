"""
Error Recovery Strategies
Provides strategies for handling and recovering from errors
"""

from typing import Callable, Any, Optional, List
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    FAIL = "fail"


class ErrorRecovery:
    """Error recovery handler"""
    
    def __init__(
        self,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        max_attempts: int = 3,
        fallback_func: Optional[Callable] = None
    ):
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.fallback_func = fallback_func
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with error recovery
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        if self.strategy == RecoveryStrategy.RETRY:
            return await self._execute_with_retry(func, *args, **kwargs)
        elif self.strategy == RecoveryStrategy.FALLBACK:
            return await self._execute_with_fallback(func, *args, **kwargs)
        elif self.strategy == RecoveryStrategy.SKIP:
            return await self._execute_with_skip(func, *args, **kwargs)
        else:
            return await func(*args, **kwargs)
    
    async def _execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts:
                    delay = 0.5 * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt}/{self.max_attempts} failed, "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_attempts} attempts failed")
        
        if last_exception:
            raise last_exception
    
    async def _execute_with_fallback(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with fallback function"""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed, using fallback: {e}")
            
            if self.fallback_func:
                try:
                    if asyncio.iscoroutinefunction(self.fallback_func):
                        return await self.fallback_func(*args, **kwargs)
                    return self.fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback function also failed: {fallback_error}")
                    raise fallback_error
            
            raise e
    
    async def _execute_with_skip(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with skip on error"""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Skipping operation due to error: {e}")
            return None


def with_error_recovery(
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    max_attempts: int = 3,
    fallback_func: Optional[Callable] = None
):
    """
    Decorator for error recovery
    
    Args:
        strategy: Recovery strategy to use
        max_attempts: Maximum retry attempts
        fallback_func: Fallback function
    """
    def decorator(func: Callable) -> Callable:
        recovery = ErrorRecovery(
            strategy=strategy,
            max_attempts=max_attempts,
            fallback_func=fallback_func
        )
        
        async def async_wrapper(*args, **kwargs):
            return await recovery.execute(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(recovery.execute(func, *args, **kwargs))
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator















