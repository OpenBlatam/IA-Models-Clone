"""
Advanced Retry System
=====================

Advanced retry system with multiple strategies and exponential backoff.
"""

import asyncio
import logging
import random
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

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
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    on_retry: Optional[Callable] = None


@dataclass
class RetryResult:
    """Retry result."""
    success: bool
    attempts: int
    total_duration: float
    last_error: Optional[Exception] = None
    attempts_history: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedRetryManager:
    """Advanced retry manager with multiple strategies."""
    
    def __init__(self, default_config: Optional[RetryConfig] = None):
        """
        Initialize advanced retry manager.
        
        Args:
            default_config: Optional default retry configuration
        """
        self.default_config = default_config or RetryConfig()
        self.retry_history: List[RetryResult] = []
    
    async def execute_with_retry(
        self,
        func: Callable,
        config: Optional[RetryConfig] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            config: Optional retry configuration
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        retry_config = config or self.default_config
        start_time = datetime.now()
        attempts = 0
        attempts_history = []
        last_error = None
        
        while attempts < retry_config.max_attempts:
            attempts += 1
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success
                total_duration = (datetime.now() - start_time).total_seconds()
                retry_result = RetryResult(
                    success=True,
                    attempts=attempts,
                    total_duration=total_duration,
                    attempts_history=attempts_history
                )
                self.retry_history.append(retry_result)
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Check if exception is retryable
                if retry_config.retryable_exceptions:
                    if not any(isinstance(e, exc_type) for exc_type in retry_config.retryable_exceptions):
                        # Not retryable, raise immediately
                        raise
                
                # Check if we have more attempts
                if attempts >= retry_config.max_attempts:
                    # No more attempts
                    total_duration = (datetime.now() - start_time).total_seconds()
                    retry_result = RetryResult(
                        success=False,
                        attempts=attempts,
                        total_duration=total_duration,
                        last_error=last_error,
                        attempts_history=attempts_history
                    )
                    self.retry_history.append(retry_result)
                    raise
                
                # Calculate delay
                delay = self._calculate_delay(attempts, retry_config)
                
                # Record attempt
                attempts_history.append({
                    "attempt": attempts,
                    "error": str(e),
                    "delay": delay,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Call on_retry callback
                if retry_config.on_retry:
                    try:
                        if asyncio.iscoroutinefunction(retry_config.on_retry):
                            await retry_config.on_retry(attempts, e, delay)
                        else:
                            retry_config.on_retry(attempts, e, delay)
                    except Exception as callback_error:
                        logger.warning(f"Retry callback failed: {callback_error}")
                
                logger.warning(f"Attempt {attempts} failed: {e}. Retrying in {delay:.2f}s...")
                
                # Wait before retry
                await asyncio.sleep(delay)
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt."""
        if config.strategy == RetryStrategy.FIXED:
            delay = config.initial_delay
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.initial_delay * attempt
        else:
            delay = config.initial_delay
        
        # Apply max delay
        delay = min(delay, config.max_delay)
        
        # Apply jitter
        if config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retry statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.retry_history:
            return {
                "total_retries": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0,
                "average_attempts": 0,
                "average_duration": 0
            }
        
        total = len(self.retry_history)
        successful = sum(1 for r in self.retry_history if r.success)
        failed = total - successful
        avg_attempts = sum(r.attempts for r in self.retry_history) / total
        avg_duration = sum(r.total_duration for r in self.retry_history) / total
        
        return {
            "total_retries": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "average_attempts": avg_attempts,
            "average_duration": avg_duration
        }
    
    def clear_history(self):
        """Clear retry history."""
        self.retry_history.clear()



