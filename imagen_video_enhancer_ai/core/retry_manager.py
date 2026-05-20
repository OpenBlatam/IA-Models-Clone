"""
Retry Manager for Imagen Video Enhancer AI
===========================================

Automatic retry system for failed tasks.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    retryable_errors: list = None
    
    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = [
                "timeout",
                "connection",
                "rate_limit",
                "temporary"
            ]


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    timestamp: datetime
    error: str
    delay: float
    success: bool = False


class RetryManager:
    """
    Manages automatic retries for failed tasks.
    
    Features:
    - Configurable retry strategies
    - Error classification
    - Retry history tracking
    - Automatic retry scheduling
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self._retry_history: Dict[str, List[RetryAttempt]] = {}
        self._stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0,
        }
    
    def should_retry(
        self,
        task_id: str,
        error: Exception,
        attempt_number: int
    ) -> bool:
        """
        Determine if a task should be retried.
        
        Args:
            task_id: Task identifier
            error: Exception that occurred
            attempt_number: Current attempt number
            
        Returns:
            True if should retry, False otherwise
        """
        if attempt_number >= self.config.max_retries:
            return False
        
        # Check if error is retryable
        error_str = str(error).lower()
        is_retryable = any(
            retryable in error_str
            for retryable in self.config.retryable_errors
        )
        
        if not is_retryable:
            # Check error type
            error_type = type(error).__name__.lower()
            is_retryable = any(
                retryable in error_type
                for retryable in self.config.retryable_errors
            )
        
        return is_retryable
    
    def calculate_delay(self, attempt_number: int) -> float:
        """
        Calculate delay before next retry.
        
        Args:
            attempt_number: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            return self.config.initial_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay * (self.config.multiplier ** attempt_number)
            return min(delay, self.config.max_delay)
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * (attempt_number + 1)
            return min(delay, self.config.max_delay)
        
        return self.config.initial_delay
    
    async def retry_task(
        self,
        task_id: str,
        task_func: Callable,
        error: Exception,
        attempt_number: int
    ) -> Any:
        """
        Retry a failed task.
        
        Args:
            task_id: Task identifier
            task_func: Function to retry (async callable)
            error: Original error
            attempt_number: Current attempt number
            
        Returns:
            Task result if successful
            
        Raises:
            Exception if retry fails
        """
        if not self.should_retry(task_id, error, attempt_number):
            raise error
        
        delay = self.calculate_delay(attempt_number)
        
        if delay > 0:
            logger.info(f"Retrying task {task_id} (attempt {attempt_number + 1}/{self.config.max_retries}) after {delay}s")
            await asyncio.sleep(delay)
        
        # Record retry attempt
        attempt = RetryAttempt(
            attempt_number=attempt_number + 1,
            timestamp=datetime.now(),
            error=str(error),
            delay=delay
        )
        
        if task_id not in self._retry_history:
            self._retry_history[task_id] = []
        self._retry_history[task_id].append(attempt)
        
        self._stats["total_retries"] += 1
        
        try:
            # Execute retry - handle both sync and async functions
            if asyncio.iscoroutinefunction(task_func):
                result = await task_func()
            else:
                result = task_func()
            
            attempt.success = True
            self._stats["successful_retries"] += 1
            logger.info(f"Task {task_id} retry successful")
            return result
        except Exception as e:
            attempt.success = False
            self._stats["failed_retries"] += 1
            logger.warning(f"Task {task_id} retry failed: {e}")
            raise
    
    def get_retry_history(self, task_id: str) -> List[RetryAttempt]:
        """Get retry history for a task."""
        return self._retry_history.get(task_id, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        success_rate = (
            self._stats["successful_retries"] / self._stats["total_retries"]
            if self._stats["total_retries"] > 0
            else 0.0
        )
        
        return {
            **self._stats,
            "success_rate": success_rate,
            "tasks_with_retries": len(self._retry_history),
        }

