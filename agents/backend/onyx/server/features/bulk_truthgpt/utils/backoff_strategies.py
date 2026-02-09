"""
Backoff Strategies
===================

Advanced backoff strategies for retries.
"""

import asyncio
import time
import random
import logging
from typing import Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class BackoffStrategy(str, Enum):
    """Backoff strategies."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    POLYNOMIAL = "polynomial"
    CUSTOM = "custom"

class AdvancedBackoff:
    """Advanced backoff with multiple strategies."""
    
    def __init__(
        self,
        strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
        jitter_range: float = 0.1
    ):
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.jitter_range = jitter_range
        
        # Fibonacci sequence
        self.fib_cache = [1, 1]
    
    def get_delay(self, attempt: int) -> float:
        """Get delay for attempt number."""
        if attempt < 1:
            attempt = 1
        
        if self.strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.multiplier ** (attempt - 1))
        elif self.strategy == BackoffStrategy.FIBONACCI:
            delay = self.base_delay * self._get_fibonacci(attempt)
        elif self.strategy == BackoffStrategy.POLYNOMIAL:
            delay = self.base_delay * (attempt ** self.multiplier)
        else:
            delay = self.base_delay
        
        # Apply max delay
        delay = min(delay, self.max_delay)
        
        # Apply jitter
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            jitter_value = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter_value)
        
        return delay
    
    def _get_fibonacci(self, n: int) -> int:
        """Get Fibonacci number."""
        while len(self.fib_cache) <= n:
            self.fib_cache.append(
                self.fib_cache[-1] + self.fib_cache[-2]
            )
        return self.fib_cache[n]
    
    async def wait(self, attempt: int):
        """Wait for the calculated delay."""
        delay = self.get_delay(attempt)
        await asyncio.sleep(delay)
    
    def get_stats(self) -> dict:
        """Get backoff statistics."""
        return {
            "strategy": self.strategy.value,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "multiplier": self.multiplier,
            "jitter": self.jitter,
            "jitter_range": self.jitter_range
        }

# Global instances
exponential_backoff = AdvancedBackoff(
    strategy=BackoffStrategy.EXPONENTIAL,
    base_delay=1.0,
    max_delay=60.0,
    jitter=True
)

fibonacci_backoff = AdvancedBackoff(
    strategy=BackoffStrategy.FIBONACCI,
    base_delay=1.0,
    max_delay=60.0,
    jitter=True
)
































