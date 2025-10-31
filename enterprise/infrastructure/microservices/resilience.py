from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from enum import Enum
import random
from typing import Any, List, Dict, Optional
"""
Resilience Patterns Implementation
=================================

Advanced resilience patterns for microservices:
- Circuit Breaker
- Bulkhead
- Retry with exponential backoff
- Timeout policies
- Rate limiting
"""


logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    RANDOM = "random"


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.initial_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay * (1 + attempt)
        else:  # RANDOM
            delay = random.uniform(self.initial_delay, self.max_delay)
        
        # Apply jitter if enabled
        if self.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return min(delay, self.max_delay)


@dataclass
class TimeoutPolicy:
    """Timeout policy configuration."""
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    total_timeout: float = 60.0


class BulkheadPattern:
    """Bulkhead pattern implementation for resource isolation."""
    
    def __init__(self, max_concurrent: int = 10):
        
    """__init__ function."""
self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.total_requests = 0
        self.rejected_requests = 0
        
    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with bulkhead protection."""
        self.total_requests += 1
        
        try:
            # Try to acquire semaphore (non-blocking)
            if self.semaphore.locked():
                self.rejected_requests += 1
                raise Exception("Bulkhead capacity exceeded")
            
            async with self.semaphore:
                self.active_requests += 1
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                finally:
                    self.active_requests -= 1
                    
        except Exception as e:
            logger.warning(f"Bulkhead execution failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "max_concurrent": self.max_concurrent,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": self.rejected_requests / max(1, self.total_requests),
            "available_capacity": self.max_concurrent - self.active_requests
        }


class ResilienceManager:
    """Comprehensive resilience manager combining multiple patterns."""
    
    def __init__(self) -> Any:
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.timeout_policies: Dict[str, TimeoutPolicy] = {}
        self.bulkheads: Dict[str, BulkheadPattern] = {}
        
    def add_retry_policy(self, name: str, policy: RetryPolicy):
        """Add a retry policy."""
        self.retry_policies[name] = policy
        logger.info(f"Added retry policy: {name}")
    
    def add_timeout_policy(self, name: str, policy: TimeoutPolicy):
        """Add a timeout policy."""
        self.timeout_policies[name] = policy
        logger.info(f"Added timeout policy: {name}")
    
    def add_bulkhead(self, name: str, bulkhead: BulkheadPattern):
        """Add a bulkhead."""
        self.bulkheads[name] = bulkhead
        logger.info(f"Added bulkhead: {name}")
    
    async def execute_with_resilience(self,
                                    func: Callable,
                                    *args,
                                    retry_policy: str = None,
                                    timeout_policy: str = None,
                                    bulkhead: str = None,
                                    **kwargs):
        """Execute function with applied resilience patterns."""
        
        # Apply bulkhead if specified
        if bulkhead and bulkhead in self.bulkheads:
            return await self.bulkheads[bulkhead].execute(
                self._execute_with_retry_and_timeout,
                func, args, kwargs, retry_policy, timeout_policy
            )
        else:
            return await self._execute_with_retry_and_timeout(
                func, args, kwargs, retry_policy, timeout_policy
            )
    
    async def _execute_with_retry_and_timeout(self,
                                            func: Callable,
                                            args: tuple,
                                            kwargs: dict,
                                            retry_policy: str = None,
                                            timeout_policy: str = None):
        """Execute function with retry and timeout."""
        
        # Get policies
        retry_pol = self.retry_policies.get(retry_policy) if retry_policy else None
        timeout_pol = self.timeout_policies.get(timeout_policy) if timeout_policy else None
        
        last_exception = None
        attempts = 0
        max_attempts = (retry_pol.max_retries + 1) if retry_pol else 1
        
        while attempts < max_attempts:
            try:
                # Apply timeout if specified
                if timeout_pol:
                    return await asyncio.wait_for(
                        self._execute_function(func, args, kwargs),
                        timeout=timeout_pol.total_timeout
                    )
                else:
                    return await self._execute_function(func, args, kwargs)
                    
            except Exception as e:
                last_exception = e
                attempts += 1
                
                # Check if we should retry
                if attempts >= max_attempts or not retry_pol:
                    break
                
                # Calculate delay and wait
                delay = retry_pol.calculate_delay(attempts - 1)
                logger.warning(f"Attempt {attempts} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        # All attempts failed
        logger.error(f"All {attempts} attempts failed")
        raise last_exception
    
    async def _execute_function(self, func: Callable, args: tuple, kwargs: dict):
        """Execute function (async or sync)."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resilience manager statistics."""
        return {
            "retry_policies": list(self.retry_policies.keys()),
            "timeout_policies": list(self.timeout_policies.keys()),
            "bulkheads": {
                name: bulkhead.get_stats()
                for name, bulkhead in self.bulkheads.items()
            }
        } 