"""
Async Helpers for Document Analyzer
====================================

Advanced async utilities for concurrent processing, throttling, and resource management.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, TypeVar, Awaitable
from datetime import datetime
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')

class AsyncPool:
    """Async worker pool for concurrent processing"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.active_workers = 0
        logger.info(f"AsyncPool initialized with {max_workers} workers")
    
    async def execute(self, coro: Awaitable[T]) -> T:
        """Execute coroutine with pool limits"""
        async with self.semaphore:
            self.active_workers += 1
            try:
                return await coro
            finally:
                self.active_workers -= 1
    
    async def map(self, items: List[Any], func: Callable, *args, **kwargs) -> List[Any]:
        """Map function over items with concurrency control"""
        tasks = [self.execute(func(item, *args, **kwargs)) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

class AsyncThrottle:
    """Async throttling for rate limiting"""
    
    def __init__(self, rate: float, per: float = 1.0):
        """
        Initialize throttler
        
        Args:
            rate: Number of operations allowed
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()
        self.lock = asyncio.Lock()
        logger.info(f"AsyncThrottle initialized: {rate} ops per {per}s")
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        """Acquire tokens, returns True if allowed"""
        async with self.lock:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current
            
            self.allowance += time_passed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance >= tokens:
                self.allowance -= tokens
                return True
            return False
    
    async def wait(self, tokens: float = 1.0):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)

async def gather_with_concurrency(
    limit: int,
    *coros: Awaitable[T],
    return_exceptions: bool = False
) -> List[T]:
    """Gather coroutines with concurrency limit"""
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *[bounded_coro(coro) for coro in coros],
        return_exceptions=return_exceptions
    )

async def batch_process_async(
    items: List[Any],
    func: Callable,
    batch_size: int = 10,
    max_concurrent: int = 5,
    **kwargs
) -> List[Any]:
    """Process items in batches with concurrency control"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await gather_with_concurrency(
            max_concurrent,
            *[func(item, **kwargs) for item in batch],
            return_exceptions=True
        )
        results.extend(batch_results)
    
    return results

class AsyncRateLimiter:
    """Advanced async rate limiter"""
    
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.times: deque = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Try to acquire, returns True if allowed"""
        async with self.lock:
            now = time.monotonic()
            
            # Remove old entries
            while self.times and self.times[0] <= now - self.period:
                self.times.popleft()
            
            if len(self.times) < self.calls:
                self.times.append(now)
                return True
            return False
    
    async def wait(self):
        """Wait until rate limit allows"""
        while not await self.acquire():
            if self.times:
                sleep_time = self.period - (time.monotonic() - self.times[0])
                if sleep_time > 0:
                    await asyncio.sleep(min(sleep_time, 0.1))
            else:
                await asyncio.sleep(0.1)
















