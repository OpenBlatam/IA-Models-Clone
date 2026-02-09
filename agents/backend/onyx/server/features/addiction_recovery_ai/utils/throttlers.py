"""
Throttling utilities
Rate throttling patterns
"""

from typing import Callable, TypeVar, Optional
from collections import deque
import time
import asyncio

T = TypeVar('T')
U = TypeVar('U')


class Throttler:
    """
    Throttler for rate limiting
    """
    
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
    
    async def acquire(self) -> None:
        """Acquire throttle permission"""
        now = time.time()
        
        # Remove old calls
        while self.calls and self.calls[0] < now - self.period:
            self.calls.popleft()
        
        # Wait if at limit
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                # Remove expired calls after sleep
                while self.calls and self.calls[0] < time.time() - self.period:
                    self.calls.popleft()
        
        # Record this call
        self.calls.append(time.time())
    
    def reset(self) -> None:
        """Reset throttler"""
        self.calls.clear()


def create_throttler(max_calls: int, period: float) -> Throttler:
    """Create new throttler"""
    return Throttler(max_calls, period)


async def with_throttle(
    throttler: Throttler,
    func: Callable[[], U]
) -> U:
    """
    Execute function with throttle
    
    Args:
        throttler: Throttler instance
        func: Function to execute
    
    Returns:
        Function result
    """
    await throttler.acquire()
    
    if asyncio.iscoroutinefunction(func):
        return await func()
    return func()


class Debouncer:
    """
    Debouncer for delaying function calls
    """
    
    def __init__(self, delay: float):
        self.delay = delay
        self.last_call_time = 0.0
        self.pending_task: Optional[asyncio.Task] = None
    
    async def debounce(self, func: Callable[[], U]) -> Optional[U]:
        """
        Debounce function call
        
        Args:
            func: Function to debounce
        
        Returns:
            Function result or None if debounced
        """
        now = time.time()
        self.last_call_time = now
        
        # Cancel pending task if exists
        if self.pending_task and not self.pending_task.done():
            self.pending_task.cancel()
        
        async def delayed_call():
            await asyncio.sleep(self.delay)
            
            # Check if still valid
            if time.time() - self.last_call_time >= self.delay:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                return func()
            return None
        
        self.pending_task = asyncio.create_task(delayed_call())
        return await self.pending_task


def create_debouncer(delay: float) -> Debouncer:
    """Create new debouncer"""
    return Debouncer(delay)

