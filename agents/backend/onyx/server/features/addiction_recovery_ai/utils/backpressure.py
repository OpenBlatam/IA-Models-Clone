"""
Backpressure utilities
Control flow and backpressure patterns
"""

from typing import TypeVar, Callable, Optional
from asyncio import Semaphore, Event
import asyncio

T = TypeVar('T')


class BackpressureController:
    """
    Backpressure controller for flow control
    """
    
    def __init__(self, max_pending: int = 10):
        self.max_pending = max_pending
        self.semaphore = Semaphore(max_pending)
        self.pause_event = Event()
        self.pause_event.set()  # Start unpaused
    
    async def acquire(self) -> None:
        """Acquire backpressure control"""
        await self.pause_event.wait()
        await self.semaphore.acquire()
    
    def release(self) -> None:
        """Release backpressure control"""
        self.semaphore.release()
    
    async def pause(self) -> None:
        """Pause processing"""
        self.pause_event.clear()
    
    async def resume(self) -> None:
        """Resume processing"""
        self.pause_event.set()
    
    def pending_count(self) -> int:
        """Get pending count"""
        return self.max_pending - self.semaphore._value
    
    async def wait_for_capacity(self, min_capacity: int = 1) -> None:
        """Wait until minimum capacity available"""
        while self.pending_count() >= (self.max_pending - min_capacity + 1):
            await asyncio.sleep(0.1)


def create_backpressure_controller(max_pending: int = 10) -> BackpressureController:
    """Create new backpressure controller"""
    return BackpressureController(max_pending)


async def with_backpressure(
    controller: BackpressureController,
    func: Callable[[], T]
) -> T:
    """
    Execute function with backpressure control
    
    Args:
        controller: Backpressure controller
        func: Function to execute
    
    Returns:
        Function result
    """
    await controller.acquire()
    try:
        if asyncio.iscoroutinefunction(func):
            return await func()
        return func()
    finally:
        controller.release()

