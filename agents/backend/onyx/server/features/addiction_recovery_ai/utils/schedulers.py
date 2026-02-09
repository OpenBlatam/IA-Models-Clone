"""
Scheduler utilities
Task scheduling and execution
"""

from typing import Callable, Optional, Any, TypeVar
from datetime import datetime
import asyncio

try:
    from utils.date_helpers import get_current_utc
except ImportError:
    from ..date_helpers import get_current_utc

T = TypeVar('T')


class Scheduler:
    """
    Scheduler for task execution
    """
    
    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False
    
    async def schedule_once(
        self,
        func: Callable,
        delay: float,
        task_id: Optional[str] = None
    ) -> str:
        """
        Schedule function to run once after delay
        
        Args:
            func: Function to execute
            delay: Delay in seconds
            task_id: Optional task ID
        
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{datetime.now().timestamp()}"
        
        async def delayed_execution():
            await asyncio.sleep(delay)
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
        
        task = asyncio.create_task(delayed_execution())
        self._tasks[task_id] = task
        
        return task_id
    
    async def schedule_recurring(
        self,
        func: Callable,
        interval: float,
        task_id: Optional[str] = None,
        count: Optional[int] = None
    ) -> str:
        """
        Schedule function to run repeatedly
        
        Args:
            func: Function to execute
            interval: Interval in seconds
            task_id: Optional task ID
            count: Optional number of times to run
        
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"recurring_{datetime.now().timestamp()}"
        
        async def recurring_execution():
            iterations = 0
            while count is None or iterations < count:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
                
                iterations += 1
                if count is None or iterations < count:
                    await asyncio.sleep(interval)
        
        task = asyncio.create_task(recurring_execution())
        self._tasks[task_id] = task
        
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel scheduled task
        
        Args:
            task_id: Task ID to cancel
        
        Returns:
            True if cancelled, False if not found
        """
        if task_id in self._tasks:
            self._tasks[task_id].cancel()
            del self._tasks[task_id]
            return True
        return False
    
    def cancel_all(self) -> None:
        """Cancel all scheduled tasks"""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()


def create_scheduler() -> Scheduler:
    """Create new scheduler"""
    return Scheduler()

