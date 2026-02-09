"""Scheduler utilities."""

from typing import Callable, Optional, Any
import asyncio
from datetime import datetime, timedelta
from threading import Timer
import time


class Scheduler:
    """Simple scheduler."""
    
    def __init__(self):
        self._tasks: list = []
        self._running = False
    
    def schedule(self, func: Callable, delay: float, *args, **kwargs) -> None:
        """
        Schedule function to run after delay.
        
        Args:
            func: Function to run
            delay: Delay in seconds
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        def run():
            time.sleep(delay)
            func(*args, **kwargs)
        
        thread = Timer(delay, func, args, kwargs)
        thread.start()
        self._tasks.append(thread)
    
    def schedule_at(self, func: Callable, when: datetime, *args, **kwargs) -> None:
        """
        Schedule function to run at specific time.
        
        Args:
            func: Function to run
            when: When to run
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        now = datetime.now()
        if when < now:
            return
        
        delay = (when - now).total_seconds()
        self.schedule(func, delay, *args, **kwargs)
    
    def schedule_recurring(
        self,
        func: Callable,
        interval: float,
        *args,
        **kwargs
    ) -> None:
        """
        Schedule recurring function.
        
        Args:
            func: Function to run
            interval: Interval in seconds
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        def wrapper():
            func(*args, **kwargs)
            self.schedule_recurring(func, interval, *args, **kwargs)
        
        self.schedule(wrapper, interval)


class AsyncScheduler:
    """Async scheduler."""
    
    def __init__(self):
        self._tasks: list = []
        self._running = False
    
    async def schedule(
        self,
        func: Callable,
        delay: float,
        *args,
        **kwargs
    ) -> None:
        """
        Schedule async function to run after delay.
        
        Args:
            func: Function to run
            delay: Delay in seconds
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        await asyncio.sleep(delay)
        if asyncio.iscoroutinefunction(func):
            await func(*args, **kwargs)
        else:
            func(*args, **kwargs)
    
    async def schedule_at(
        self,
        func: Callable,
        when: datetime,
        *args,
        **kwargs
    ) -> None:
        """
        Schedule async function to run at specific time.
        
        Args:
            func: Function to run
            when: When to run
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        now = datetime.now()
        if when < now:
            return
        
        delay = (when - now).total_seconds()
        await self.schedule(func, delay, *args, **kwargs)
    
    async def schedule_recurring(
        self,
        func: Callable,
        interval: float,
        *args,
        **kwargs
    ) -> None:
        """
        Schedule recurring async function.
        
        Args:
            func: Function to run
            interval: Interval in seconds
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        while True:
            await asyncio.sleep(interval)
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
    
    def start_recurring(
        self,
        func: Callable,
        interval: float,
        *args,
        **kwargs
    ) -> asyncio.Task:
        """
        Start recurring task.
        
        Returns:
            Task
        """
        task = asyncio.create_task(
            self.schedule_recurring(func, interval, *args, **kwargs)
        )
        self._tasks.append(task)
        return task
    
    def cancel_all(self) -> None:
        """Cancel all tasks."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()



