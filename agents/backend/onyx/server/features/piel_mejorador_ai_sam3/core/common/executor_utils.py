"""
Executor and Scheduler Utilities for Piel Mejorador AI SAM3
==========================================================

Unified executor and scheduler pattern utilities.
"""

import asyncio
import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ScheduledTask:
    """Scheduled task definition."""
    task_id: str
    func: Callable[[], Any]
    schedule_time: datetime
    interval: Optional[timedelta] = None
    max_runs: Optional[int] = None
    run_count: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class Executor(ABC):
    """Base executor interface."""
    
    @abstractmethod
    async def execute(self, func: Callable[..., R], *args, **kwargs) -> R:
        """Execute function."""
        pass


class SimpleExecutor(Executor):
    """Simple executor that executes functions directly."""
    
    async def execute(self, func: Callable[..., R], *args, **kwargs) -> R:
        """Execute function."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)


class ThreadPoolExecutor(Executor):
    """Thread pool executor."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize thread pool executor.
        
        Args:
            max_workers: Maximum number of workers
        """
        from concurrent.futures import ThreadPoolExecutor as TPE
        self._executor = TPE(max_workers=max_workers)
        self.max_workers = max_workers
    
    async def execute(self, func: Callable[..., R], *args, **kwargs) -> R:
        """Execute function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor."""
        self._executor.shutdown(wait=wait)


class Scheduler:
    """Task scheduler."""
    
    def __init__(self):
        """Initialize scheduler."""
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    def schedule(
        self,
        task_id: str,
        func: Callable[[], Any],
        schedule_time: datetime,
        interval: Optional[timedelta] = None,
        max_runs: Optional[int] = None,
        **metadata
    ) -> ScheduledTask:
        """
        Schedule task.
        
        Args:
            task_id: Unique task ID
            func: Function to execute
            schedule_time: When to execute
            interval: Optional interval for recurring tasks
            max_runs: Optional maximum number of runs
            **metadata: Additional metadata
            
        Returns:
            ScheduledTask
        """
        task = ScheduledTask(
            task_id=task_id,
            func=func,
            schedule_time=schedule_time,
            interval=interval,
            max_runs=max_runs,
            metadata=metadata
        )
        self._tasks[task_id] = task
        logger.debug(f"Scheduled task {task_id} for {schedule_time}")
        return task
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel scheduled task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            logger.debug(f"Cancelled task {task_id}")
            return True
        return False
    
    async def start(self):
        """Start scheduler."""
        if self._running:
            return
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        """Stop scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def _scheduler_loop(self):
        """Scheduler main loop."""
        while self._running:
            try:
                now = datetime.now()
                tasks_to_run = []
                
                for task in list(self._tasks.values()):
                    if not task.enabled:
                        continue
                    
                    if task.max_runs and task.run_count >= task.max_runs:
                        continue
                    
                    if now >= task.schedule_time:
                        tasks_to_run.append(task)
                
                # Execute tasks
                for task in tasks_to_run:
                    try:
                        if asyncio.iscoroutinefunction(task.func):
                            await task.func()
                        else:
                            task.func()
                        
                        task.run_count += 1
                        
                        # Reschedule if interval provided
                        if task.interval:
                            task.schedule_time = now + task.interval
                        else:
                            # One-time task, remove it
                            if task.task_id in self._tasks:
                                del self._tasks[task.task_id]
                    
                    except Exception as e:
                        logger.error(f"Error executing scheduled task {task.task_id}: {e}")
                
                await asyncio.sleep(1.0)  # Check every second
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(1.0)
    
    def list_tasks(self) -> List[ScheduledTask]:
        """List all scheduled tasks."""
        return list(self._tasks.values())
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get scheduled task."""
        return self._tasks.get(task_id)


class ExecutorUtils:
    """Unified executor utilities."""
    
    @staticmethod
    def create_executor() -> SimpleExecutor:
        """
        Create simple executor.
        
        Returns:
            SimpleExecutor
        """
        return SimpleExecutor()
    
    @staticmethod
    def create_thread_pool_executor(max_workers: int = 4) -> ThreadPoolExecutor:
        """
        Create thread pool executor.
        
        Args:
            max_workers: Maximum number of workers
            
        Returns:
            ThreadPoolExecutor
        """
        return ThreadPoolExecutor(max_workers)
    
    @staticmethod
    def create_scheduler() -> Scheduler:
        """
        Create scheduler.
        
        Returns:
            Scheduler
        """
        return Scheduler()


# Convenience functions
def create_executor() -> SimpleExecutor:
    """Create executor."""
    return ExecutorUtils.create_executor()


def create_thread_pool_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Create thread pool executor."""
    return ExecutorUtils.create_thread_pool_executor(max_workers)


def create_scheduler() -> Scheduler:
    """Create scheduler."""
    return ExecutorUtils.create_scheduler()




