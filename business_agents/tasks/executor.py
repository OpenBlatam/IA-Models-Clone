"""
Task Executor
=============

Task execution engine for background processing.
"""

import asyncio
import logging
import traceback
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import importlib

from .types import Task, TaskResult, TaskStatus, TaskCallback

logger = logging.getLogger(__name__)

class TaskExecutor:
    """Base task executor."""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.callbacks: List[TaskCallback] = []
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    def add_callback(self, callback: TaskCallback):
        """Add a task callback."""
        self.callbacks.append(callback)
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        start_time = datetime.now()
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = start_time
            
            # Execute the task
            result = await self._execute_function(task)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create successful result
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                retry_count=task.retry_count
            )
            
            # Update task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Execute callbacks
            await self._execute_callbacks(task, task_result, "completed")
            
            logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            return task_result
            
        except Exception as e:
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create failed result
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                retry_count=task.retry_count
            )
            
            # Update task
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
            
            # Execute callbacks
            await self._execute_callbacks(task, task_result, "failed")
            
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            return task_result
    
    async def _execute_function(self, task: Task) -> Any:
        """Execute the task function."""
        try:
            # Import the module and get the function
            module_name, function_name = task.function_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            
            # Execute the function
            if asyncio.iscoroutinefunction(function):
                result = await function(*task.args, **task.kwargs)
            else:
                result = function(*task.args, **task.kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Function execution failed for task {task.task_id}: {str(e)}")
            raise
    
    async def _execute_callbacks(self, task: Task, result: TaskResult, event: str):
        """Execute registered callbacks."""
        for callback in self.callbacks:
            if callback.event == event:
                try:
                    await callback.execute(task, result)
                except Exception as e:
                    logger.error(f"Callback execution failed: {str(e)}")

class AsyncTaskExecutor(TaskExecutor):
    """Asynchronous task executor with concurrency control."""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        super().__init__(max_concurrent_tasks)
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
    
    async def start(self, task_queue):
        """Start the task executor."""
        if self._running:
            return
        
        self._running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent_tasks):
            worker_task = asyncio.create_task(self._worker(task_queue, i))
            self._worker_tasks.append(worker_task)
        
        logger.info(f"Started task executor with {self.max_concurrent_tasks} workers")
    
    async def stop(self):
        """Stop the task executor."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all worker tasks
        for worker_task in self._worker_tasks:
            worker_task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Cancel any running tasks
        for task_id, running_task in self.running_tasks.items():
            running_task.cancel()
        
        self._worker_tasks.clear()
        self.running_tasks.clear()
        
        logger.info("Task executor stopped")
    
    async def _worker(self, task_queue, worker_id: int):
        """Worker coroutine for processing tasks."""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get next task from queue
                task = await task_queue.dequeue()
                
                if task is None:
                    # No tasks available, wait a bit
                    await asyncio.sleep(1)
                    continue
                
                # Acquire semaphore
                async with self._semaphore:
                    # Execute task
                    execution_task = asyncio.create_task(self.execute_task(task))
                    self.running_tasks[task.task_id] = execution_task
                    
                    try:
                        await execution_task
                    finally:
                        # Remove from running tasks
                        self.running_tasks.pop(task.task_id, None)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def submit_task(self, task: Task, task_queue) -> str:
        """Submit a task for execution."""
        try:
            # Add task to queue
            success = await task_queue.enqueue(task)
            if not success:
                raise Exception("Failed to enqueue task")
            
            logger.info(f"Submitted task: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {str(e)}")
            raise
    
    def get_running_tasks(self) -> List[str]:
        """Get list of currently running task IDs."""
        return list(self.running_tasks.keys())
    
    def get_executor_status(self) -> Dict[str, Any]:
        """Get executor status information."""
        return {
            "running": self._running,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "active_workers": len(self._worker_tasks),
            "running_tasks": len(self.running_tasks),
            "running_task_ids": list(self.running_tasks.keys())
        }

class TaskRetryHandler:
    """Handles task retry logic."""
    
    @staticmethod
    async def should_retry(task: Task, error: Exception) -> bool:
        """Determine if a task should be retried."""
        if task.retry_count >= task.max_retries:
            return False
        
        # Check if error is retryable
        retryable_errors = [
            "timeout",
            "connection",
            "temporary",
            "rate_limit"
        ]
        
        error_str = str(error).lower()
        return any(retryable_error in error_str for retryable_error in retryable_errors)
    
    @staticmethod
    async def get_retry_delay(task: Task) -> int:
        """Get retry delay with exponential backoff."""
        base_delay = task.retry_delay
        exponential_delay = base_delay * (2 ** task.retry_count)
        return min(exponential_delay, 3600)  # Max 1 hour
    
    @staticmethod
    async def prepare_retry_task(task: Task) -> Task:
        """Prepare task for retry."""
        task.retry_count += 1
        task.status = TaskStatus.RETRYING
        task.started_at = None
        task.completed_at = None
        task.error = None
        task.result = None
        
        return task
