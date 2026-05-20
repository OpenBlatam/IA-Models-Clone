"""
Parallel Executor
=================

Manages parallel execution of tasks with worker pool and queue management.
Enhanced with dynamic scaling and auto-healing for 24/7 operation.

Refactored with:
- TaskExecutor integration
- WorkerPool pattern
"""

import asyncio
import logging
from typing import Callable, Any, Dict, Optional, List, Set
from datetime import datetime
import traceback

from .task_executor import TaskExecutor

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    Executes tasks in parallel with configurable worker pool.
    
    Features:
    - Worker pool management
    - Task queue
    - Concurrent execution
    - Error handling
    - Resource management
    - Dynamic scaling (add/remove workers)
    - Auto-healing (respawn dead workers)
    """
    
    def __init__(self, max_workers: int = 10, min_workers: int = 2):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            min_workers: Minimum number of workers to maintain
        """
        self.max_workers = max_workers
        self.min_workers = min_workers
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._workers: Dict[str, asyncio.Task] = {}
        self._worker_counter = 0
        self._running = False
        self._draining = False  # Graceful shutdown mode
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_workers": 0,
            "workers_spawned": 0,
            "workers_respawned": 0,
        }
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized ParallelExecutor with {max_workers} max workers")
    
    async def start(self):
        """Start the parallel executor and worker pool."""
        if self._running:
            logger.warning("ParallelExecutor is already running")
            return
        
        self._running = True
        self._draining = False
        
        # Start initial worker tasks
        for i in range(self.min_workers):
            await self._spawn_worker()
        
        # Start health check loop for auto-healing
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Started {len(self._workers)} workers (min={self.min_workers}, max={self.max_workers})")
    
    async def stop(self, graceful: bool = True, timeout: float = 30.0):
        """
        Stop the parallel executor and all workers.
        
        Args:
            graceful: If True, wait for pending tasks to complete
            timeout: Maximum seconds to wait for graceful shutdown
        """
        if not self._running:
            return
        
        logger.info("Stopping ParallelExecutor...")
        self._draining = True  # Stop accepting new tasks
        
        if graceful:
            # Wait for queue to empty with timeout
            try:
                await asyncio.wait_for(self._task_queue.join(), timeout=timeout)
                logger.info("All pending tasks completed")
            except asyncio.TimeoutError:
                logger.warning(f"Graceful shutdown timed out after {timeout}s")
        
        self._running = False
        
        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all workers
        for worker_id, worker in list(self._workers.items()):
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers.values(), return_exceptions=True)
        
        self._workers.clear()
        logger.info("ParallelExecutor stopped")
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> asyncio.Task:
        """
        Submit a task for parallel execution.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Task handle
        """
        if not self._running:
            raise RuntimeError("ParallelExecutor is not running")
        
        task_data = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "created_at": datetime.now(),
        }
        
        await self._task_queue.put(task_data)
        
        async with self._lock:
            self._stats["total_tasks"] += 1
        
        logger.debug(f"Submitted task: {func.__name__}")
        
        # Return a future that will be resolved when task completes
        future = asyncio.Future()
        task_data["future"] = future
        return future
    
    async def _worker(self, worker_id: str):
        """
        Worker coroutine that processes tasks from the queue.
        
        Args:
            worker_id: Unique worker identifier
        """
        logger.info(f"Worker {worker_id} started")
        
        async with self._lock:
            self._stats["active_workers"] += 1
        
        try:
            while self._running:
                try:
                    # Get task from queue with timeout
                    task_data = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0
                    )
                    
                    logger.debug(f"Worker {worker_id} processing task: {task_data['func'].__name__}")
                    
                    try:
                        # Execute task using TaskExecutor
                        await TaskExecutor.execute_task(
                            func=task_data["func"],
                            args=task_data["args"],
                            kwargs=task_data["kwargs"],
                            task_queue=self._task_queue,
                            stats=self._stats,
                            lock=self._lock,
                            future=task_data.get("future")
                        )
                        
                        logger.debug(f"Worker {worker_id} completed task: {task_data['func'].__name__}")
                        
                    except Exception as e:
                        # TaskExecutor handles stats and future rejection, but we log here too
                        logger.error(
                            f"Worker {worker_id} failed task {task_data['func'].__name__}: {e}",
                            exc_info=True
                        )
                
                except asyncio.TimeoutError:
                    # Timeout waiting for task, continue loop
                    continue
                
                except asyncio.CancelledError:
                    logger.info(f"Worker {worker_id} cancelled")
                    break
                
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                    await asyncio.sleep(1.0)
        
        finally:
            async with self._lock:
                self._stats["active_workers"] -= 1
            logger.info(f"Worker {worker_id} stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        async def _get_stats():
            async with self._lock:
                return {
                    **self._stats,
                    "queue_size": self._task_queue.qsize(),
                    "running": self._running,
                }
        
        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, return current stats (may be slightly stale)
                return {
                    **self._stats,
                    "queue_size": self._task_queue.qsize(),
                    "running": self._running,
                    "max_workers": self.max_workers,
                    "min_workers": self.min_workers,
                    "draining": self._draining,
                }
            else:
                return loop.run_until_complete(_get_stats())
        except RuntimeError:
            return {
                **self._stats,
                "queue_size": self._task_queue.qsize(),
                "running": self._running,
                "max_workers": self.max_workers,
                "min_workers": self.min_workers,
                "draining": self._draining,
            }
    
    async def _spawn_worker(self) -> str:
        """
        Spawn a new worker.
        
        Returns:
            Worker ID
        """
        async with self._lock:
            self._worker_counter += 1
            worker_id = f"worker-{self._worker_counter}"
            worker = asyncio.create_task(self._worker(worker_id))
            self._workers[worker_id] = worker
            self._stats["workers_spawned"] += 1
        
        logger.debug(f"Spawned worker {worker_id}")
        return worker_id
    
    async def add_workers(self, count: int = 1) -> List[str]:
        """
        Add workers to the pool.
        
        Args:
            count: Number of workers to add
            
        Returns:
            List of new worker IDs
        """
        if not self._running:
            raise RuntimeError("ParallelExecutor is not running")
        
        current = len(self._workers)
        can_add = min(count, self.max_workers - current)
        
        if can_add <= 0:
            logger.warning(f"Cannot add workers: already at max ({self.max_workers})")
            return []
        
        new_workers = []
        for _ in range(can_add):
            worker_id = await self._spawn_worker()
            new_workers.append(worker_id)
        
        logger.info(f"Added {len(new_workers)} workers (total: {len(self._workers)})")
        return new_workers
    
    async def remove_workers(self, count: int = 1) -> int:
        """
        Remove workers from the pool.
        
        Args:
            count: Number of workers to remove
            
        Returns:
            Number of workers actually removed
        """
        if not self._running:
            return 0
        
        current = len(self._workers)
        can_remove = min(count, current - self.min_workers)
        
        if can_remove <= 0:
            logger.warning(f"Cannot remove workers: already at min ({self.min_workers})")
            return 0
        
        removed = 0
        async with self._lock:
            workers_to_remove = list(self._workers.items())[:can_remove]
            
            for worker_id, worker in workers_to_remove:
                worker.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(worker), timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                del self._workers[worker_id]
                removed += 1
        
        logger.info(f"Removed {removed} workers (total: {len(self._workers)})")
        return removed
    
    async def _health_check_loop(self):
        """Monitor worker health and respawn dead workers."""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if self._draining:
                    continue
                
                # Check for dead workers
                async with self._lock:
                    dead_workers = [
                        worker_id for worker_id, worker in self._workers.items()
                        if worker.done()
                    ]
                    
                    for worker_id in dead_workers:
                        # Check if it was an unexpected death
                        worker = self._workers[worker_id]
                        if worker.cancelled():
                            continue  # Expected cancellation
                        
                        exception = worker.exception() if not worker.cancelled() else None
                        if exception:
                            logger.warning(
                                f"Worker {worker_id} died unexpectedly: {exception}"
                            )
                        
                        # Remove dead worker
                        del self._workers[worker_id]
                
                # Respawn if below minimum
                current = len(self._workers)
                if current < self.min_workers:
                    to_spawn = self.min_workers - current
                    for _ in range(to_spawn):
                        await self._spawn_worker()
                        async with self._lock:
                            self._stats["workers_respawned"] += 1
                    
                    logger.info(f"Respawned {to_spawn} workers (auto-healing)")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}", exc_info=True)
                await asyncio.sleep(10)
