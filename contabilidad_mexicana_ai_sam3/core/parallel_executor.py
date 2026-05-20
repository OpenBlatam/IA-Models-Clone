"""
Parallel Executor for Contabilidad Mexicana AI SAM3
===================================================

Refactored with:
- Integration with TaskExecutor
- WorkerPool pattern
- Graceful shutdown
"""

import asyncio
import logging
from typing import Callable, Any, Dict, Optional, List
from datetime import datetime

from .task_executor import TaskExecutor, ExecutionContext

logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages a pool of workers."""
    
    def __init__(self, size: int, queue: asyncio.Queue, stats: Dict[str, Any], lock: asyncio.Lock):
        self.size = size
        self.queue = queue
        self.stats = stats
        self.lock = lock
        self.workers: List[asyncio.Task] = []
        self.running = False
    
    async def start(self):
        """Start all workers."""
        if self.running:
            return
        self.running = True
        for i in range(self.size):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.workers.append(worker)
        logger.info(f"Started {self.size} workers")
    
    async def stop(self):
        """Stop all workers."""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.info("Stopped all workers")
    
    async def _worker_loop(self, worker_id: str):
        """Main worker loop."""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue
                task_data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                async with self.lock:
                    self.stats["active_workers"] += 1
                
                try:
                    # Execute task using TaskExecutor
                    await TaskExecutor.execute_task(
                        func=task_data["func"],
                        args=task_data["args"],
                        kwargs=task_data["kwargs"],
                        task_queue=self.queue,
                        stats=self.stats,
                        lock=self.lock,
                        future=task_data.get("future")
                    )
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                finally:
                    async with self.lock:
                        self.stats["active_workers"] -= 1
            
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} loop error: {e}")
                await asyncio.sleep(1.0)


class ParallelExecutor:
    """
    Executes tasks in parallel with configurable worker pool.
    
    Refactored with:
    - WorkerPool delegation
    - TaskExecutor integration
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_workers": 0,
        }
        
        self._pool = WorkerPool(
            size=max_workers,
            queue=self._task_queue,
            stats=self._stats,
            lock=self._lock
        )
    
    async def start(self):
        """Start the executor."""
        await self._pool.start()
    
    async def stop(self):
        """Stop the executor."""
        await self._pool.stop()
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> asyncio.Future:
        """
        Submit a task for parallel execution.
        
        Returns:
            Future that will resolve with task result
        """
        if not self._pool.running:
            raise RuntimeError("ParallelExecutor is not running")
        
        future = asyncio.Future()
        
        task_data = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "future": future,
            "created_at": datetime.now(),
        }
        
        await self._task_queue.put(task_data)
        
        async with self._lock:
            self._stats["total_tasks"] += 1
        
        return future
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            **self._stats,
            "queue_size": self._task_queue.qsize(),
            "running": self._pool.running,
        }
