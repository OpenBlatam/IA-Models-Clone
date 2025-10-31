"""
Parallel Execution System - Multi-threading and multi-processing
Optimized for CPU and I/O intensive tasks
"""

import asyncio
import threading
import multiprocessing as mp
from typing import Any, List, Dict, Callable, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import time
import os

@dataclass
class ExecutionResult:
    """Result of parallel execution"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    process_id: Optional[int] = None

class TaskPool:
    """Pool of workers for parallel task execution"""
    
    def __init__(
        self,
        max_workers: int = None,
        use_threads: bool = True,
        use_processes: bool = False,
        queue_size: int = 1000
    ):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.use_threads = use_threads
        self.use_processes = use_processes
        self.queue_size = queue_size
        
        # Execution pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Task queue and result storage
        self.task_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.results: Dict[str, ExecutionResult] = {}
        self.result_lock = threading.Lock()
        
        # Workers
        self.workers: List[threading.Thread] = []
        self.running = False
        
        # Statistics
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0

    def start(self):
        """Start the task pool"""
        if self.running:
            return
        
        self.running = True
        
        # Initialize execution pools
        if self.use_threads:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        if self.use_processes:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"TaskPool-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def stop(self):
        """Stop the task pool"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        
        # Shutdown execution pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None

    def _worker(self):
        """Worker thread that processes tasks"""
        while self.running:
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1.0)
                
                # Process task
                result = self._process_task(task)
                
                # Store result
                with self.result_lock:
                    self.results[task["id"]] = result
                
                # Mark task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    def _process_task(self, task: Dict[str, Any]) -> ExecutionResult:
        """Process a single task"""
        start_time = time.time()
        task_id = task["id"]
        
        try:
            func = task["func"]
            args = task.get("args", ())
            kwargs = task.get("kwargs", {})
            execution_type = task.get("execution_type", "thread")
            
            # Execute based on type
            if execution_type == "thread" and self.thread_pool:
                future = self.thread_pool.submit(func, *args, **kwargs)
                result = future.result()
            elif execution_type == "process" and self.process_pool:
                future = self.process_pool.submit(func, *args, **kwargs)
                result = future.result()
            else:
                # Fallback to direct execution
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            self.completed_tasks += 1
            self.total_execution_time += execution_time
            
            return ExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=threading.current_thread().name,
                process_id=os.getpid()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_tasks += 1
            
            return ExecutionResult(
                success=False,
                error=e,
                execution_time=execution_time,
                worker_id=threading.current_thread().name,
                process_id=os.getpid()
            )

    def submit(
        self, 
        func: Callable, 
        *args, 
        execution_type: str = "thread",
        **kwargs
    ) -> str:
        """Submit a task for execution"""
        task_id = f"task-{int(time.time() * 1000)}-{id(func)}"
        
        task = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "execution_type": execution_type,
            "submitted_at": time.time()
        }
        
        self.task_queue.put(task)
        return task_id

    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        """Get result for a specific task"""
        with self.result_lock:
            return self.results.get(task_id)

    def wait_for_completion(self, task_ids: List[str], timeout: float = None) -> List[ExecutionResult]:
        """Wait for specific tasks to complete"""
        start_time = time.time()
        results = []
        
        while task_ids:
            if timeout and (time.time() - start_time) > timeout:
                break
            
            completed_ids = []
            for task_id in task_ids:
                result = self.get_result(task_id)
                if result:
                    results.append(result)
                    completed_ids.append(task_id)
            
            # Remove completed tasks
            for task_id in completed_ids:
                task_ids.remove(task_id)
            
            if task_ids:  # Still waiting
                time.sleep(0.01)  # Small delay
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        avg_execution_time = (
            self.total_execution_time / self.completed_tasks 
            if self.completed_tasks > 0 else 0
        )
        
        return {
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": (
                self.completed_tasks / (self.completed_tasks + self.failed_tasks)
                if (self.completed_tasks + self.failed_tasks) > 0 else 0
            ),
            "avg_execution_time": avg_execution_time,
            "total_execution_time": self.total_execution_time,
            "queue_size": self.task_queue.qsize(),
            "active_workers": len(self.workers),
            "running": self.running
        }

class ParallelExecutor:
    """High-level parallel execution coordinator"""
    
    def __init__(
        self,
        max_threads: int = None,
        max_processes: int = None,
        auto_scale: bool = True
    ):
        self.max_threads = max_threads or min(32, (mp.cpu_count() or 1) + 4)
        self.max_processes = max_processes or min(8, mp.cpu_count() or 1)
        self.auto_scale = auto_scale
        
        # Task pools
        self.thread_pool = TaskPool(
            max_workers=self.max_threads,
            use_threads=True,
            use_processes=False
        )
        self.process_pool = TaskPool(
            max_workers=self.max_processes,
            use_threads=False,
            use_processes=True
        )
        
        # Performance monitoring
        self.performance_history: List[Dict[str, Any]] = []
        self.auto_scaling_enabled = auto_scale

    def start(self):
        """Start all task pools"""
        self.thread_pool.start()
        self.process_pool.start()
        
        if self.auto_scaling_enabled:
            threading.Thread(target=self._auto_scale_worker, daemon=True).start()

    def stop(self):
        """Stop all task pools"""
        self.thread_pool.stop()
        self.process_pool.stop()

    def _auto_scale_worker(self):
        """Background worker for auto-scaling"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                self._adjust_worker_count()
            except Exception as e:
                print(f"Auto-scaling error: {e}")

    def _adjust_worker_count(self):
        """Adjust worker count based on performance"""
        thread_stats = self.thread_pool.get_stats()
        process_stats = self.process_pool.get_stats()
        
        # Calculate load metrics
        thread_load = thread_stats["queue_size"] / max(thread_stats["active_workers"], 1)
        process_load = process_stats["queue_size"] / max(process_stats["active_workers"], 1)
        
        # Record performance
        self.performance_history.append({
            "timestamp": time.time(),
            "thread_load": thread_load,
            "process_load": process_load,
            "thread_workers": thread_stats["active_workers"],
            "process_workers": process_stats["active_workers"]
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    async def execute_parallel(
        self,
        tasks: List[Tuple[Callable, tuple, dict, str]],  # (func, args, kwargs, execution_type)
        max_concurrent: int = None
    ) -> List[ExecutionResult]:
        """Execute tasks in parallel with optimal resource allocation"""
        if not max_concurrent:
            max_concurrent = min(len(tasks), self.max_threads + self.max_processes)
        
        # Categorize tasks by type
        thread_tasks = []
        process_tasks = []
        
        for func, args, kwargs, execution_type in tasks:
            if execution_type == "process":
                process_tasks.append((func, args, kwargs))
            else:
                thread_tasks.append((func, args, kwargs))
        
        # Submit tasks to appropriate pools
        thread_task_ids = []
        process_task_ids = []
        
        for func, args, kwargs in thread_tasks:
            task_id = self.thread_pool.submit(func, *args, **kwargs, execution_type="thread")
            thread_task_ids.append(task_id)
        
        for func, args, kwargs in process_tasks:
            task_id = self.process_pool.submit(func, *args, **kwargs, execution_type="process")
            process_task_ids.append(task_id)
        
        # Wait for completion
        all_task_ids = thread_task_ids + process_task_ids
        results = self.thread_pool.wait_for_completion(thread_task_ids)
        results.extend(self.process_pool.wait_for_completion(process_task_ids))
        
        return results

    def execute_map(
        self,
        func: Callable,
        items: List[Any],
        execution_type: str = "thread",
        chunk_size: int = None
    ) -> List[Any]:
        """Apply function to items in parallel (map operation)"""
        if not chunk_size:
            chunk_size = max(1, len(items) // (self.max_threads + self.max_processes))
        
        # Split items into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Create tasks for each chunk
        tasks = []
        for chunk in chunks:
            tasks.append((self._process_chunk, (func, chunk), {}, execution_type))
        
        # Execute tasks
        results = []
        for func, args, kwargs, exec_type in tasks:
            if exec_type == "process":
                task_id = self.process_pool.submit(func, *args, **kwargs, execution_type="process")
                result = self.process_pool.wait_for_completion([task_id])[0]
            else:
                task_id = self.thread_pool.submit(func, *args, **kwargs, execution_type="thread")
                result = self.thread_pool.wait_for_completion([task_id])[0]
            
            if result.success:
                results.extend(result.result)
            else:
                print(f"Chunk processing failed: {result.error}")
        
        return results

    def _process_chunk(self, func: Callable, items: List[Any]) -> List[Any]:
        """Process a chunk of items"""
        results = []
        for item in items:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                print(f"Item processing failed: {e}")
                results.append(None)
        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        thread_stats = self.thread_pool.get_stats()
        process_stats = self.process_pool.get_stats()
        
        # Calculate efficiency metrics
        total_tasks = thread_stats["completed_tasks"] + process_stats["completed_tasks"]
        total_time = thread_stats["total_execution_time"] + process_stats["total_execution_time"]
        
        return {
            "thread_pool": thread_stats,
            "process_pool": process_stats,
            "total_tasks": total_tasks,
            "total_execution_time": total_time,
            "performance_history": self.performance_history[-10:],  # Last 10 entries
            "auto_scaling_enabled": self.auto_scaling_enabled
        }

# Utility functions for common parallel patterns

def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: int = None,
    use_processes: bool = False
) -> List[Any]:
    """Apply function to items in parallel (synchronous)"""
    executor = ParallelExecutor(
        max_threads=max_workers if not use_processes else None,
        max_processes=max_workers if use_processes else None
    )
    
    try:
        executor.start()
        return executor.execute_map(func, items, "process" if use_processes else "thread")
    finally:
        executor.stop()

def parallel_filter(
    func: Callable,
    items: List[Any],
    max_workers: int = None,
    use_processes: bool = False
) -> List[Any]:
    """Filter items using parallel function (synchronous)"""
    def filter_func(item):
        try:
            return func(item)
        except:
            return False
    
    filtered_items = parallel_map(filter_func, items, max_workers, use_processes)
    return [item for item, keep in zip(items, filtered_items) if keep]

def parallel_reduce(
    func: Callable,
    items: List[Any],
    initial: Any = None,
    max_workers: int = None,
    use_processes: bool = False
) -> Any:
    """Reduce items using parallel function (synchronous)"""
    if not items:
        return initial
    
    if len(items) == 1:
        return items[0]
    
    executor = ParallelExecutor(
        max_threads=max_workers if not use_processes else None,
        max_processes=max_workers if use_processes else None
    )
    
    try:
        executor.start()
        current_items = items.copy()
        
        # Reduce in pairs until we have one result
        while len(current_items) > 1:
            # Process pairs
            pairs = [
                (current_items[i], current_items[i + 1])
                for i in range(0, len(current_items) - 1, 2)
            ]
            
            # Add odd item if exists
            if len(current_items) % 2 == 1:
                pairs.append((current_items[-1], initial))
            
            # Process pairs in parallel
            pair_results = executor.execute_map(
                lambda pair: func(pair[0], pair[1]),
                pairs,
                "process" if use_processes else "thread"
            )
            
            # Update current items with results
            current_items = [result for result in pair_results if result is not None]
        
        return current_items[0] if current_items else initial
    finally:
        executor.stop()





