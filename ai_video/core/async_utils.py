from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import weakref
            import random
from typing import Any, List, Dict, Optional
"""
AI Video System - Async Utilities

Advanced async utilities including concurrency control, async patterns,
retry mechanisms, and async helpers for production use.
"""


logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry operations."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    exceptions: tuple = (Exception,)
    timeout: Optional[float] = None


@dataclass
class AsyncTask:
    """Represents an async task with metadata."""
    id: str
    coroutine: Coroutine
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    exception: Optional[Exception] = None
    status: str = 'pending'  # pending, running, completed, failed, cancelled
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncTaskManager:
    """
    Manages async tasks with lifecycle tracking and control.
    
    Features:
    - Task lifecycle management
    - Task cancellation
    - Task monitoring
    - Resource cleanup
    """
    
    def __init__(self, max_concurrent_tasks: int = 100):
        
    """__init__ function."""
self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, AsyncTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._task_counter = 0
        self._lock = asyncio.Lock()
    
    async def submit_task(
        self,
        coro: Coroutine,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a task for execution."""
        if task_id is None:
            async with self._lock:
                self._task_counter += 1
                task_id = f"task_{self._task_counter}"
        
        async_task = AsyncTask(
            id=task_id,
            coroutine=coro,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = async_task
        
        # Start the task
        asyncio.create_task(self._execute_task(async_task))
        
        return task_id
    
    async def _execute_task(self, async_task: AsyncTask) -> None:
        """Execute a task with lifecycle management."""
        async_task.status = 'running'
        async_task.started_at = datetime.now()
        
        try:
            async with self.semaphore:
                task = asyncio.create_task(async_task.coroutine)
                self.running_tasks[async_task.id] = task
                
                result = await task
                
                async_task.result = result
                async_task.status = 'completed'
                async_task.completed_at = datetime.now()
                
        except asyncio.CancelledError:
            async_task.status = 'cancelled'
            async_task.completed_at = datetime.now()
            raise
        except Exception as e:
            async_task.status = 'failed'
            async_task.exception = e
            async_task.completed_at = datetime.now()
            logger.error(f"Task {async_task.id} failed: {e}")
        finally:
            self.running_tasks.pop(async_task.id, None)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            return True
        
        return False
    
    async def cancel_all_tasks(self) -> int:
        """Cancel all running tasks."""
        cancelled_count = 0
        
        for task_id in list(self.running_tasks.keys()):
            if await self.cancel_task(task_id):
                cancelled_count += 1
        
        return cancelled_count
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            'id': task.id,
            'status': task.status,
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'metadata': task.metadata,
            'has_result': task.result is not None,
            'has_exception': task.exception is not None
        }
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks."""
        return [self.get_task_status(task_id) for task_id in self.tasks.keys()]
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a task to complete and return its result."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        if task.status == 'completed':
            return task.result
        
        if task.status == 'failed':
            raise task.exception
        
        if task.status == 'cancelled':
            raise asyncio.CancelledError()
        
        # Wait for the task to complete
        start_time = time.time()
        while task.status == 'running':
            if timeout and time.time() - start_time > timeout:
                raise asyncio.TimeoutError()
            await asyncio.sleep(0.1)
        
        if task.status == 'completed':
            return task.result
        elif task.status == 'failed':
            raise task.exception
        else:
            raise asyncio.CancelledError()
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up completed tasks older than specified age."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if (task.status in ['completed', 'failed', 'cancelled'] and
                task.completed_at and task.completed_at < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            self.tasks.pop(task_id, None)
        
        return len(tasks_to_remove)


class AsyncRateLimiter:
    """
    Advanced async rate limiter with multiple algorithms.
    
    Features:
    - Token bucket algorithm
    - Leaky bucket algorithm
    - Sliding window
    - Distributed rate limiting
    """
    
    def __init__(
        self,
        max_tokens: int,
        refill_rate: float,
        algorithm: str = 'token_bucket',
        distributed: bool = False
    ):
        
    """__init__ function."""
self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.algorithm = algorithm
        self.distributed = distributed
        
        if algorithm == 'token_bucket':
            self.tokens = max_tokens
            self.last_refill = time.time()
        elif algorithm == 'leaky_bucket':
            self.tokens = 0
            self.last_leak = time.time()
        
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the rate limiter."""
        async with self.lock:
            if self.algorithm == 'token_bucket':
                return await self._token_bucket_acquire(tokens)
            elif self.algorithm == 'leaky_bucket':
                return await self._leaky_bucket_acquire(tokens)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    async def _token_bucket_acquire(self, tokens: int) -> bool:
        """Token bucket algorithm implementation."""
        current_time = time.time()
        
        # Refill tokens
        time_passed = current_time - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = current_time
        
        # Check if we can acquire tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    async def _leaky_bucket_acquire(self, tokens: int) -> bool:
        """Leaky bucket algorithm implementation."""
        current_time = time.time()
        
        # Leak tokens
        time_passed = current_time - self.last_leak
        tokens_to_leak = time_passed * self.refill_rate
        self.tokens = max(0, self.tokens - tokens_to_leak)
        self.last_leak = current_time
        
        # Check if we can add tokens
        if self.tokens + tokens <= self.max_tokens:
            self.tokens += tokens
            return True
        
        return False
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until tokens are available."""
        start_time = time.time()
        
        while True:
            if await self.acquire(tokens):
                return True
            
            if timeout and time.time() - start_time > timeout:
                return False
            
            # Wait a bit before trying again
            await asyncio.sleep(0.1)


class AsyncRetry:
    """
    Advanced retry mechanism with exponential backoff and jitter.
    
    Features:
    - Exponential backoff
    - Jitter for distributed systems
    - Configurable exceptions
    - Timeout support
    """
    
    def __init__(self, config: RetryConfig):
        
    """__init__ function."""
self.config = config
    
    async def execute(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs
    ) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if self.config.timeout:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    return await func(*args, **kwargs)
                    
            except self.config.exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    raise last_exception
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class AsyncBatchProcessor:
    """
    Process items in batches with concurrency control.
    
    Features:
    - Batch processing
    - Concurrency limits
    - Progress tracking
    - Error handling
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 5,
        max_workers_per_batch: int = 3
    ):
        
    """__init__ function."""
self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.max_workers_per_batch = max_workers_per_batch
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def process_batches(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Coroutine[Any, Any, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items in batches."""
        results = []
        total_items = len(items)
        processed_items = 0
        
        # Create batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, total_items, self.batch_size)
        ]
        
        # Process batches concurrently
        batch_tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch, processor_func))
            batch_tasks.append(task)
        
        # Collect results
        for task in asyncio.as_completed(batch_tasks):
            batch_results = await task
            results.extend(batch_results)
            processed_items += len(batch_results)
            
            if progress_callback:
                progress_callback(processed_items, total_items)
        
        return results
    
    async def _process_batch(
        self,
        batch: List[Any],
        processor_func: Callable[[Any], Coroutine[Any, Any, Any]]
    ) -> List[Any]:
        """Process a single batch."""
        async with self.semaphore:
            # Process items in batch with limited concurrency
            semaphore = asyncio.Semaphore(self.max_workers_per_batch)
            
            async def process_item(item) -> Any:
                async with semaphore:
                    return await processor_func(item)
            
            tasks = [process_item(item) for item in batch]
            return await asyncio.gather(*tasks, return_exceptions=True)


class AsyncCache:
    """
    Async cache with TTL and automatic cleanup.
    
    Features:
    - TTL support
    - Automatic cleanup
    - Thread-safe operations
    - Memory-efficient
    """
    
    def __init__(self, default_ttl: Optional[float] = None):
        
    """__init__ function."""
self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry['expires_at'] and time.time() > entry['expires_at']:
                del self.cache[key]
                return None
            
            return entry['value']
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache."""
        async with self.lock:
            expires_at = None
            if ttl or self.default_ttl:
                expires_at = time.time() + (ttl or self.default_ttl)
            
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        async with self.lock:
            for key, entry in self.cache.items():
                if entry['expires_at'] and current_time > entry['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
        
        return len(expired_keys)
    
    async def start_cleanup_task(self, interval: float = 300) -> None:
        """Start automatic cleanup task."""
        if self._cleanup_task:
            return
        
        async def cleanup_loop():
            
    """cleanup_loop function."""
while True:
                try:
                    await asyncio.sleep(interval)
                    await self.cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop_cleanup_task(self) -> None:
        """Stop automatic cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None


# Async context managers
@asynccontextmanager
async def timeout_context(timeout: float):
    """Context manager for timeout control."""
    try:
        yield
    except asyncio.TimeoutError:
        raise
    except Exception as e:
        raise e


@asynccontextmanager
async def retry_context(config: RetryConfig):
    """Context manager for retry operations."""
    retry = AsyncRetry(config)
    try:
        yield retry
    except Exception as e:
        raise e


# Async decorators
def async_retry(config: RetryConfig):
    """Decorator for async retry functionality."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            retry = AsyncRetry(config)
            return await retry.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def async_timeout(timeout: float):
    """Decorator for async timeout control."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator


def async_rate_limit(rate_limiter: AsyncRateLimiter):
    """Decorator for async rate limiting."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if not await rate_limiter.acquire():
                raise Exception("Rate limit exceeded")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global instances
task_manager = AsyncTaskManager()
default_cache = AsyncCache(default_ttl=3600)  # 1 hour default TTL


# Utility functions
async def gather_with_concurrency_limit(
    coros: List[Coroutine],
    limit: int,
    return_exceptions: bool = False
) -> List[Any]:
    """Execute coroutines with concurrency limit."""
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_coro(coro) -> Any:
        async with semaphore:
            return await coro
    
    limited_coros = [limited_coro(coro) for coro in coros]
    return await asyncio.gather(*limited_coros, return_exceptions=return_exceptions)


async def wait_for_first(
    coros: List[Coroutine],
    timeout: Optional[float] = None
) -> Any:
    """Wait for the first coroutine to complete."""
    done, pending = await asyncio.wait(
        coros,
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # Cancel pending tasks
    for task in pending:
        task.cancel()
    
    # Return first completed result
    return done.pop().result()


async def run_in_executor(
    func: Callable,
    *args,
    executor: Optional[Any] = None,
    **kwargs
) -> Any:
    """Run function in executor to avoid blocking."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        functools.partial(func, *args, **kwargs)
    )


async def cleanup_async_resources() -> None:
    """Cleanup async resources."""
    # Cleanup task manager
    await task_manager.cancel_all_tasks()
    await task_manager.cleanup_completed_tasks()
    
    # Cleanup cache
    await default_cache.stop_cleanup_task()
    await default_cache.clear()
    
    logger.info("Async resources cleaned up") 