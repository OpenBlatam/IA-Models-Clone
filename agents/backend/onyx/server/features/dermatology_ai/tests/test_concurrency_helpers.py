"""
Concurrency Testing Helpers
Specialized helpers for concurrent and parallel testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ConcurrencyTestHelpers:
    """Helpers for concurrency testing"""
    
    @staticmethod
    async def run_concurrent_async(
        tasks: List[Callable],
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Run async tasks concurrently with optional limit"""
        if max_concurrent:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def run_with_limit(task):
                async with semaphore:
                    return await task()
            
            return await asyncio.gather(*[run_with_limit(task) for task in tasks])
        else:
            return await asyncio.gather(*[task() for task in tasks])
    
    @staticmethod
    def run_concurrent_sync(
        tasks: List[Callable],
        max_workers: int = 5
    ) -> List[Any]:
        """Run sync tasks concurrently"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            return [future.result() for future in as_completed(futures)]
    
    @staticmethod
    async def assert_no_race_condition(
        operation: Callable,
        iterations: int = 100,
        concurrent: int = 10
    ) -> bool:
        """Assert operation has no race condition"""
        results = []
        
        async def run_operation():
            result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
            results.append(result)
        
        tasks = [run_operation for _ in range(iterations)]
        await ConcurrencyTestHelpers.run_concurrent_async(tasks, max_concurrent=concurrent)
        
        # Check if all results are consistent
        if len(set(results)) > 1:
            return False  # Race condition detected
        return True


class LockHelpers:
    """Helpers for lock testing"""
    
    @staticmethod
    def create_mock_lock() -> Mock:
        """Create mock lock"""
        lock = Mock()
        lock.acquire = Mock(return_value=True)
        lock.release = Mock()
        lock.locked = Mock(return_value=False)
        lock.__enter__ = Mock(return_value=lock)
        lock.__exit__ = Mock(return_value=False)
        return lock
    
    @staticmethod
    def assert_lock_acquired(lock: Mock):
        """Assert lock was acquired"""
        assert lock.acquire.called, "Lock was not acquired"
    
    @staticmethod
    def assert_lock_released(lock: Mock):
        """Assert lock was released"""
        assert lock.release.called, "Lock was not released"


class SemaphoreHelpers:
    """Helpers for semaphore testing"""
    
    @staticmethod
    async def test_semaphore_limit(
        semaphore: asyncio.Semaphore,
        max_concurrent: int,
        operations: List[Callable]
    ) -> Dict[str, Any]:
        """Test semaphore enforces limit"""
        active_count = 0
        max_active = 0
        
        async def run_with_tracking(operation):
            nonlocal active_count, max_active
            async with semaphore:
                active_count += 1
                max_active = max(max_active, active_count)
                try:
                    if asyncio.iscoroutinefunction(operation):
                        result = await operation()
                    else:
                        result = operation()
                    return result
                finally:
                    active_count -= 1
        
        await asyncio.gather(*[run_with_tracking(op) for op in operations])
        
        return {
            "max_concurrent": max_active,
            "limit_enforced": max_active <= max_concurrent
        }
    
    @staticmethod
    def assert_semaphore_limit_enforced(
        result: Dict[str, Any],
        expected_limit: int
    ):
        """Assert semaphore limit was enforced"""
        assert result["limit_enforced"], \
            f"Semaphore limit not enforced: {result['max_concurrent']} > {expected_limit}"


class ThreadHelpers:
    """Helpers for thread testing"""
    
    @staticmethod
    def create_mock_thread(
        target: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Mock:
        """Create mock thread"""
        thread = Mock(spec=threading.Thread)
        thread.target = target
        thread.args = args
        thread.kwargs = kwargs or {}
        thread.start = Mock()
        thread.join = Mock()
        thread.is_alive = Mock(return_value=False)
        return thread
    
    @staticmethod
    def assert_thread_started(thread: Mock):
        """Assert thread was started"""
        assert thread.start.called, "Thread was not started"
    
    @staticmethod
    def assert_thread_joined(thread: Mock):
        """Assert thread was joined"""
        assert thread.join.called, "Thread was not joined"


class AsyncQueueHelpers:
    """Helpers for async queue testing"""
    
    @staticmethod
    def create_mock_async_queue(
        items: Optional[List[Any]] = None
    ) -> Mock:
        """Create mock async queue"""
        queue_items = list(items) if items else []
        queue = Mock()
        
        async def get_side_effect():
            if queue_items:
                return queue_items.pop(0)
            raise asyncio.QueueEmpty()
        
        async def put_side_effect(item: Any):
            queue_items.append(item)
        
        queue.get = AsyncMock(side_effect=get_side_effect)
        queue.put = AsyncMock(side_effect=put_side_effect)
        queue.qsize = Mock(return_value=len(queue_items))
        queue.empty = Mock(return_value=len(queue_items) == 0)
        queue.items = queue_items
        return queue
    
    @staticmethod
    def assert_item_queued(queue: Mock, item: Any):
        """Assert item was queued"""
        assert queue.put.called, "Item was not queued"
        if hasattr(queue, "items"):
            assert item in queue.items, f"Item {item} not in queue"
    
    @staticmethod
    def assert_item_dequeued(queue: Mock, expected_item: Any):
        """Assert item was dequeued"""
        assert queue.get.called, "Item was not dequeued"


# Convenience exports
run_concurrent_async = ConcurrencyTestHelpers.run_concurrent_async
run_concurrent_sync = ConcurrencyTestHelpers.run_concurrent_sync
assert_no_race_condition = ConcurrencyTestHelpers.assert_no_race_condition

create_mock_lock = LockHelpers.create_mock_lock
assert_lock_acquired = LockHelpers.assert_lock_acquired
assert_lock_released = LockHelpers.assert_lock_released

test_semaphore_limit = SemaphoreHelpers.test_semaphore_limit
assert_semaphore_limit_enforced = SemaphoreHelpers.assert_semaphore_limit_enforced

create_mock_thread = ThreadHelpers.create_mock_thread
assert_thread_started = ThreadHelpers.assert_thread_started
assert_thread_joined = ThreadHelpers.assert_thread_joined

create_mock_async_queue = AsyncQueueHelpers.create_mock_async_queue
assert_item_queued = AsyncQueueHelpers.assert_item_queued
assert_item_dequeued = AsyncQueueHelpers.assert_item_dequeued



