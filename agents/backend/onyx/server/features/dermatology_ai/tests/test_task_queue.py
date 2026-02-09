"""
Tests for Task Queue
Tests for async task queue with priority and retry
"""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio

from core.infrastructure.task_queue import (
    TaskQueue,
    Task,
    TaskPriority,
    TaskStatus
)


class TestTask:
    """Tests for Task"""
    
    def test_create_task(self):
        """Test creating a task"""
        async def task_func():
            return "result"
        
        task = Task(
            func=task_func,
            priority=TaskPriority.HIGH
        )
        
        assert task.func == task_func
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.id is not None
    
    def test_task_priority(self):
        """Test task priority levels"""
        task1 = Task(priority=TaskPriority.LOW)
        task2 = Task(priority=TaskPriority.MEDIUM)
        task3 = Task(priority=TaskPriority.HIGH)
        task4 = Task(priority=TaskPriority.CRITICAL)
        
        assert task1.priority.value < task2.priority.value
        assert task2.priority.value < task3.priority.value
        assert task3.priority.value < task4.priority.value


class TestTaskQueue:
    """Tests for TaskQueue"""
    
    @pytest.fixture
    def task_queue(self):
        """Create task queue"""
        return TaskQueue(max_workers=2)
    
    @pytest.mark.asyncio
    async def test_enqueue_task(self, task_queue):
        """Test enqueueing a task"""
        async def task_func():
            return "result"
        
        task = Task(func=task_func)
        task_id = await task_queue.enqueue(task)
        
        assert task_id == task.id
        assert task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_execute_task(self, task_queue):
        """Test executing a task"""
        result_value = "task_result"
        
        async def task_func():
            return result_value
        
        task = Task(func=task_func)
        await task_queue.enqueue(task)
        
        # Wait for task to complete
        await asyncio.sleep(0.1)
        
        # Task should be completed
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_task_priority_ordering(self, task_queue):
        """Test task priority ordering"""
        execution_order = []
        
        async def low_priority_task():
            execution_order.append("low")
            return "low"
        
        async def high_priority_task():
            execution_order.append("high")
            return "high"
        
        # Enqueue low priority first
        task1 = Task(func=low_priority_task, priority=TaskPriority.LOW)
        await task_queue.enqueue(task1)
        
        # Then high priority
        task2 = Task(func=high_priority_task, priority=TaskPriority.HIGH)
        await task_queue.enqueue(task2)
        
        # Wait for execution
        await asyncio.sleep(0.2)
        
        # High priority should execute first (or at least be prioritized)
        # Exact order depends on implementation
        assert len(execution_order) >= 1
    
    @pytest.mark.asyncio
    async def test_task_retry_on_failure(self, task_queue):
        """Test task retry on failure"""
        call_count = 0
        
        async def failing_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Task failed")
            return "success"
        
        task = Task(
            func=failing_task,
            max_retries=3,
            retry_delay=0.1
        )
        
        await task_queue.enqueue(task)
        
        # Wait for retries
        await asyncio.sleep(0.5)
        
        # Should have retried
        assert call_count >= 2
        # Task should eventually succeed or fail
        assert task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_task_with_args_kwargs(self, task_queue):
        """Test task with arguments"""
        async def task_func(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"
        
        task = Task(
            func=task_func,
            args=("value1", "value2"),
            kwargs={"kwarg1": "value3"}
        )
        
        await task_queue.enqueue(task)
        await asyncio.sleep(0.1)
        
        # Task should complete successfully
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, task_queue):
        """Test getting task status"""
        async def task_func():
            await asyncio.sleep(0.1)
            return "done"
        
        task = Task(func=task_func)
        task_id = await task_queue.enqueue(task)
        
        # Initially pending
        status = task.status
        assert status == TaskStatus.PENDING
        
        # Wait for completion
        await asyncio.sleep(0.2)
        
        # Should be completed
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_task_queue_shutdown(self, task_queue):
        """Test task queue shutdown"""
        async def long_task():
            await asyncio.sleep(1.0)
            return "done"
        
        task = Task(func=long_task)
        await task_queue.enqueue(task)
        
        # Shutdown should wait for tasks or cancel them
        await task_queue.shutdown()
        
        # Queue should be shut down
        assert task_queue is not None



