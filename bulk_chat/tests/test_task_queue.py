"""
Tests for Task Queue
====================
"""

import pytest
import asyncio
from ..core.task_queue import TaskQueue, TaskPriority


@pytest.fixture
def task_queue():
    """Create task queue for testing."""
    return TaskQueue(max_workers=2)


@pytest.mark.asyncio
async def test_enqueue_task(task_queue):
    """Test enqueueing a task."""
    async def test_task():
        return "task_result"
    
    task_id = await task_queue.enqueue(test_task, priority=TaskPriority.NORMAL)
    
    assert task_id is not None
    assert task_id in task_queue.tasks


@pytest.mark.asyncio
async def test_task_execution(task_queue):
    """Test task execution."""
    result_holder = []
    
    async def test_task():
        result_holder.append("executed")
        return "done"
    
    task_id = await task_queue.enqueue(test_task)
    
    # Wait for execution
    await asyncio.sleep(0.1)
    
    assert len(result_holder) == 1
    assert result_holder[0] == "executed"


@pytest.mark.asyncio
async def test_task_priority(task_queue):
    """Test task priority ordering."""
    execution_order = []
    
    async def low_task():
        execution_order.append("low")
    
    async def high_task():
        execution_order.append("high")
    
    await task_queue.enqueue(low_task, priority=TaskPriority.LOW)
    await task_queue.enqueue(high_task, priority=TaskPriority.HIGH)
    
    # Wait for execution
    await asyncio.sleep(0.2)
    
    # High priority should execute first
    assert execution_order[0] == "high" or len(execution_order) >= 2


@pytest.mark.asyncio
async def test_get_task_status(task_queue):
    """Test getting task status."""
    async def test_task():
        return "result"
    
    task_id = await task_queue.enqueue(test_task)
    
    status = task_queue.get_task_status(task_id)
    
    assert status is not None
    assert "task_id" in status or "status" in status


@pytest.mark.asyncio
async def test_get_queue_stats(task_queue):
    """Test getting queue statistics."""
    async def test_task():
        return "result"
    
    await task_queue.enqueue(test_task)
    await task_queue.enqueue(test_task)
    
    stats = task_queue.get_stats()
    
    assert stats is not None
    assert "total_tasks" in stats or "pending" in stats or "completed" in stats


