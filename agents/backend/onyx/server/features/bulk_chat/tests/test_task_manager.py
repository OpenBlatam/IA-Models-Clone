"""
Tests for Task Manager
=======================
"""

import pytest
import asyncio
from ..core.task_manager import TaskManager, TaskPriority, TaskStatus


@pytest.fixture
def task_manager():
    """Create task manager for testing."""
    return TaskManager()


@pytest.mark.asyncio
async def test_create_task(task_manager):
    """Test creating a task."""
    task_id = task_manager.create_task(
        task_id="task1",
        title="Test Task",
        description="Test task description",
        priority=TaskPriority.HIGH,
        due_date=None
    )
    
    assert task_id == "task1"
    assert task_id in task_manager.tasks


@pytest.mark.asyncio
async def test_update_task_status(task_manager):
    """Test updating task status."""
    task_id = task_manager.create_task("task1", "Test", "Description")
    
    task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)
    
    task = task_manager.get_task(task_id)
    assert task.status == TaskStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_add_task_dependency(task_manager):
    """Test adding task dependency."""
    task1_id = task_manager.create_task("task1", "Task 1", "")
    task2_id = task_manager.create_task("task2", "Task 2", "")
    
    task_manager.add_dependency(task2_id, task1_id)
    
    task2 = task_manager.get_task(task2_id)
    assert task1_id in task2.dependencies or task1_id in task_manager.dependencies.get(task2_id, [])


@pytest.mark.asyncio
async def test_get_tasks_by_priority(task_manager):
    """Test getting tasks by priority."""
    task_manager.create_task("task1", "Low", "", priority=TaskPriority.LOW)
    task_manager.create_task("task2", "High", "", priority=TaskPriority.HIGH)
    task_manager.create_task("task3", "Medium", "", priority=TaskPriority.MEDIUM)
    
    high_priority_tasks = task_manager.get_tasks_by_priority(TaskPriority.HIGH)
    
    assert len(high_priority_tasks) >= 1
    assert all(t.priority == TaskPriority.HIGH for t in high_priority_tasks)


@pytest.mark.asyncio
async def test_get_task_manager_summary(task_manager):
    """Test getting task manager summary."""
    task_manager.create_task("task1", "Task 1", "")
    task_manager.create_task("task2", "Task 2", "")
    
    summary = task_manager.get_task_manager_summary()
    
    assert summary is not None
    assert "total_tasks" in summary or "tasks_by_status" in summary


