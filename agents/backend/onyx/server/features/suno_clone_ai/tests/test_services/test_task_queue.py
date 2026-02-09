"""
Comprehensive Unit Tests for Task Queue

Tests cover task queue functionality with diverse test cases
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from services.task_queue import (
    TaskQueue,
    Task,
    TaskPriority,
    TaskStatus,
    get_task_queue,
    process_sqs_message
)


class TestTask:
    """Test cases for Task dataclass"""
    
    def test_task_creation(self):
        """Test creating a task"""
        task = Task(
            id="task123",
            task_type="generate_music",
            payload={"prompt": "test"}
        )
        
        assert task.id == "task123"
        assert task.task_type == "generate_music"
        assert task.payload == {"prompt": "test"}
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert task.created_at is not None
    
    def test_task_custom_priority(self):
        """Test task with custom priority"""
        task = Task(
            id="task123",
            task_type="test",
            payload={},
            priority=TaskPriority.HIGH
        )
        assert task.priority == TaskPriority.HIGH
    
    def test_task_to_dict(self):
        """Test converting task to dictionary"""
        task = Task(
            id="task123",
            task_type="test",
            payload={"key": "value"}
        )
        
        result = task.to_dict()
        
        assert result["id"] == "task123"
        assert result["task_type"] == "test"
        assert result["priority"] == TaskPriority.NORMAL.value
        assert result["status"] == TaskStatus.PENDING.value
        assert "created_at" in result


class TestTaskQueue:
    """Test cases for TaskQueue class"""
    
    def test_task_queue_init_without_celery(self):
        """Test initializing queue without Celery"""
        queue = TaskQueue(use_celery=False)
        assert queue.use_celery is False
        assert len(queue.tasks) == 0
    
    def test_task_queue_enqueue_basic(self):
        """Test enqueueing a basic task"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue(
            task_type="test_task",
            payload={"data": "test"}
        )
        
        assert task_id is not None
        assert task_id in queue.tasks
        assert queue.tasks[task_id].task_type == "test_task"
        assert queue.tasks[task_id].status == TaskStatus.QUEUED
    
    def test_task_queue_enqueue_with_custom_id(self):
        """Test enqueueing with custom task ID"""
        queue = TaskQueue(use_celery=False)
        custom_id = "custom-task-123"
        task_id = queue.enqueue(
            task_type="test",
            payload={},
            task_id=custom_id
        )
        
        assert task_id == custom_id
        assert custom_id in queue.tasks
    
    def test_task_queue_enqueue_with_priority(self):
        """Test enqueueing with custom priority"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue(
            task_type="test",
            payload={},
            priority=TaskPriority.CRITICAL
        )
        
        assert queue.tasks[task_id].priority == TaskPriority.CRITICAL
    
    def test_task_queue_get_task(self):
        """Test getting a task by ID"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue("test", {})
        
        task = queue.get_task(task_id)
        assert task is not None
        assert task.id == task_id
    
    def test_task_queue_get_task_not_found(self):
        """Test getting non-existent task"""
        queue = TaskQueue(use_celery=False)
        task = queue.get_task("nonexistent")
        assert task is None
    
    def test_task_queue_update_status_processing(self):
        """Test updating task status to processing"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue("test", {})
        
        queue.update_task_status(task_id, TaskStatus.PROCESSING)
        
        task = queue.get_task(task_id)
        assert task.status == TaskStatus.PROCESSING
        assert task.started_at is not None
    
    def test_task_queue_update_status_completed(self):
        """Test updating task status to completed"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue("test", {})
        
        queue.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            result={"output": "success"}
        )
        
        task = queue.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result == {"output": "success"}
    
    def test_task_queue_update_status_failed(self):
        """Test updating task status to failed"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue("test", {})
        
        queue.update_task_status(
            task_id,
            TaskStatus.FAILED,
            error="Test error"
        )
        
        task = queue.get_task(task_id)
        assert task.status == TaskStatus.RETRYING  # Should retry
        assert task.error == "Test error"
        assert task.retry_count == 1
    
    def test_task_queue_update_status_failed_max_retries(self):
        """Test failed task with max retries"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue("test", {}, max_retries=1)
        
        # Fail multiple times
        queue.update_task_status(task_id, TaskStatus.FAILED, error="Error 1")
        queue.update_task_status(task_id, TaskStatus.FAILED, error="Error 2")
        
        task = queue.get_task(task_id)
        assert task.retry_count >= 1
    
    def test_task_queue_cancel_task(self):
        """Test canceling a task"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue("test", {})
        
        result = queue.cancel_task(task_id)
        assert result is True
        
        task = queue.get_task(task_id)
        assert task.status == TaskStatus.CANCELLED
    
    def test_task_queue_cancel_task_not_found(self):
        """Test canceling non-existent task"""
        queue = TaskQueue(use_celery=False)
        result = queue.cancel_task("nonexistent")
        assert result is False
    
    def test_task_queue_cancel_task_already_processing(self):
        """Test canceling task that's already processing"""
        queue = TaskQueue(use_celery=False)
        task_id = queue.enqueue("test", {})
        queue.update_task_status(task_id, TaskStatus.PROCESSING)
        
        result = queue.cancel_task(task_id)
        # Should not cancel if already processing
        assert result is False
    
    def test_task_queue_get_queue_stats(self):
        """Test getting queue statistics"""
        queue = TaskQueue(use_celery=False)
        
        # Add tasks with different statuses
        task1 = queue.enqueue("test1", {})
        task2 = queue.enqueue("test2", {})
        queue.update_task_status(task1, TaskStatus.COMPLETED)
        queue.update_task_status(task2, TaskStatus.PROCESSING)
        
        stats = queue.get_queue_stats()
        
        assert stats["total_tasks"] == 2
        assert "tasks_by_status" in stats
        assert stats["completed"] >= 1
        assert stats["processing"] >= 1
    
    def test_task_queue_get_tasks_by_status(self):
        """Test getting tasks by status"""
        queue = TaskQueue(use_celery=False)
        
        task1 = queue.enqueue("test1", {})
        task2 = queue.enqueue("test2", {})
        queue.update_task_status(task1, TaskStatus.COMPLETED)
        
        completed_tasks = queue.get_tasks_by_status(TaskStatus.COMPLETED)
        assert len(completed_tasks) >= 1
        assert all(task.status == TaskStatus.COMPLETED for task in completed_tasks)


class TestGetTaskQueue:
    """Test cases for get_task_queue function"""
    
    def test_get_task_queue_singleton(self):
        """Test that get_task_queue returns singleton"""
        queue1 = get_task_queue(use_celery=False)
        queue2 = get_task_queue(use_celery=False)
        
        assert queue1 is queue2
        assert isinstance(queue1, TaskQueue)


class TestProcessSqsMessage:
    """Test cases for process_sqs_message function"""
    
    def test_process_sqs_message_generate_music(self):
        """Test processing generate_music message"""
        message_body = {
            "task_type": "generate_music",
            "payload": {
                "prompt": "test song",
                "duration": 30
            },
            "task_id": "test123"
        }
        
        with patch('services.task_queue.get_music_generator') as mock_gen:
            mock_generator = MagicMock()
            mock_generator.generate.return_value = {"audio": "data"}
            mock_gen.return_value = mock_generator
            
            result = process_sqs_message(message_body)
            
            assert result["status"] == "success"
            assert "result" in result
    
    def test_process_sqs_message_unknown_type(self):
        """Test processing unknown task type"""
        message_body = {
            "task_type": "unknown_task",
            "payload": {}
        }
        
        result = process_sqs_message(message_body)
        
        assert result["status"] == "error"
        assert "error" in result
    
    def test_process_sqs_message_error_handling(self):
        """Test error handling in process_sqs_message"""
        message_body = {
            "task_type": "generate_music",
            "payload": {}
        }
        
        with patch('services.task_queue.get_music_generator', side_effect=Exception("Test error")):
            result = process_sqs_message(message_body)
            
            assert result["status"] == "error"
            assert "error" in result















