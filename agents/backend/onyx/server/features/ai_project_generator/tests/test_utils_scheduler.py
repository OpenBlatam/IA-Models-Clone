"""
Tests for TaskScheduler utility
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from ..utils.scheduler import TaskScheduler


class TestTaskScheduler:
    """Test suite for TaskScheduler"""

    def test_init(self):
        """Test TaskScheduler initialization"""
        scheduler = TaskScheduler()
        assert scheduler.tasks == {}
        assert scheduler.running_tasks == {}
        assert scheduler.task_history == []

    def test_schedule_task_interval(self):
        """Test scheduling task with interval"""
        scheduler = TaskScheduler()
        task_func = Mock()
        
        task_id = scheduler.schedule_task(
            task_id="test_task",
            task_func=task_func,
            schedule_type="interval",
            schedule_value=60,  # 60 seconds
            enabled=False  # Don't auto-run
        )
        
        assert task_id == "test_task"
        assert "test_task" in scheduler.tasks
        assert scheduler.tasks["test_task"]["schedule_type"] == "interval"
        assert scheduler.tasks["test_task"]["enabled"] is False

    def test_schedule_task_once(self):
        """Test scheduling task to run once"""
        scheduler = TaskScheduler()
        task_func = Mock()
        
        run_time = (datetime.now() + timedelta(hours=1)).isoformat()
        task_id = scheduler.schedule_task(
            task_id="once_task",
            task_func=task_func,
            schedule_type="once",
            schedule_value=run_time,
            enabled=False
        )
        
        assert scheduler.tasks["once_task"]["schedule_type"] == "once"
        assert scheduler.tasks["once_task"]["next_run"] == run_time

    def test_schedule_task_cron(self):
        """Test scheduling task with cron"""
        scheduler = TaskScheduler()
        task_func = Mock()
        
        task_id = scheduler.schedule_task(
            task_id="cron_task",
            task_func=task_func,
            schedule_type="cron",
            schedule_value="0 * * * *",  # Every hour
            enabled=False
        )
        
        assert scheduler.tasks["cron_task"]["schedule_type"] == "cron"

    def test_calculate_next_run_interval(self):
        """Test calculating next run for interval"""
        scheduler = TaskScheduler()
        
        next_run = scheduler._calculate_next_run("interval", 60)
        
        assert next_run is not None
        # Should be approximately 60 seconds from now
        run_time = datetime.fromisoformat(next_run)
        assert (run_time - datetime.now()).total_seconds() > 50
        assert (run_time - datetime.now()).total_seconds() < 70

    def test_calculate_next_run_once(self):
        """Test calculating next run for once"""
        scheduler = TaskScheduler()
        
        future_time = (datetime.now() + timedelta(hours=2)).isoformat()
        next_run = scheduler._calculate_next_run("once", future_time)
        
        assert next_run == future_time

    @pytest.mark.asyncio
    async def test_run_task_manual(self):
        """Test manually running a task"""
        scheduler = TaskScheduler()
        task_func = AsyncMock(return_value="success")
        
        task_id = scheduler.schedule_task(
            task_id="manual_task",
            task_func=task_func,
            schedule_type="interval",
            schedule_value=60,
            enabled=False
        )
        
        result = await scheduler.run_task(task_id)
        
        assert result["success"] is True
        assert scheduler.tasks["manual_task"]["run_count"] == 1
        assert scheduler.tasks["manual_task"]["success_count"] == 1

    @pytest.mark.asyncio
    async def test_run_task_error(self):
        """Test running task that fails"""
        scheduler = TaskScheduler()
        task_func = AsyncMock(side_effect=Exception("Task error"))
        
        task_id = scheduler.schedule_task(
            task_id="error_task",
            task_func=task_func,
            schedule_type="interval",
            schedule_value=60,
            enabled=False
        )
        
        result = await scheduler.run_task(task_id)
        
        assert result["success"] is False
        assert scheduler.tasks["error_task"]["failure_count"] == 1

    def test_enable_task(self):
        """Test enabling a task"""
        scheduler = TaskScheduler()
        task_func = Mock()
        
        task_id = scheduler.schedule_task(
            task_id="disable_task",
            task_func=task_func,
            schedule_type="interval",
            schedule_value=60,
            enabled=False
        )
        
        scheduler.enable_task(task_id)
        
        assert scheduler.tasks["disable_task"]["enabled"] is True

    def test_disable_task(self):
        """Test disabling a task"""
        scheduler = TaskScheduler()
        task_func = Mock()
        
        task_id = scheduler.schedule_task(
            task_id="enable_task",
            task_func=task_func,
            schedule_type="interval",
            schedule_value=60,
            enabled=True
        )
        
        scheduler.disable_task(task_id)
        
        assert scheduler.tasks["enable_task"]["enabled"] is False

    def test_get_task_status(self):
        """Test getting task status"""
        scheduler = TaskScheduler()
        task_func = Mock()
        
        task_id = scheduler.schedule_task(
            task_id="status_task",
            task_func=task_func,
            schedule_type="interval",
            schedule_value=60,
            enabled=False
        )
        
        status = scheduler.get_task_status(task_id)
        
        assert status["id"] == "status_task"
        assert status["enabled"] is False
        assert status["run_count"] == 0

    def test_list_tasks(self):
        """Test listing all tasks"""
        scheduler = TaskScheduler()
        task_func = Mock()
        
        scheduler.schedule_task("task1", task_func, "interval", 60, enabled=False)
        scheduler.schedule_task("task2", task_func, "interval", 120, enabled=False)
        
        tasks = scheduler.list_tasks()
        
        assert len(tasks) == 2

    def test_get_task_history(self):
        """Test getting task execution history"""
        scheduler = TaskScheduler()
        task_func = AsyncMock()
        
        task_id = scheduler.schedule_task(
            task_id="history_task",
            task_func=task_func,
            schedule_type="interval",
            schedule_value=60,
            enabled=False
        )
        
        # Run task
        asyncio.run(scheduler.run_task(task_id))
        
        history = scheduler.get_task_history(task_id)
        
        assert len(history) > 0
        assert history[0]["task_id"] == task_id

