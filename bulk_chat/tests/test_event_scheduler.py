"""
Tests for Event Scheduler
==========================
"""

import pytest
import asyncio
from ..core.event_scheduler import EventScheduler, ScheduleType


@pytest.fixture
def event_scheduler():
    """Create event scheduler for testing."""
    return EventScheduler()


@pytest.mark.asyncio
async def test_schedule_event_cron(event_scheduler):
    """Test scheduling event with cron."""
    async def event_handler():
        return "executed"
    
    schedule_id = event_scheduler.schedule_event(
        schedule_id="test_schedule",
        schedule_type=ScheduleType.CRON,
        cron_expression="0 * * * *",  # Every hour
        handler=event_handler
    )
    
    assert schedule_id == "test_schedule"
    assert schedule_id in event_scheduler.schedules


@pytest.mark.asyncio
async def test_schedule_event_interval(event_scheduler):
    """Test scheduling event with interval."""
    async def event_handler():
        return "executed"
    
    schedule_id = event_scheduler.schedule_event(
        "test_interval",
        ScheduleType.INTERVAL,
        interval_seconds=60.0,
        handler=event_handler
    )
    
    assert schedule_id is not None
    assert schedule_id in event_scheduler.schedules


@pytest.mark.asyncio
async def test_cancel_schedule(event_scheduler):
    """Test canceling a schedule."""
    async def handler():
        return "executed"
    
    schedule_id = event_scheduler.schedule_event(
        "test_schedule",
        ScheduleType.INTERVAL,
        interval_seconds=60.0,
        handler=handler
    )
    
    assert schedule_id in event_scheduler.schedules
    
    event_scheduler.cancel_schedule(schedule_id)
    
    assert schedule_id not in event_scheduler.schedules


@pytest.mark.asyncio
async def test_get_schedule_status(event_scheduler):
    """Test getting schedule status."""
    async def handler():
        return "executed"
    
    schedule_id = event_scheduler.schedule_event(
        "test_schedule",
        ScheduleType.INTERVAL,
        interval_seconds=60.0,
        handler=handler
    )
    
    status = event_scheduler.get_schedule_status(schedule_id)
    
    assert status is not None
    assert "schedule_id" in status or "enabled" in status


@pytest.mark.asyncio
async def test_get_event_scheduler_summary(event_scheduler):
    """Test getting event scheduler summary."""
    async def handler():
        return "executed"
    
    event_scheduler.schedule_event("schedule1", ScheduleType.INTERVAL, 60.0, handler)
    event_scheduler.schedule_event("schedule2", ScheduleType.INTERVAL, 120.0, handler)
    
    summary = event_scheduler.get_event_scheduler_summary()
    
    assert summary is not None
    assert "total_schedules" in summary or "active_schedules" in summary


