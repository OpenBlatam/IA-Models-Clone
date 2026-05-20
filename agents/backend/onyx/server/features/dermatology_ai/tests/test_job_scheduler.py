"""
Tests for Job Scheduler
Tests for scheduled job execution
"""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio
from datetime import datetime, timedelta

from core.infrastructure.job_scheduler import JobScheduler


class TestJobScheduler:
    """Tests for JobScheduler"""
    
    @pytest.fixture
    def job_scheduler(self):
        """Create job scheduler"""
        return JobScheduler()
    
    @pytest.mark.asyncio
    async def test_schedule_job(self, job_scheduler):
        """Test scheduling a job"""
        executed = False
        
        async def job_func():
            nonlocal executed
            executed = True
        
        job_id = await job_scheduler.schedule(
            job_func,
            run_at=datetime.utcnow() + timedelta(seconds=0.1)
        )
        
        assert job_id is not None
        
        # Wait for execution
        await asyncio.sleep(0.2)
        
        assert executed is True
    
    @pytest.mark.asyncio
    async def test_schedule_recurring_job(self, job_scheduler):
        """Test scheduling recurring job"""
        execution_count = 0
        
        async def recurring_job():
            nonlocal execution_count
            execution_count += 1
        
        job_id = await job_scheduler.schedule_recurring(
            recurring_job,
            interval=timedelta(seconds=0.1),
            max_runs=3
        )
        
        assert job_id is not None
        
        # Wait for executions
        await asyncio.sleep(0.5)
        
        # Should have executed multiple times
        assert execution_count >= 2
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, job_scheduler):
        """Test canceling a scheduled job"""
        executed = False
        
        async def job_func():
            nonlocal executed
            executed = True
        
        job_id = await job_scheduler.schedule(
            job_func,
            run_at=datetime.utcnow() + timedelta(seconds=0.5)
        )
        
        # Cancel before execution
        await job_scheduler.cancel(job_id)
        
        # Wait past execution time
        await asyncio.sleep(0.6)
        
        # Should not have executed
        assert executed is False
    
    @pytest.mark.asyncio
    async def test_get_job_status(self, job_scheduler):
        """Test getting job status"""
        async def job_func():
            return "done"
        
        job_id = await job_scheduler.schedule(
            job_func,
            run_at=datetime.utcnow() + timedelta(seconds=0.1)
        )
        
        # Get status
        status = await job_scheduler.get_status(job_id)
        
        assert status is not None
    
    @pytest.mark.asyncio
    async def test_list_scheduled_jobs(self, job_scheduler):
        """Test listing scheduled jobs"""
        async def job_func():
            return "done"
        
        # Schedule multiple jobs
        job1 = await job_scheduler.schedule(
            job_func,
            run_at=datetime.utcnow() + timedelta(seconds=1)
        )
        job2 = await job_scheduler.schedule(
            job_func,
            run_at=datetime.utcnow() + timedelta(seconds=2)
        )
        
        jobs = await job_scheduler.list_jobs()
        
        assert len(jobs) >= 2
        assert job1 in [j.id for j in jobs]
        assert job2 in [j.id for j in jobs]



