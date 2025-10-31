"""
Scheduler API - Advanced Implementation
======================================

Advanced scheduler API with cron-like scheduling and task management.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime, timedelta

from ..services import scheduler_service, ScheduleType, ScheduleStatus

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class CronScheduleRequest(BaseModel):
    """Cron schedule request model"""
    name: str
    cron_expression: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_runs: Optional[int] = None
    timeout: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class IntervalScheduleRequest(BaseModel):
    """Interval schedule request model"""
    name: str
    interval_seconds: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_runs: Optional[int] = None
    timeout: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class DailyScheduleRequest(BaseModel):
    """Daily schedule request model"""
    name: str
    hour: int
    minute: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_runs: Optional[int] = None
    timeout: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class WeeklyScheduleRequest(BaseModel):
    """Weekly schedule request model"""
    name: str
    day_of_week: int  # 0=Monday, 6=Sunday
    hour: int
    minute: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_runs: Optional[int] = None
    timeout: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class MonthlyScheduleRequest(BaseModel):
    """Monthly schedule request model"""
    name: str
    day_of_month: int
    hour: int
    minute: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_runs: Optional[int] = None
    timeout: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ScheduleResponse(BaseModel):
    """Schedule response model"""
    schedule_id: str
    name: str
    type: str
    status: str
    created_at: str
    next_run: Optional[str]
    message: str


class ScheduleInfoResponse(BaseModel):
    """Schedule info response model"""
    id: str
    name: str
    type: str
    cron_expression: Optional[str]
    interval_seconds: Optional[int]
    status: str
    created_at: str
    last_run: Optional[str]
    next_run: Optional[str]
    run_count: int
    max_runs: Optional[int]
    timeout: Optional[int]
    metadata: Dict[str, Any]


class SchedulerStatsResponse(BaseModel):
    """Scheduler statistics response model"""
    is_running: bool
    total_schedules: int
    active_schedules: int
    paused_schedules: int
    completed_schedules: int
    failed_schedules: int
    total_runs: int
    successful_runs: int
    failed_runs: int
    schedules_by_type: Dict[str, int]
    running_tasks: int


# Schedule creation endpoints
@router.post("/schedules/cron", response_model=ScheduleResponse)
async def create_cron_schedule(request: CronScheduleRequest):
    """Create cron-based schedule"""
    try:
        # Parse start and end times
        start_time = None
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        
        end_time = None
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)
        
        # Create a simple test function
        def test_function():
            return {"message": f"Cron job executed: {request.name}"}
        
        schedule_id = await scheduler_service.add_cron_schedule(
            name=request.name,
            cron_expression=request.cron_expression,
            function=test_function,
            start_time=start_time,
            end_time=end_time,
            max_runs=request.max_runs,
            timeout=request.timeout,
            metadata=request.metadata or {}
        )
        
        return ScheduleResponse(
            schedule_id=schedule_id,
            name=request.name,
            type="cron",
            status="active",
            created_at=datetime.utcnow().isoformat(),
            next_run=None,  # Will be calculated by scheduler
            message="Cron schedule created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create cron schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create cron schedule: {str(e)}"
        )


@router.post("/schedules/interval", response_model=ScheduleResponse)
async def create_interval_schedule(request: IntervalScheduleRequest):
    """Create interval-based schedule"""
    try:
        # Parse start and end times
        start_time = None
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        
        end_time = None
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)
        
        # Create a simple test function
        def test_function():
            return {"message": f"Interval job executed: {request.name}"}
        
        schedule_id = await scheduler_service.add_interval_schedule(
            name=request.name,
            interval_seconds=request.interval_seconds,
            function=test_function,
            start_time=start_time,
            end_time=end_time,
            max_runs=request.max_runs,
            timeout=request.timeout,
            metadata=request.metadata or {}
        )
        
        return ScheduleResponse(
            schedule_id=schedule_id,
            name=request.name,
            type="interval",
            status="active",
            created_at=datetime.utcnow().isoformat(),
            next_run=None,  # Will be calculated by scheduler
            message="Interval schedule created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create interval schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create interval schedule: {str(e)}"
        )


@router.post("/schedules/daily", response_model=ScheduleResponse)
async def create_daily_schedule(request: DailyScheduleRequest):
    """Create daily schedule"""
    try:
        # Parse start and end times
        start_time = None
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        
        end_time = None
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)
        
        # Create a simple test function
        def test_function():
            return {"message": f"Daily job executed: {request.name}"}
        
        schedule_id = await scheduler_service.add_daily_schedule(
            name=request.name,
            hour=request.hour,
            minute=request.minute,
            function=test_function,
            start_time=start_time,
            end_time=end_time,
            max_runs=request.max_runs,
            timeout=request.timeout,
            metadata=request.metadata or {}
        )
        
        return ScheduleResponse(
            schedule_id=schedule_id,
            name=request.name,
            type="daily",
            status="active",
            created_at=datetime.utcnow().isoformat(),
            next_run=None,  # Will be calculated by scheduler
            message="Daily schedule created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create daily schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create daily schedule: {str(e)}"
        )


@router.post("/schedules/weekly", response_model=ScheduleResponse)
async def create_weekly_schedule(request: WeeklyScheduleRequest):
    """Create weekly schedule"""
    try:
        # Parse start and end times
        start_time = None
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        
        end_time = None
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)
        
        # Create a simple test function
        def test_function():
            return {"message": f"Weekly job executed: {request.name}"}
        
        schedule_id = await scheduler_service.add_weekly_schedule(
            name=request.name,
            day_of_week=request.day_of_week,
            hour=request.hour,
            minute=request.minute,
            function=test_function,
            start_time=start_time,
            end_time=end_time,
            max_runs=request.max_runs,
            timeout=request.timeout,
            metadata=request.metadata or {}
        )
        
        return ScheduleResponse(
            schedule_id=schedule_id,
            name=request.name,
            type="weekly",
            status="active",
            created_at=datetime.utcnow().isoformat(),
            next_run=None,  # Will be calculated by scheduler
            message="Weekly schedule created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create weekly schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create weekly schedule: {str(e)}"
        )


@router.post("/schedules/monthly", response_model=ScheduleResponse)
async def create_monthly_schedule(request: MonthlyScheduleRequest):
    """Create monthly schedule"""
    try:
        # Parse start and end times
        start_time = None
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        
        end_time = None
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)
        
        # Create a simple test function
        def test_function():
            return {"message": f"Monthly job executed: {request.name}"}
        
        schedule_id = await scheduler_service.add_monthly_schedule(
            name=request.name,
            day_of_month=request.day_of_month,
            hour=request.hour,
            minute=request.minute,
            function=test_function,
            start_time=start_time,
            end_time=end_time,
            max_runs=request.max_runs,
            timeout=request.timeout,
            metadata=request.metadata or {}
        )
        
        return ScheduleResponse(
            schedule_id=schedule_id,
            name=request.name,
            type="monthly",
            status="active",
            created_at=datetime.utcnow().isoformat(),
            next_run=None,  # Will be calculated by scheduler
            message="Monthly schedule created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create monthly schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create monthly schedule: {str(e)}"
        )


# Schedule management endpoints
@router.get("/schedules/{schedule_id}", response_model=ScheduleInfoResponse)
async def get_schedule(schedule_id: str):
    """Get schedule information"""
    try:
        schedule_info = await scheduler_service.get_schedule(schedule_id)
        if not schedule_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule not found"
            )
        
        return ScheduleInfoResponse(**schedule_info)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get schedule: {str(e)}"
        )


@router.get("/schedules", response_model=List[Dict[str, Any]])
async def list_schedules(
    status: Optional[str] = None,
    schedule_type: Optional[str] = None,
    limit: int = 100
):
    """List schedules with filtering"""
    try:
        # Convert string parameters to enums
        status_enum = ScheduleStatus(status) if status else None
        type_enum = ScheduleType(schedule_type) if schedule_type else None
        
        schedules = await scheduler_service.list_schedules(
            status=status_enum,
            schedule_type=type_enum,
            limit=limit
        )
        
        return schedules
    
    except Exception as e:
        logger.error(f"Failed to list schedules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list schedules: {str(e)}"
        )


@router.post("/schedules/{schedule_id}/pause")
async def pause_schedule(schedule_id: str):
    """Pause schedule"""
    try:
        success = await scheduler_service.pause_schedule(schedule_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule not found"
            )
        
        return {
            "schedule_id": schedule_id,
            "message": "Schedule paused successfully",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause schedule: {str(e)}"
        )


@router.post("/schedules/{schedule_id}/resume")
async def resume_schedule(schedule_id: str):
    """Resume schedule"""
    try:
        success = await scheduler_service.resume_schedule(schedule_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule not found"
            )
        
        return {
            "schedule_id": schedule_id,
            "message": "Schedule resumed successfully",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume schedule: {str(e)}"
        )


@router.post("/schedules/{schedule_id}/cancel")
async def cancel_schedule(schedule_id: str):
    """Cancel schedule"""
    try:
        success = await scheduler_service.cancel_schedule(schedule_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule not found"
            )
        
        return {
            "schedule_id": schedule_id,
            "message": "Schedule cancelled successfully",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel schedule: {str(e)}"
        )


# Service management endpoints
@router.post("/service/start")
async def start_scheduler_service():
    """Start scheduler service"""
    try:
        await scheduler_service.start()
        return {
            "message": "Scheduler service started successfully",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to start scheduler service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start scheduler service: {str(e)}"
        )


@router.post("/service/stop")
async def stop_scheduler_service():
    """Stop scheduler service"""
    try:
        await scheduler_service.stop()
        return {
            "message": "Scheduler service stopped successfully",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to stop scheduler service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop scheduler service: {str(e)}"
        )


@router.get("/service/stats", response_model=SchedulerStatsResponse)
async def get_scheduler_stats():
    """Get scheduler service statistics"""
    try:
        stats = await scheduler_service.get_scheduler_stats()
        return SchedulerStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get scheduler stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scheduler stats: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def scheduler_health():
    """Scheduler service health check"""
    try:
        stats = await scheduler_service.get_scheduler_stats()
        
        return {
            "service": "scheduler_service",
            "status": "healthy" if stats["is_running"] else "stopped",
            "is_running": stats["is_running"],
            "active_schedules": stats["active_schedules"],
            "total_schedules": stats["total_schedules"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Scheduler service health check failed: {e}")
        return {
            "service": "scheduler_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

