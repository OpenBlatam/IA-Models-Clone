"""
Background Tasks API - Advanced Implementation
============================================

Advanced background tasks API with task management and monitoring.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel
from datetime import datetime, timedelta

from ..services import background_service, TaskType, TaskPriority, TaskStatus

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class TaskSubmitRequest(BaseModel):
    """Task submission request model"""
    name: str
    task_type: str = "data_processing"
    priority: str = "normal"
    delay_seconds: Optional[int] = None
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None


class WorkflowExecutionRequest(BaseModel):
    """Workflow execution request model"""
    workflow_id: int
    parameters: Optional[Dict[str, Any]] = None
    priority: str = "normal"


class AIProcessingRequest(BaseModel):
    """AI processing request model"""
    prompt: str
    provider: str = "openai"
    priority: str = "normal"


class NotificationRequest(BaseModel):
    """Notification request model"""
    channel: str
    recipient: str
    subject: str
    message: str
    priority: str = "normal"


class AnalyticsProcessingRequest(BaseModel):
    """Analytics processing request model"""
    data: Dict[str, Any]
    priority: str = "low"


class CleanupRequest(BaseModel):
    """Cleanup request model"""
    cleanup_type: str
    parameters: Optional[Dict[str, Any]] = None
    priority: str = "low"


class TaskResponse(BaseModel):
    """Task response model"""
    task_id: str
    name: str
    status: str
    created_at: str
    message: str


class TaskStatusResponse(BaseModel):
    """Task status response model"""
    id: str
    name: str
    type: str
    status: str
    priority: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    retry_count: int
    max_retries: int
    result: Optional[Any]
    error: Optional[str]
    metadata: Dict[str, Any]


class BackgroundStatsResponse(BaseModel):
    """Background service statistics response model"""
    is_running: bool
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    running_tasks: int
    pending_tasks: int
    tasks_by_type: Dict[str, int]
    tasks_by_priority: Dict[str, int]
    queue_size: int
    scheduled_tasks: int
    worker_pool_size: int
    process_pool_size: int


# Task submission endpoints
@router.post("/tasks/submit", response_model=TaskResponse)
async def submit_task(request: TaskSubmitRequest):
    """Submit a generic background task"""
    try:
        # Convert string enums to actual enums
        task_type = TaskType(request.task_type)
        priority = TaskPriority(request.priority)
        
        # Create delay if specified
        delay = timedelta(seconds=request.delay_seconds) if request.delay_seconds else None
        timeout = timedelta(seconds=request.timeout_seconds) if request.timeout_seconds else None
        
        # Submit task
        task_id = await background_service.submit_task(
            name=request.name,
            function=lambda: {"message": "Generic task executed"},
            task_type=task_type,
            priority=priority,
            delay=delay,
            timeout=timeout,
            max_retries=request.max_retries,
            metadata=request.metadata or {}
        )
        
        return TaskResponse(
            task_id=task_id,
            name=request.name,
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            message="Task submitted successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )


@router.post("/tasks/workflow/execute", response_model=TaskResponse)
async def submit_workflow_execution(request: WorkflowExecutionRequest):
    """Submit workflow execution task"""
    try:
        priority = TaskPriority(request.priority)
        
        task_id = await background_service.submit_workflow_execution(
            workflow_id=request.workflow_id,
            parameters=request.parameters,
            priority=priority
        )
        
        return TaskResponse(
            task_id=task_id,
            name=f"Execute Workflow {request.workflow_id}",
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            message="Workflow execution task submitted successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to submit workflow execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit workflow execution: {str(e)}"
        )


@router.post("/tasks/ai/process", response_model=TaskResponse)
async def submit_ai_processing(request: AIProcessingRequest):
    """Submit AI processing task"""
    try:
        priority = TaskPriority(request.priority)
        
        task_id = await background_service.submit_ai_processing(
            prompt=request.prompt,
            provider=request.provider,
            priority=priority
        )
        
        return TaskResponse(
            task_id=task_id,
            name=f"AI Processing - {request.provider}",
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            message="AI processing task submitted successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to submit AI processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit AI processing: {str(e)}"
        )


@router.post("/tasks/notification/send", response_model=TaskResponse)
async def submit_notification_sending(request: NotificationRequest):
    """Submit notification sending task"""
    try:
        priority = TaskPriority(request.priority)
        
        task_id = await background_service.submit_notification_sending(
            channel=request.channel,
            recipient=request.recipient,
            subject=request.subject,
            message=request.message,
            priority=priority
        )
        
        return TaskResponse(
            task_id=task_id,
            name=f"Send {request.channel} notification to {request.recipient}",
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            message="Notification sending task submitted successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to submit notification sending: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit notification sending: {str(e)}"
        )


@router.post("/tasks/analytics/process", response_model=TaskResponse)
async def submit_analytics_processing(request: AnalyticsProcessingRequest):
    """Submit analytics processing task"""
    try:
        priority = TaskPriority(request.priority)
        
        task_id = await background_service.submit_analytics_processing(
            data=request.data,
            priority=priority
        )
        
        return TaskResponse(
            task_id=task_id,
            name="Analytics Processing",
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            message="Analytics processing task submitted successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to submit analytics processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit analytics processing: {str(e)}"
        )


@router.post("/tasks/cleanup", response_model=TaskResponse)
async def submit_cleanup_task(request: CleanupRequest):
    """Submit cleanup task"""
    try:
        priority = TaskPriority(request.priority)
        
        task_id = await background_service.submit_cleanup_task(
            cleanup_type=request.cleanup_type,
            parameters=request.parameters,
            priority=priority
        )
        
        return TaskResponse(
            task_id=task_id,
            name=f"Cleanup - {request.cleanup_type}",
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            message="Cleanup task submitted successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to submit cleanup task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit cleanup task: {str(e)}"
        )


# Task management endpoints
@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status"""
    try:
        status_data = await background_service.get_task_status(task_id)
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        return TaskStatusResponse(**status_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """Get task result"""
    try:
        result = await background_service.get_task_result(task_id)
        return {
            "task_id": task_id,
            "result": result,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to get task result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task result: {str(e)}"
        )


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel task"""
    try:
        success = await background_service.cancel_task(task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found or cannot be cancelled"
            )
        
        return {
            "task_id": task_id,
            "message": "Task cancelled successfully",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@router.get("/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 100
):
    """List tasks with filtering"""
    try:
        # Convert string parameters to enums
        status_enum = TaskStatus(status) if status else None
        task_type_enum = TaskType(task_type) if task_type else None
        priority_enum = TaskPriority(priority) if priority else None
        
        tasks = await background_service.list_tasks(
            status=status_enum,
            task_type=task_type_enum,
            priority=priority_enum,
            limit=limit
        )
        
        return tasks
    
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )


# Service management endpoints
@router.post("/service/start")
async def start_background_service():
    """Start background service"""
    try:
        await background_service.start()
        return {
            "message": "Background service started successfully",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to start background service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start background service: {str(e)}"
        )


@router.post("/service/stop")
async def stop_background_service():
    """Stop background service"""
    try:
        await background_service.stop()
        return {
            "message": "Background service stopped successfully",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to stop background service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop background service: {str(e)}"
        )


@router.get("/service/stats", response_model=BackgroundStatsResponse)
async def get_background_service_stats():
    """Get background service statistics"""
    try:
        stats = await background_service.get_service_stats()
        return BackgroundStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get background service stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get background service stats: {str(e)}"
        )


# Bulk operations endpoints
@router.post("/tasks/bulk/submit")
async def submit_bulk_tasks(requests: List[TaskSubmitRequest]):
    """Submit multiple tasks in bulk"""
    try:
        results = []
        
        for request in requests:
            try:
                task_type = TaskType(request.task_type)
                priority = TaskPriority(request.priority)
                
                delay = timedelta(seconds=request.delay_seconds) if request.delay_seconds else None
                timeout = timedelta(seconds=request.timeout_seconds) if request.timeout_seconds else None
                
                task_id = await background_service.submit_task(
                    name=request.name,
                    function=lambda: {"message": "Bulk task executed"},
                    task_type=task_type,
                    priority=priority,
                    delay=delay,
                    timeout=timeout,
                    max_retries=request.max_retries,
                    metadata=request.metadata or {}
                )
                
                results.append({
                    "task_id": task_id,
                    "name": request.name,
                    "status": "submitted",
                    "success": True
                })
            
            except Exception as e:
                results.append({
                    "name": request.name,
                    "status": "failed",
                    "error": str(e),
                    "success": False
                })
        
        return {
            "results": results,
            "total_submitted": len([r for r in results if r["success"]]),
            "total_failed": len([r for r in results if not r["success"]]),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to submit bulk tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit bulk tasks: {str(e)}"
        )


@router.post("/tasks/bulk/cancel")
async def cancel_bulk_tasks(task_ids: List[str]):
    """Cancel multiple tasks in bulk"""
    try:
        results = []
        
        for task_id in task_ids:
            try:
                success = await background_service.cancel_task(task_id)
                results.append({
                    "task_id": task_id,
                    "cancelled": success,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "task_id": task_id,
                    "cancelled": False,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "results": results,
            "total_cancelled": len([r for r in results if r["cancelled"]]),
            "total_failed": len([r for r in results if not r["success"]]),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to cancel bulk tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel bulk tasks: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def background_service_health():
    """Background service health check"""
    try:
        stats = await background_service.get_service_stats()
        
        return {
            "service": "background_service",
            "status": "healthy" if stats["is_running"] else "stopped",
            "is_running": stats["is_running"],
            "active_tasks": stats["running_tasks"],
            "pending_tasks": stats["pending_tasks"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Background service health check failed: {e}")
        return {
            "service": "background_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

