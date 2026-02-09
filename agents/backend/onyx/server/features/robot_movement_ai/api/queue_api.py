"""
Queue API Endpoints
===================

Endpoints para message queue y job queue.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional
import logging

from ..core.message_queue import (
    get_message_queue_manager,
    MessagePriority
)
from ..core.job_queue import (
    get_job_queue_manager,
    JobStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/queues", tags=["queues"])


@router.post("/message-queues/{queue_name}/messages")
async def enqueue_message(
    queue_name: str,
    payload: Dict[str, Any] = Body(...),
    priority: str = "normal",
    max_attempts: int = 3
) -> Dict[str, Any]:
    """Agregar mensaje a la cola."""
    try:
        manager = get_message_queue_manager()
        queue = manager.get_queue(queue_name)
        
        if not queue:
            queue = manager.create_queue(queue_name)
        
        priority_enum = MessagePriority[priority.upper()]
        message = await queue.enqueue(
            payload=payload,
            priority=priority_enum,
            max_attempts=max_attempts
        )
        
        return {
            "message_id": message.message_id,
            "queue_name": message.queue_name,
            "priority": message.priority.value
        }
    except Exception as e:
        logger.error(f"Error enqueuing message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/message-queues/{queue_name}/statistics")
async def get_message_queue_statistics(queue_name: str) -> Dict[str, Any]:
    """Obtener estadísticas de cola de mensajes."""
    try:
        manager = get_message_queue_manager()
        queue = manager.get_queue(queue_name)
        
        if not queue:
            raise HTTPException(status_code=404, detail="Queue not found")
        
        stats = queue.get_statistics()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting queue statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/job-queues/{queue_name}/jobs")
async def enqueue_job(
    queue_name: str,
    job_type: str,
    payload: Dict[str, Any] = Body(...),
    priority: int = 5,
    max_attempts: int = 3
) -> Dict[str, Any]:
    """Agregar trabajo a la cola."""
    try:
        manager = get_job_queue_manager()
        queue = manager.get_queue(queue_name)
        
        if not queue:
            queue = manager.create_queue(queue_name)
        
        job = await queue.enqueue(
            job_type=job_type,
            payload=payload,
            priority=priority,
            max_attempts=max_attempts
        )
        
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "status": job.status.value,
            "priority": job.priority
        }
    except Exception as e:
        logger.error(f"Error enqueuing job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job-queues/{queue_name}/jobs/{job_id}")
async def get_job(
    queue_name: str,
    job_id: str
) -> Dict[str, Any]:
    """Obtener trabajo."""
    try:
        manager = get_job_queue_manager()
        queue = manager.get_queue(queue_name)
        
        if not queue:
            raise HTTPException(status_code=404, detail="Queue not found")
        
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "status": job.status.value,
            "priority": job.priority,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "result": job.result,
            "error": job.error,
            "attempts": job.attempts
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job-queues/{queue_name}/statistics")
async def get_job_queue_statistics(queue_name: str) -> Dict[str, Any]:
    """Obtener estadísticas de cola de trabajos."""
    try:
        manager = get_job_queue_manager()
        queue = manager.get_queue(queue_name)
        
        if not queue:
            raise HTTPException(status_code=404, detail="Queue not found")
        
        stats = queue.get_statistics()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job queue statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






