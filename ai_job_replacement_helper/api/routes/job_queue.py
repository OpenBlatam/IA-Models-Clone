"""
Job Queue endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.job_queue import JobQueueService, JobPriority

router = APIRouter()
queue_service = JobQueueService()


@router.post("/enqueue")
async def enqueue_job(
    job_type: str,
    payload: Dict[str, Any],
    priority: str = "normal"
) -> Dict[str, Any]:
    """Agregar trabajo a la cola"""
    try:
        priority_enum = JobPriority(priority)
        job = queue_service.enqueue_job(job_type, payload, priority_enum)
        
        return {
            "id": job.id,
            "job_type": job.job_type,
            "status": job.status.value,
            "priority": job.priority.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Obtener estado de trabajo"""
    try:
        status = queue_service.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_queue_stats() -> Dict[str, Any]:
    """Obtener estadísticas de la cola"""
    try:
        stats = queue_service.get_queue_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




