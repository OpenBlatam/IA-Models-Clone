"""
Application Tracker endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.application_tracker import ApplicationTrackerService, ApplicationStatus

router = APIRouter()
tracker_service = ApplicationTrackerService()


@router.post("/create/{user_id}")
async def create_application(
    user_id: str,
    job_id: str,
    job_title: str,
    company: str,
    platform: str,
    cover_letter: Optional[str] = None
) -> Dict[str, Any]:
    """Crear nueva aplicación"""
    try:
        application = tracker_service.create_application(
            user_id, job_id, job_title, company, platform, cover_letter
        )
        return {
            "id": application.id,
            "job_title": application.job_title,
            "company": application.company,
            "status": application.status.value,
            "applied_date": application.applied_date.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_applications(
    user_id: str,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener aplicaciones del usuario"""
    try:
        status_enum = ApplicationStatus(status) if status else None
        applications = tracker_service.get_user_applications(user_id, status_enum)
        return {
            "applications": [
                {
                    "id": app.id,
                    "job_title": app.job_title,
                    "company": app.company,
                    "status": app.status.value,
                    "applied_date": app.applied_date.isoformat(),
                    "next_action": app.next_action,
                    "next_action_date": app.next_action_date.isoformat() if app.next_action_date else None,
                }
                for app in applications
            ],
            "total": len(applications),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-status/{user_id}/{application_id}")
async def update_status(
    user_id: str,
    application_id: str,
    new_status: str,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Actualizar estado de aplicación"""
    try:
        status_enum = ApplicationStatus(new_status)
        application = tracker_service.update_application_status(
            user_id, application_id, status_enum, notes
        )
        return {
            "id": application.id,
            "status": application.status.value,
            "updated_at": application.updated_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-interview/{user_id}/{application_id}")
async def add_interview(
    user_id: str,
    application_id: str,
    interview_date: str,
    interview_type: Optional[str] = None
) -> Dict[str, Any]:
    """Agregar fecha de entrevista"""
    try:
        interview_dt = datetime.fromisoformat(interview_date)
        application = tracker_service.add_interview(
            user_id, application_id, interview_dt, interview_type
        )
        return {
            "id": application.id,
            "status": application.status.value,
            "interview_dates": [d.isoformat() for d in application.interview_dates],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/{user_id}")
async def get_statistics(user_id: str) -> Dict[str, Any]:
    """Obtener estadísticas de aplicaciones"""
    try:
        stats = tracker_service.get_application_statistics(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming-actions/{user_id}")
async def get_upcoming_actions(user_id: str) -> Dict[str, Any]:
    """Obtener próximas acciones pendientes"""
    try:
        actions = tracker_service.get_upcoming_actions(user_id)
        return {
            "actions": actions,
            "total": len(actions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




