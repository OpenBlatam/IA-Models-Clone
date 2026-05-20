"""
Application Automation endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.application_automation import ApplicationAutomationService

router = APIRouter()
automation_service = ApplicationAutomationService()


@router.post("/create-template/{user_id}")
async def create_template(
    user_id: str,
    name: str,
    resume_id: str,
    cover_letter_template: str,
    default_answers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Crear plantilla de aplicación"""
    try:
        template = automation_service.create_template(
            user_id, name, resume_id, cover_letter_template, default_answers
        )
        return {
            "id": template.id,
            "name": template.name,
            "resume_id": template.resume_id,
            "auto_fill_fields": template.auto_fill_fields,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prepare/{user_id}")
async def prepare_application(
    user_id: str,
    job_id: str,
    job_title: str,
    company: str,
    template_id: Optional[str] = None
) -> Dict[str, Any]:
    """Preparar aplicación automatizada"""
    try:
        application = automation_service.prepare_application(
            user_id, job_id, job_title, company, template_id
        )
        return {
            "id": application.id,
            "job_title": application.job_title,
            "company": application.company,
            "status": application.status.value,
            "auto_fill_data": application.auto_fill_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-fill/{application_id}")
async def auto_fill_application(
    application_id: str,
    form_fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Auto-completar formulario de aplicación"""
    try:
        result = automation_service.auto_fill_application(application_id, form_fields)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit/{application_id}")
async def submit_application(
    application_id: str,
    submission_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Enviar aplicación"""
    try:
        result = automation_service.submit_application(application_id, submission_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-apply/{user_id}")
async def batch_apply(
    user_id: str,
    job_ids: List[str],
    template_id: Optional[str] = None
) -> Dict[str, Any]:
    """Aplicar a múltiples trabajos en batch"""
    try:
        result = automation_service.batch_apply(user_id, job_ids, template_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{application_id}")
async def get_application_status(application_id: str) -> Dict[str, Any]:
    """Obtener estado de aplicación"""
    try:
        status = automation_service.get_application_status(application_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




