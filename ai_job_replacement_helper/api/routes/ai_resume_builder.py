"""
AI Resume Builder endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.ai_resume_builder import AIResumeBuilderService, ResumeFormat

router = APIRouter()
resume_service = AIResumeBuilderService()


@router.post("/create/{user_id}")
async def create_resume(
    user_id: str,
    title: str,
    format: str = "ats_friendly",
    target_job: Optional[str] = None
) -> Dict[str, Any]:
    """Crear nuevo CV"""
    try:
        resume_format = ResumeFormat(format)
        resume = resume_service.create_resume(user_id, title, resume_format, target_job)
        return {
            "id": resume.id,
            "user_id": resume.user_id,
            "title": resume.title,
            "format": resume.format.value,
            "target_job": resume.target_job,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-section/{resume_id}")
async def add_section(
    resume_id: str,
    section_type: str,
    content: Dict[str, Any],
    order: Optional[int] = None
) -> Dict[str, Any]:
    """Agregar sección al CV"""
    try:
        section = resume_service.add_section(resume_id, section_type, content, order)
        return {
            "section_type": section.section_type,
            "order": section.order,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/{resume_id}")
async def optimize_for_job(
    resume_id: str,
    job_description: str
) -> Dict[str, Any]:
    """Optimizar CV para un trabajo específico"""
    try:
        result = resume_service.optimize_for_job(resume_id, job_description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate-pdf/{resume_id}")
async def generate_resume_pdf(resume_id: str) -> Dict[str, Any]:
    """Generar PDF del CV"""
    try:
        result = resume_service.generate_resume_pdf(resume_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{resume_id}")
async def get_resume_analysis(resume_id: str) -> Dict[str, Any]:
    """Obtener análisis completo del CV"""
    try:
        analysis = resume_service.get_resume_analysis(resume_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




