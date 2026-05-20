"""
Jobs endpoints - LinkedIn integration estilo Tinder
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.linkedin_integration import LinkedInIntegrationService, JobAction
from models.schemas import JobSearchRequest, JobSwipeRequest, JobSwipeResponse

router = APIRouter()
linkedin_service = LinkedInIntegrationService()


@router.get("/search/{user_id}")
async def search_jobs(
    user_id: str,
    keywords: str = None,
    location: str = None,
    experience_level: str = None,
    job_type: str = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Buscar trabajos (estilo Tinder - swipe)"""
    try:
        jobs = await linkedin_service.search_jobs(
            user_id=user_id,
            keywords=keywords,
            location=location,
            experience_level=experience_level,
            job_type=job_type,
            limit=limit
        )
        
        # Convertir a dict
        jobs_dict = [
            {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "description": job.description,
                "salary_range": job.salary_range,
                "job_type": job.job_type,
                "posted_date": job.posted_date.isoformat() if job.posted_date else None,
                "application_url": job.application_url,
                "required_skills": job.required_skills,
                "preferred_skills": job.preferred_skills,
                "match_score": job.match_score,
                "match_reasons": job.match_reasons,
            }
            for job in jobs
        ]
        
        return {
            "jobs": jobs_dict,
            "total": len(jobs_dict),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/swipe/{user_id}")
async def swipe_job(
    user_id: str,
    request: JobSwipeRequest
) -> JobSwipeResponse:
    """Hacer swipe (like/dislike/save) en un trabajo"""
    try:
        action = JobAction(request.action)
        result = linkedin_service.swipe_job(user_id, request.job_id, action)
        
        return JobSwipeResponse(
            success=result["success"],
            action=result["action"],
            job_id=result["job_id"],
            timestamp=result["timestamp"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply/{user_id}")
async def apply_to_job(
    user_id: str,
    job_id: str,
    cover_letter: str = None
) -> Dict[str, Any]:
    """Aplicar a un trabajo"""
    try:
        result = linkedin_service.apply_to_job(user_id, job_id, cover_letter)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/saved/{user_id}")
async def get_saved_jobs(user_id: str) -> Dict[str, Any]:
    """Obtener trabajos guardados"""
    try:
        saved_job_ids = linkedin_service.get_saved_jobs(user_id)
        return {
            "saved_jobs": saved_job_ids,
            "total": len(saved_job_ids),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liked/{user_id}")
async def get_liked_jobs(user_id: str) -> Dict[str, Any]:
    """Obtener trabajos que le gustaron al usuario"""
    try:
        liked_job_ids = linkedin_service.get_liked_jobs(user_id)
        return {
            "liked_jobs": liked_job_ids,
            "total": len(liked_job_ids),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/matches/{user_id}")
async def get_matches(user_id: str) -> Dict[str, Any]:
    """Obtener matches (trabajos con interés mutuo)"""
    try:
        matches = linkedin_service.get_job_matches(user_id)
        return {
            "matches": matches,
            "total": len(matches),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/{user_id}")
async def get_statistics(user_id: str) -> Dict[str, Any]:
    """Obtener estadísticas del usuario"""
    try:
        stats = linkedin_service.get_user_statistics(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




