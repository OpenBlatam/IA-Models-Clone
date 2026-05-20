"""
Job Platforms endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.job_platforms import JobPlatformsService, PlatformType

router = APIRouter()
platforms_service = JobPlatformsService()


@router.get("/search/{user_id}")
async def search_jobs(
    user_id: str,
    keywords: str,
    location: Optional[str] = None,
    platforms: Optional[str] = None,
    limit_per_platform: int = 10
) -> Dict[str, Any]:
    """Buscar trabajos en múltiples plataformas"""
    try:
        platforms_list = None
        if platforms:
            platforms_list = [PlatformType(p.strip()) for p in platforms.split(",")]
        
        jobs = platforms_service.search_jobs_across_platforms(
            user_id, keywords, location, platforms_list, limit_per_platform
        )
        
        return {
            "jobs": [
                {
                    "id": job.id,
                    "platform": job.platform.value,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "salary_range": job.salary_range,
                    "match_score": job.match_score,
                    "application_url": job.application_url,
                }
                for job in jobs
            ],
            "total": len(jobs),
            "platforms_searched": [p.value for p in (platforms_list or [])],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_platform_stats() -> Dict[str, Any]:
    """Obtener estadísticas de plataformas"""
    try:
        stats = platforms_service.get_platform_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




