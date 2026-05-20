"""
Health check endpoints
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/")
async def health_check():
    """Health check básico"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Job Replacement Helper",
    }


@router.get("/detailed")
async def detailed_health():
    """Health check detallado"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Job Replacement Helper",
        "version": "1.0.0",
        "components": {
            "gamification": "operational",
            "steps_guide": "operational",
            "linkedin_integration": "operational",
            "recommendations": "operational",
        }
    }




