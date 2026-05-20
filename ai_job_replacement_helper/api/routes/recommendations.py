"""
Recommendations endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.recommendations import RecommendationService
from models.schemas import UserProfile

router = APIRouter()
recommendation_service = RecommendationService()


@router.get("/skills/{user_id}")
async def get_skill_recommendations(
    user_id: str,
    target_industry: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener recomendaciones de habilidades"""
    try:
        # En producción, esto vendría de la base de datos
        user_skills = []  # TODO: obtener del perfil del usuario
        user_interests = []  # TODO: obtener del perfil del usuario
        
        recommendations = recommendation_service.recommend_skills(
            user_skills=user_skills,
            user_interests=user_interests,
            target_industry=target_industry
        )
        
        return {
            "recommendations": [
                {
                    "skill": rec.skill,
                    "category": rec.category,
                    "priority": rec.priority,
                    "reason": rec.reason,
                    "learning_resources": rec.learning_resources,
                    "estimated_time": rec.estimated_time,
                    "market_demand": rec.market_demand,
                }
                for rec in recommendations
            ],
            "total": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{user_id}")
async def get_job_recommendations(
    user_id: str,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener recomendaciones de trabajos"""
    try:
        # En producción, esto vendría de la base de datos
        user_skills = []  # TODO: obtener del perfil del usuario
        
        recommendations = recommendation_service.recommend_jobs(
            user_skills=user_skills,
            location=location
        )
        
        return {
            "recommendations": [
                {
                    "job_id": rec.job_id,
                    "title": rec.title,
                    "company": rec.company,
                    "match_score": rec.match_score,
                    "match_reasons": rec.match_reasons,
                    "required_skills": rec.required_skills,
                    "missing_skills": rec.missing_skills,
                    "skill_gap_score": rec.skill_gap_score,
                }
                for rec in recommendations
            ],
            "total": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/next-steps/{user_id}")
async def get_next_steps_recommendations(user_id: str) -> Dict[str, Any]:
    """Obtener recomendaciones de próximos pasos"""
    try:
        # En producción, esto vendría de la base de datos
        completed_steps = []  # TODO: obtener del progreso del usuario
        user_skills = []  # TODO: obtener del perfil del usuario
        user_goals = []  # TODO: obtener del perfil del usuario
        
        recommendations = recommendation_service.recommend_next_steps(
            completed_steps=completed_steps,
            user_skills=user_skills,
            user_goals=user_goals
        )
        
        return {
            "recommendations": recommendations,
            "total": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




