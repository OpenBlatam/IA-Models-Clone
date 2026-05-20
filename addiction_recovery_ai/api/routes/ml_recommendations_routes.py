"""
ML recommendations routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.ml_recommendation_service import MLRecommendationService
except ImportError:
    from ...services.ml_recommendation_service import MLRecommendationService

router = APIRouter()

ml_recommendations = MLRecommendationService()


@router.post("/ml-recommendations/get")
async def get_ml_recommendations(
    user_id: str = Body(...),
    user_profile: Dict = Body(...),
    context: Dict = Body(...)
):
    """Obtiene recomendaciones basadas en ML"""
    try:
        recommendations = ml_recommendations.get_ml_recommendations(
            user_id, user_profile, context
        )
        return JSONResponse(content=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recomendaciones: {str(e)}")


@router.post("/ml-recommendations/collaborative")
async def get_collaborative_recommendations(
    user_id: str = Body(...),
    similar_users: List[str] = Body(...)
):
    """Obtiene recomendaciones colaborativas"""
    try:
        recommendations = ml_recommendations.get_collaborative_recommendations(
            user_id, similar_users
        )
        return JSONResponse(content={
            "user_id": user_id,
            "recommendations": recommendations,
            "total": len(recommendations),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recomendaciones: {str(e)}")



