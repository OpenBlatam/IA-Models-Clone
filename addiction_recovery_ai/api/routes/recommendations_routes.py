"""
Recommendations routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.recommendation_service import RecommendationService
except ImportError:
    from ...services.recommendation_service import RecommendationService

router = APIRouter()

recommendations = RecommendationService()


@router.post("/recommendations/personalized")
async def get_personalized_recommendations(
    user_id: str = Body(...),
    context: Dict = Body(...)
):
    """Obtiene recomendaciones personalizadas"""
    try:
        recs = recommendations.get_personalized_recommendations(user_id, context)
        return JSONResponse(content=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recomendaciones: {str(e)}")


@router.get("/recommendations/resources/{user_id}")
async def get_resource_recommendations(
    user_id: str,
    resource_type: Optional[str] = Query(None)
):
    """Obtiene recomendaciones de recursos"""
    try:
        resources = recommendations.get_resource_recommendations(user_id, resource_type)
        return JSONResponse(content={
            "user_id": user_id,
            "resources": resources,
            "total": len(resources),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")



