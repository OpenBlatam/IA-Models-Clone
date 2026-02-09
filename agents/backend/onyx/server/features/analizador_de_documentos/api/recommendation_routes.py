"""
Rutas para Recommendation Engine
==================================

Endpoints para motor de recomendaciones.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.recommendation_engine import (
    get_recommendation_engine,
    RecommendationEngine,
    RecommendationType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/recommendations",
    tags=["Recommendation Engine"]
)


class CreateProfileRequest(BaseModel):
    """Request para crear perfil"""
    preferences: Dict[str, Any] = Field(..., description="Preferencias")
    history: List[Dict[str, Any]] = Field(..., description="Historial")


class GenerateRecommendationsRequest(BaseModel):
    """Request para generar recomendaciones"""
    recommendation_type: str = Field("hybrid", description="Tipo")
    num_recommendations: int = Field(10, description="Número")
    context: Optional[Dict[str, Any]] = Field(None, description="Contexto")


@router.post("/users/{user_id}/profile")
async def create_user_profile(
    user_id: str,
    request: CreateProfileRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Crear perfil de usuario"""
    try:
        engine.create_user_profile(
            user_id,
            request.preferences,
            request.history
        )
        
        return {"status": "created", "user_id": user_id}
    except Exception as e:
        logger.error(f"Error creando perfil: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/recommendations")
async def generate_recommendations(
    user_id: str,
    request: GenerateRecommendationsRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Generar recomendaciones"""
    try:
        rec_type = RecommendationType(request.recommendation_type)
        recommendations = engine.generate_recommendations(
            user_id,
            rec_type,
            request.num_recommendations,
            request.context
        )
        
        return {
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "item_id": r.item_id,
                    "item_type": r.item_type,
                    "score": r.score,
                    "reason": r.reason
                }
                for r in recommendations
            ],
            "count": len(recommendations)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generando recomendaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/recommendations/{recommendation_id}/explain")
async def explain_recommendation(
    user_id: str,
    recommendation_id: str,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Explicar recomendación"""
    try:
        explanation = engine.get_recommendation_explanation(user_id, recommendation_id)
        
        return explanation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error explicando recomendación: {e}")
        raise HTTPException(status_code=500, detail=str(e))
