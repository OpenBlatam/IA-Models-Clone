"""
API de Recomendaciones

Endpoints para:
- Recomendaciones basadas en contenido
- Recomendaciones colaborativas
- Recomendaciones híbridas
- Trending y popular
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

from services.recommendation_engine import get_recommendation_engine
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"]
)


@router.get("/content-based")
async def get_content_based_recommendations(
    user_id: Optional[str] = Query(None, description="ID del usuario"),
    limit: int = Query(10, ge=1, le=50, description="Número de recomendaciones"),
    min_similarity: float = Query(0.3, ge=0.0, le=1.0, description="Similitud mínima"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene recomendaciones basadas en contenido.
    """
    try:
        # Usar user_id del token si está disponible
        if not user_id and current_user:
            user_id = current_user.get("user_id") or current_user.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required"
            )
        
        engine = get_recommendation_engine()
        recommendations = engine.get_content_based_recommendations(
            user_id=user_id,
            limit=limit,
            min_similarity=min_similarity
        )
        
        return {
            "user_id": user_id,
            "type": "content_based",
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content-based recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recommendations: {str(e)}"
        )


@router.get("/collaborative")
async def get_collaborative_recommendations(
    user_id: Optional[str] = Query(None, description="ID del usuario"),
    limit: int = Query(10, ge=1, le=50, description="Número de recomendaciones"),
    min_users: int = Query(2, ge=1, description="Mínimo de usuarios"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene recomendaciones colaborativas.
    """
    try:
        if not user_id and current_user:
            user_id = current_user.get("user_id") or current_user.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required"
            )
        
        engine = get_recommendation_engine()
        recommendations = engine.get_collaborative_recommendations(
            user_id=user_id,
            limit=limit,
            min_users=min_users
        )
        
        return {
            "user_id": user_id,
            "type": "collaborative",
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collaborative recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recommendations: {str(e)}"
        )


@router.get("/hybrid")
async def get_hybrid_recommendations(
    user_id: Optional[str] = Query(None, description="ID del usuario"),
    limit: int = Query(10, ge=1, le=50, description="Número de recomendaciones"),
    content_weight: float = Query(0.5, ge=0.0, le=1.0, description="Peso de contenido"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene recomendaciones híbridas (contenido + colaborativo).
    """
    try:
        if not user_id and current_user:
            user_id = current_user.get("user_id") or current_user.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required"
            )
        
        engine = get_recommendation_engine()
        recommendations = engine.get_hybrid_recommendations(
            user_id=user_id,
            limit=limit,
            content_weight=content_weight
        )
        
        return {
            "user_id": user_id,
            "type": "hybrid",
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting hybrid recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recommendations: {str(e)}"
        )


@router.get("/trending")
async def get_trending(
    limit: int = Query(10, ge=1, le=50, description="Número de items"),
    hours: int = Query(24, ge=1, le=168, description="Ventana de tiempo en horas")
) -> Dict[str, Any]:
    """
    Obtiene items trending.
    """
    try:
        engine = get_recommendation_engine()
        trending = engine.get_trending(limit=limit, hours=hours)
        
        return {
            "type": "trending",
            "hours": hours,
            "items": trending,
            "count": len(trending)
        }
    except Exception as e:
        logger.error(f"Error getting trending: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting trending: {str(e)}"
        )


@router.get("/popular")
async def get_popular(
    limit: int = Query(10, ge=1, le=50, description="Número de items")
) -> Dict[str, Any]:
    """
    Obtiene items populares.
    """
    try:
        engine = get_recommendation_engine()
        popular = engine.get_popular(limit=limit)
        
        return {
            "type": "popular",
            "items": popular,
            "count": len(popular)
        }
    except Exception as e:
        logger.error(f"Error getting popular: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting popular: {str(e)}"
        )


@router.post("/interaction")
async def record_interaction(
    user_id: str,
    item_id: str,
    interaction_type: str = Query("view", description="Tipo de interacción"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Registra una interacción usuario-item.
    """
    try:
        engine = get_recommendation_engine()
        engine.record_interaction(
            user_id=user_id,
            item_id=item_id,
            interaction_type=interaction_type
        )
        
        return {
            "message": "Interaction recorded successfully",
            "user_id": user_id,
            "item_id": item_id,
            "type": interaction_type
        }
    except Exception as e:
        logger.error(f"Error recording interaction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recording interaction: {str(e)}"
        )
