"""
Rutas de Ratings y Favoritos
=============================

Endpoints para gestionar ratings y favoritos de manuales.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import Optional, List
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.session import get_async_session
from ...services.rating.rating_service import RatingService
from ...services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ratings"])


# Modelos Pydantic
class RatingRequest(BaseModel):
    """Request para agregar rating."""
    rating: int = Field(..., ge=1, le=5, description="Rating de 1 a 5")
    comment: Optional[str] = Field(None, description="Comentario opcional")


class RatingResponse(BaseModel):
    """Response de rating."""
    id: int
    manual_id: int
    user_id: Optional[str]
    rating: int
    comment: Optional[str]
    created_at: str
    
    class Config:
        from_attributes = True


class FavoriteResponse(BaseModel):
    """Response de favorito."""
    id: int
    manual_id: int
    user_id: str
    created_at: str
    
    class Config:
        from_attributes = True


# Dependencies
async def get_db_session() -> AsyncSession:
    """Obtener sesión de base de datos."""
    async for session in get_async_session():
        yield session


# Endpoints
@router.post("/manuals/{manual_id}/rating", response_model=RatingResponse)
async def add_rating(
    manual_id: int = Path(..., description="ID del manual"),
    request: RatingRequest = ...,
    user_id: Optional[str] = Query(None, description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Agregar o actualizar rating de un manual.
    
    - **manual_id**: ID del manual
    - **rating**: Rating de 1 a 5
    - **comment**: Comentario opcional
    - **user_id**: ID del usuario (opcional)
    """
    try:
        service = RatingService(db)
        rating = await service.add_rating(
            manual_id=manual_id,
            rating=request.rating,
            user_id=user_id,
            comment=request.comment
        )
        
        return RatingResponse(
            id=rating.id,
            manual_id=rating.manual_id,
            user_id=rating.user_id,
            rating=rating.rating,
            comment=rating.comment,
            created_at=rating.created_at.isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error agregando rating: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error agregando rating: {str(e)}")


@router.get("/manuals/{manual_id}/ratings", response_model=List[RatingResponse])
async def get_ratings(
    manual_id: int = Path(..., description="ID del manual"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener ratings de un manual.
    
    - **manual_id**: ID del manual
    - **limit**: Límite de resultados
    - **offset**: Offset para paginación
    """
    try:
        service = RatingService(db)
        ratings = await service.get_ratings(manual_id, limit=limit, offset=offset)
        
        return [
            RatingResponse(
                id=r.id,
                manual_id=r.manual_id,
                user_id=r.user_id,
                rating=r.rating,
                comment=r.comment,
                created_at=r.created_at.isoformat()
            )
            for r in ratings
        ]
    
    except Exception as e:
        logger.error(f"Error obteniendo ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo ratings: {str(e)}")


@router.get("/manuals/{manual_id}/rating/user/{user_id}", response_model=Optional[RatingResponse])
async def get_user_rating(
    manual_id: int = Path(..., description="ID del manual"),
    user_id: str = Path(..., description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener rating de un usuario específico.
    
    - **manual_id**: ID del manual
    - **user_id**: ID del usuario
    """
    try:
        service = RatingService(db)
        rating = await service.get_user_rating(manual_id, user_id)
        
        if not rating:
            return None
        
        return RatingResponse(
            id=rating.id,
            manual_id=rating.manual_id,
            user_id=rating.user_id,
            rating=rating.rating,
            comment=rating.comment,
            created_at=rating.created_at.isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error obteniendo rating de usuario: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo rating: {str(e)}")


@router.post("/manuals/{manual_id}/favorite", response_model=FavoriteResponse)
async def add_favorite(
    manual_id: int = Path(..., description="ID del manual"),
    user_id: str = Query(..., description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Agregar manual a favoritos.
    
    - **manual_id**: ID del manual
    - **user_id**: ID del usuario
    """
    try:
        service = RatingService(db)
        favorite = await service.add_favorite(manual_id, user_id)
        
        return FavoriteResponse(
            id=favorite.id,
            manual_id=favorite.manual_id,
            user_id=favorite.user_id,
            created_at=favorite.created_at.isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error agregando favorito: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error agregando favorito: {str(e)}")


@router.delete("/manuals/{manual_id}/favorite")
async def remove_favorite(
    manual_id: int = Path(..., description="ID del manual"),
    user_id: str = Query(..., description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Remover manual de favoritos.
    
    - **manual_id**: ID del manual
    - **user_id**: ID del usuario
    """
    try:
        service = RatingService(db)
        removed = await service.remove_favorite(manual_id, user_id)
        
        if not removed:
            raise HTTPException(status_code=404, detail="Favorito no encontrado")
        
        return {
            "success": True,
            "message": "Favorito removido exitosamente"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removiendo favorito: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removiendo favorito: {str(e)}")


@router.get("/users/{user_id}/favorites")
async def get_user_favorites(
    user_id: str = Path(..., description="ID del usuario"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener favoritos de un usuario.
    
    - **user_id**: ID del usuario
    - **limit**: Límite de resultados
    - **offset**: Offset para paginación
    """
    try:
        service = RatingService(db)
        favorites = await service.get_user_favorites(user_id, limit=limit, offset=offset)
        
        from ...api.routes.history import ManualListItem
        
        return [ManualListItem.from_orm(m) for m in favorites]
    
    except Exception as e:
        logger.error(f"Error obteniendo favoritos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo favoritos: {str(e)}")


@router.get("/manuals/{manual_id}/favorite/check")
async def check_favorite(
    manual_id: int = Path(..., description="ID del manual"),
    user_id: str = Query(..., description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Verificar si un manual está en favoritos.
    
    - **manual_id**: ID del manual
    - **user_id**: ID del usuario
    """
    try:
        service = RatingService(db)
        is_fav = await service.is_favorite(manual_id, user_id)
        
        return {
            "is_favorite": is_fav,
            "manual_id": manual_id,
            "user_id": user_id
        }
    
    except Exception as e:
        logger.error(f"Error verificando favorito: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error verificando favorito: {str(e)}")


@router.get("/recommendations/popular")
async def get_popular_manuals(
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener manuales populares.
    
    - **category**: Filtrar por categoría (opcional)
    - **limit**: Número de manuales
    """
    try:
        service = RecommendationService(db)
        manuals = await service.get_popular_manuals(category=category, limit=limit)
        
        from ...api.routes.history import ManualListItem
        
        return [ManualListItem.from_orm(m) for m in manuals]
    
    except Exception as e:
        logger.error(f"Error obteniendo manuales populares: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo manuales: {str(e)}")


@router.get("/recommendations/top-rated")
async def get_top_rated_manuals(
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    limit: int = Query(10, ge=1, le=50),
    min_ratings: int = Query(3, ge=1, description="Mínimo de ratings"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener manuales mejor calificados.
    
    - **category**: Filtrar por categoría (opcional)
    - **limit**: Número de manuales
    - **min_ratings**: Mínimo de ratings requeridos
    """
    try:
        service = RecommendationService(db)
        manuals = await service.get_top_rated_manuals(
            category=category,
            limit=limit,
            min_ratings=min_ratings
        )
        
        from ...api.routes.history import ManualListItem
        
        return [ManualListItem.from_orm(m) for m in manuals]
    
    except Exception as e:
        logger.error(f"Error obteniendo manuales mejor calificados: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo manuales: {str(e)}")


@router.get("/recommendations/similar/{manual_id}")
async def get_similar_manuals(
    manual_id: int = Path(..., description="ID del manual"),
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener manuales similares.
    
    - **manual_id**: ID del manual de referencia
    - **limit**: Número de recomendaciones
    """
    try:
        service = RecommendationService(db)
        manuals = await service.get_similar_manuals(manual_id, limit=limit)
        
        from ...api.routes.history import ManualListItem
        
        return [ManualListItem.from_orm(m) for m in manuals]
    
    except Exception as e:
        logger.error(f"Error obteniendo manuales similares: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo manuales: {str(e)}")


@router.get("/recommendations/trending")
async def get_trending_manuals(
    days: int = Query(7, ge=1, le=30, description="Días a considerar"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener manuales en tendencia.
    
    - **days**: Días a considerar
    - **limit**: Número de manuales
    """
    try:
        service = RecommendationService(db)
        manuals = await service.get_trending_manuals(days=days, limit=limit)
        
        from ...api.routes.history import ManualListItem
        
        return [ManualListItem.from_orm(m) for m in manuals]
    
    except Exception as e:
        logger.error(f"Error obteniendo manuales en tendencia: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo manuales: {str(e)}")




