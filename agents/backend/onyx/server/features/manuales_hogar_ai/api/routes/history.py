"""
Rutas de Historial y Estadísticas
==================================

Endpoints para consultar historial de manuales y estadísticas.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

from ...database.session import get_async_session
from ...services.manual.manual_service import ManualService
from ...services.cache.cache_service import CacheService
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["historial"])


# Modelos Pydantic
class ManualListItem(BaseModel):
    """Item de lista de manual."""
    id: int
    problem_description: str
    category: str
    model_used: Optional[str]
    tokens_used: int
    images_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class ManualDetailResponse(BaseModel):
    """Response detallado de manual."""
    id: int
    problem_description: str
    category: str
    manual_content: str
    model_used: Optional[str]
    tokens_used: int
    image_analysis: Optional[str]
    detected_category: Optional[str]
    images_count: int
    format: str
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class StatisticsResponse(BaseModel):
    """Response de estadísticas."""
    total_manuals: int
    category_stats: dict
    total_tokens: int
    top_models: List[dict]
    period_days: int


# Dependencies
async def get_db_session() -> AsyncSession:
    """Obtener sesión de base de datos."""
    async for session in get_async_session():
        yield session


# Endpoints
@router.get("/manuals", response_model=List[ManualListItem])
async def list_manuals(
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    search: Optional[str] = Query(None, description="Buscar en descripciones"),
    difficulty: Optional[str] = Query(None, description="Filtrar por dificultad"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Rating mínimo"),
    max_rating: Optional[float] = Query(None, ge=0, le=5, description="Rating máximo"),
    limit: int = Query(20, ge=1, le=100, description="Límite de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Listar manuales generados.
    
    - **category**: Filtrar por categoría (opcional)
    - **search**: Buscar término en descripciones (opcional)
    - **limit**: Número máximo de resultados (1-100)
    - **offset**: Offset para paginación
    """
    try:
        service = ManualService(db)
        manuals = await service.search_manuals(
            category=category,
            search_term=search,
            difficulty=difficulty,
            min_rating=min_rating,
            max_rating=max_rating,
            limit=limit,
            offset=offset
        )
        
        return [ManualListItem.from_orm(m) for m in manuals]
    
    except Exception as e:
        logger.error(f"Error listando manuales: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listando manuales: {str(e)}")


@router.get("/manuals/{manual_id}", response_model=ManualDetailResponse)
async def get_manual(
    manual_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener manual por ID.
    
    - **manual_id**: ID del manual
    """
    try:
        service = ManualService(db)
        manual = await service.get_manual_by_id(manual_id)
        
        if not manual:
            raise HTTPException(status_code=404, detail="Manual no encontrado")
        
        return ManualDetailResponse.from_orm(manual)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo manual {manual_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo manual: {str(e)}")


@router.get("/manuals/category/{category}", response_model=List[ManualListItem])
async def get_manuals_by_category(
    category: str,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener manuales por categoría.
    
    - **category**: Categoría del oficio
    - **limit**: Número máximo de resultados
    """
    try:
        service = ManualService(db)
        manuals = await service.get_manuals_by_category(category, limit=limit)
        
        return [ManualListItem.from_orm(m) for m in manuals]
    
    except Exception as e:
        logger.error(f"Error obteniendo manuales por categoría: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo manuales: {str(e)}")


@router.get("/manuals/recent", response_model=List[ManualListItem])
async def get_recent_manuals(
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener manuales recientes.
    
    - **limit**: Número de manuales (1-50)
    - **category**: Filtrar por categoría (opcional)
    """
    try:
        service = ManualService(db)
        manuals = await service.get_recent_manuals(limit=limit, category=category)
        
        return [ManualListItem.from_orm(m) for m in manuals]
    
    except Exception as e:
        logger.error(f"Error obteniendo manuales recientes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo manuales: {str(e)}")


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    days: int = Query(30, ge=1, le=365, description="Número de días a considerar"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener estadísticas de uso.
    
    - **days**: Número de días a considerar (1-365)
    """
    try:
        service = ManualService(db)
        stats = await service.get_statistics(days=days)
        
        return StatisticsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.get("/cache/stats-db")
async def get_cache_stats_db(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener estadísticas del cache persistente.
    """
    try:
        service = CacheService(db)
        stats = await service.get_stats()
        
        return {
            "success": True,
            "cache_stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo stats del cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/clear-db")
async def clear_cache_db(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Limpiar cache persistente.
    """
    try:
        service = CacheService(db)
        deleted = await service.clear_all()
        
        return {
            "success": True,
            "message": f"Cache limpiado: {deleted} entradas eliminadas",
            "deleted_count": deleted
        }
    
    except Exception as e:
        logger.error(f"Error limpiando cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/cleanup-expired")
async def cleanup_expired_cache(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Limpiar entradas expiradas del cache.
    """
    try:
        service = CacheService(db)
        deleted = await service.clear_expired()
        
        return {
            "success": True,
            "message": f"Entradas expiradas eliminadas: {deleted}",
            "deleted_count": deleted
        }
    
    except Exception as e:
        logger.error(f"Error limpiando cache expirado: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

