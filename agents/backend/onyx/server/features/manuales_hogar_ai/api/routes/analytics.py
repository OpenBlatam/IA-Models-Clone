"""
Rutas de Analytics
==================

Endpoints para analytics y reportes avanzados.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.session import get_async_session
from ...services.analytics.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analytics"])


# Dependencies
async def get_db_session() -> AsyncSession:
    """Obtener sesión de base de datos."""
    async for session in get_async_session():
        yield session


# Endpoints
@router.get("/analytics/comprehensive")
async def get_comprehensive_stats(
    days: int = Query(30, ge=1, le=365, description="Días a considerar"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener estadísticas comprehensivas.
    
    - **days**: Días a considerar (1-365)
    """
    try:
        service = AnalyticsService(db)
        stats = await service.get_comprehensive_stats(days=days)
        
        return {
            "success": True,
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.get("/analytics/trends")
async def get_trends(
    days: int = Query(30, ge=1, le=365, description="Días a considerar"),
    interval: str = Query("day", description="Intervalo: day o week"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener tendencias temporales.
    
    - **days**: Días a considerar
    - **interval**: Intervalo (day o week)
    """
    try:
        if interval not in ["day", "week"]:
            raise HTTPException(status_code=400, detail="Interval debe ser 'day' o 'week'")
        
        service = AnalyticsService(db)
        trends = await service.get_trends(days=days, interval=interval)
        
        return {
            "success": True,
            "trends": trends,
            "period_days": days,
            "interval": interval
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo tendencias: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo tendencias: {str(e)}")


@router.get("/analytics/user/{user_id}")
async def get_user_activity(
    user_id: str = Path(..., description="ID del usuario"),
    days: int = Query(30, ge=1, le=365, description="Días a considerar"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener actividad de usuario.
    
    - **user_id**: ID del usuario
    - **days**: Días a considerar
    """
    try:
        service = AnalyticsService(db)
        activity = await service.get_user_activity(user_id=user_id, days=days)
        
        return {
            "success": True,
            "activity": activity
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo actividad: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo actividad: {str(e)}")




