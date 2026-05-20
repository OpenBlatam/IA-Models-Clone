"""
Analytics Routes - Rutas para analytics y análisis de datos.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services.analytics_service import AnalyticsService
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class TrackEventRequest(BaseModel):
    """Request para trackear evento."""
    event_type: str = Field(..., min_length=1, max_length=100)
    user_id: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)
    properties: Optional[Dict[str, Any]] = Field(None)


def get_analytics_service() -> AnalyticsService:
    """Obtener servicio de analytics."""
    try:
        return get_service("analytics_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Analytics service no disponible")


@router.post("/events")
@handle_api_errors
async def track_event(
    request: TrackEventRequest,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Trackear evento de analytics.
    
    Args:
        request: Datos del evento
        
    Returns:
        Confirmación
    """
    analytics_service.track_event(
        event_type=request.event_type,
        user_id=request.user_id,
        session_id=request.session_id,
        properties=request.properties
    )
    
    return {"message": "Event tracked", "event_type": request.event_type}


@router.get("/events")
@handle_api_errors
async def get_events(
    event_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=1000),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Obtener eventos de analytics.
    
    Args:
        event_type: Filtrar por tipo (opcional)
        user_id: Filtrar por usuario (opcional)
        start_date: Fecha de inicio ISO (opcional)
        end_date: Fecha de fin ISO (opcional)
        limit: Límite de resultados (opcional)
        
    Returns:
        Lista de eventos
    """
    start = None
    if start_date:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    
    end = None
    if end_date:
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    
    events = analytics_service.get_events(
        event_type=event_type,
        user_id=user_id,
        start_date=start,
        end_date=end,
        limit=limit
    )
    
    return {
        "total": len(events),
        "events": [
            {
                "event_type": e.event_type,
                "user_id": e.user_id,
                "session_id": e.session_id,
                "properties": e.properties,
                "timestamp": e.timestamp.isoformat()
            }
            for e in events
        ]
    }


@router.get("/events/counts")
@handle_api_errors
async def get_event_counts(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Obtener conteos de eventos por tipo.
    
    Args:
        start_date: Fecha de inicio ISO (opcional)
        end_date: Fecha de fin ISO (opcional)
        
    Returns:
        Conteos por tipo
    """
    start = None
    if start_date:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    
    end = None
    if end_date:
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    
    counts = analytics_service.get_event_counts(start_date=start, end_date=end)
    return {"counts": counts}


@router.get("/users/{user_id}/activity")
@handle_api_errors
async def get_user_activity(
    user_id: str,
    days: int = Query(7, ge=1, le=365),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Obtener actividad de un usuario.
    
    Args:
        user_id: ID del usuario
        days: Número de días hacia atrás
        
    Returns:
        Actividad del usuario
    """
    activity = analytics_service.get_user_activity(user_id, days=days)
    return activity


@router.get("/stats")
@handle_api_errors
async def get_analytics_stats(
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Obtener estadísticas de analytics.
    
    Returns:
        Estadísticas
    """
    return analytics_service.get_stats()


@router.post("/events/cleanup")
@handle_api_errors
async def cleanup_old_events(
    days: int = Query(30, ge=1, le=365),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Limpiar eventos antiguos.
    
    Args:
        days: Días hacia atrás para mantener
        
    Returns:
        Resultado de limpieza
    """
    removed = analytics_service.clear_old_events(days=days)
    return {
        "message": f"Removed {removed} old events",
        "removed_count": removed,
        "days_kept": days
    }



