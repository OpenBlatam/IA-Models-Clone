"""
API de Analytics

Endpoints para:
- Tracking de eventos
- Estadísticas de uso
- Análisis de funnels
- Cohort analysis
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from datetime import datetime, timedelta

from services.analytics import (
    get_analytics_service,
    EventType,
    AnalyticsEvent
)
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"]
)


@router.post("/track")
async def track_event(
    event_type: str = Body(..., description="Tipo de evento"),
    user_id: Optional[str] = Body(None, description="ID del usuario"),
    session_id: Optional[str] = Body(None, description="ID de sesión"),
    properties: Optional[Dict[str, Any]] = Body(None, description="Propiedades del evento"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Registra un evento de analytics.
    """
    try:
        # Usar user_id del token si está disponible
        if not user_id and current_user:
            user_id = current_user.get("user_id") or current_user.get("sub")
        
        # Validar tipo de evento
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid event type: {event_type}"
            )
        
        analytics_service = get_analytics_service()
        analytics_service.track_event(
            event_type=event_type_enum,
            user_id=user_id,
            session_id=session_id,
            properties=properties
        )
        
        return {
            "message": "Event tracked successfully",
            "event_type": event_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking event: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error tracking event: {str(e)}"
        )


@router.get("/stats")
async def get_analytics_stats(
    days: int = Query(30, ge=1, le=365, description="Número de días"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene estadísticas generales de analytics.
    """
    try:
        analytics_service = get_analytics_service()
        stats = analytics_service.get_stats(days=days)
        return stats
    except Exception as e:
        logger.error(f"Error getting analytics stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )


@router.get("/user/{user_id}/activity")
async def get_user_activity(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="Número de días"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene actividad de un usuario específico.
    """
    try:
        # Verificar permisos (solo el mismo usuario o admin)
        if current_user:
            current_user_id = current_user.get("user_id") or current_user.get("sub")
            user_roles = current_user.get("roles", [])
            if current_user_id != user_id and "admin" not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to view this user's activity"
                )
        
        analytics_service = get_analytics_service()
        activity = analytics_service.get_user_activity(user_id, days=days)
        return activity
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user activity: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user activity: {str(e)}"
        )


@router.get("/funnel")
async def get_funnel(
    steps: str = Query(..., description="Pasos del funnel separados por coma"),
    start_date: Optional[str] = Query(None, description="Fecha de inicio (ISO format)"),
    end_date: Optional[str] = Query(None, description="Fecha de fin (ISO format)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Calcula un funnel de conversión.
    """
    try:
        # Parsear pasos
        step_list = [s.strip() for s in steps.split(",")]
        event_steps = []
        for step in step_list:
            try:
                event_steps.append(EventType(step))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid event type in steps: {step}"
                )
        
        # Parsear fechas
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        analytics_service = get_analytics_service()
        funnel = analytics_service.get_funnel(event_steps, start_date=start, end_date=end)
        return funnel
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating funnel: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating funnel: {str(e)}"
        )


@router.get("/events")
async def get_event_counts(
    start_date: Optional[str] = Query(None, description="Fecha de inicio (ISO format)"),
    end_date: Optional[str] = Query(None, description="Fecha de fin (ISO format)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene conteos de eventos en un rango de fechas.
    """
    try:
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        analytics_service = get_analytics_service()
        counts = analytics_service.get_event_counts(start_date=start, end_date=end)
        
        return {
            "event_counts": counts,
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None
        }
    except Exception as e:
        logger.error(f"Error getting event counts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving event counts: {str(e)}"
        )

