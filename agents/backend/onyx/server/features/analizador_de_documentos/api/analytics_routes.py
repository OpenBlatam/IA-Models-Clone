"""
Rutas para Analytics Avanzados
===============================

Endpoints para analytics y métricas de negocio.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_analytics import get_advanced_analytics, AdvancedAnalytics

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/analytics",
    tags=["Analytics"]
)


class TrackEventRequest(BaseModel):
    """Request para registrar evento"""
    event_type: str = Field(..., description="Tipo de evento")
    data: Dict[str, Any] = Field(..., description="Datos del evento")
    user_id: Optional[str] = Field(None, description="ID de usuario")
    session_id: Optional[str] = Field(None, description="ID de sesión")


@router.post("/track")
async def track_event(
    request: TrackEventRequest,
    analytics: AdvancedAnalytics = Depends(get_advanced_analytics)
):
    """Registrar evento de analytics"""
    try:
        analytics.track_event(
            request.event_type,
            request.data,
            request.user_id,
            request.session_id
        )
        
        return {"status": "tracked"}
    except Exception as e:
        logger.error(f"Error registrando evento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/stats")
async def get_event_stats(
    event_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    analytics: AdvancedAnalytics = Depends(get_advanced_analytics)
):
    """Obtener estadísticas de eventos"""
    stats = analytics.get_event_stats(event_type, start_date, end_date)
    return stats


@router.post("/funnel")
async def get_funnel_analysis(
    steps: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    analytics: AdvancedAnalytics = Depends(get_advanced_analytics)
):
    """Análisis de funnel"""
    try:
        funnel = analytics.get_funnel_analysis(steps, start_date, end_date)
        return funnel
    except Exception as e:
        logger.error(f"Error en análisis de funnel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segments")
async def get_user_segments(
    criteria: Dict[str, Any],
    analytics: AdvancedAnalytics = Depends(get_advanced_analytics)
):
    """Obtener segmentos de usuarios"""
    try:
        segments = analytics.get_user_segments(criteria)
        return {"segments": segments, "count": len(segments)}
    except Exception as e:
        logger.error(f"Error obteniendo segmentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cohorts")
async def get_cohort_analysis(
    cohort_period: str = "week",
    analytics: AdvancedAnalytics = Depends(get_advanced_analytics)
):
    """Análisis de cohortes"""
    try:
        cohorts = analytics.get_cohort_analysis(cohort_period)
        return cohorts
    except Exception as e:
        logger.error(f"Error en análisis de cohortes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















