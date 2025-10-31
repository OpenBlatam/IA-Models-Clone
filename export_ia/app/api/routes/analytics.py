"""
Analytics API Routes - Rutas API para analytics de negocio
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from ..analytics.business_analytics import BusinessAnalytics, AnalyticsEventType, MetricType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])

# Instancia global del sistema de analytics
analytics_manager = BusinessAnalytics()


# Modelos Pydantic
class TrackEventRequest(BaseModel):
    event_type: str
    name: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrackMetricRequest(BaseModel):
    name: str
    value: float
    metric_type: str = "gauge"
    dimensions: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StartSessionRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class EndSessionRequest(BaseModel):
    session_id: str


# Rutas de Analytics
@router.post("/events/track")
async def track_event(request: TrackEventRequest):
    """Rastrear un evento de analytics."""
    try:
        event_type = AnalyticsEventType(request.event_type)
        
        event_id = await analytics_manager.track_event(
            event_type=event_type,
            name=request.name,
            user_id=request.user_id,
            session_id=request.session_id,
            properties=request.properties,
            metrics=request.metrics,
            tags=request.tags,
            metadata=request.metadata
        )
        
        return {
            "event_id": event_id,
            "success": True,
            "message": "Evento rastreado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Tipo de evento inválido: {e}")
    except Exception as e:
        logger.error(f"Error al rastrear evento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/track")
async def track_metric(request: TrackMetricRequest):
    """Rastrear una métrica de analytics."""
    try:
        metric_type = MetricType(request.metric_type)
        
        metric_id = await analytics_manager.track_metric(
            name=request.name,
            value=request.value,
            metric_type=metric_type,
            dimensions=request.dimensions,
            metadata=request.metadata
        )
        
        return {
            "metric_id": metric_id,
            "success": True,
            "message": "Métrica rastreada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Tipo de métrica inválido: {e}")
    except Exception as e:
        logger.error(f"Error al rastrear métrica: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/start")
async def start_session(request: StartSessionRequest):
    """Iniciar sesión de usuario."""
    try:
        session_id = await analytics_manager.start_user_session(
            session_id=request.session_id,
            user_id=request.user_id,
            properties=request.properties
        )
        
        return {
            "session_id": session_id,
            "success": True,
            "message": "Sesión iniciada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al iniciar sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/end")
async def end_session(request: EndSessionRequest):
    """Finalizar sesión de usuario."""
    try:
        success = await analytics_manager.end_user_session(request.session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        return {
            "success": True,
            "message": "Sesión finalizada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al finalizar sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_events(
    event_type: Optional[str] = Query(None, description="Tipo de evento"),
    user_id: Optional[str] = Query(None, description="ID de usuario"),
    session_id: Optional[str] = Query(None, description="ID de sesión"),
    start_time: Optional[datetime] = Query(None, description="Tiempo de inicio"),
    end_time: Optional[datetime] = Query(None, description="Tiempo de fin"),
    limit: int = Query(100, description="Límite de resultados")
):
    """Obtener eventos de analytics."""
    try:
        event_type_enum = AnalyticsEventType(event_type) if event_type else None
        
        events = await analytics_manager.get_events(
            event_type=event_type_enum,
            user_id=user_id,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return {
            "events": events,
            "count": len(events),
            "filters": {
                "event_type": event_type,
                "user_id": user_id,
                "session_id": session_id,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "limit": limit
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parámetro inválido: {e}")
    except Exception as e:
        logger.error(f"Error al obtener eventos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(
    metric_name: Optional[str] = Query(None, description="Nombre de métrica"),
    start_time: Optional[datetime] = Query(None, description="Tiempo de inicio"),
    end_time: Optional[datetime] = Query(None, description="Tiempo de fin"),
    limit: int = Query(100, description="Límite de resultados")
):
    """Obtener métricas de analytics."""
    try:
        metrics = await analytics_manager.get_metrics(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return {
            "metrics": metrics,
            "count": len(metrics),
            "filters": {
                "metric_name": metric_name,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "limit": limit
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_analytics_summary(
    start_time: Optional[datetime] = Query(None, description="Tiempo de inicio"),
    end_time: Optional[datetime] = Query(None, description="Tiempo de fin")
):
    """Obtener resumen de analytics."""
    try:
        summary = await analytics_manager.get_analytics_summary(
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener resumen de analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}")
async def get_user_analytics(
    user_id: str,
    start_time: Optional[datetime] = Query(None, description="Tiempo de inicio"),
    end_time: Optional[datetime] = Query(None, description="Tiempo de fin")
):
    """Obtener analytics de usuario específico."""
    try:
        user_analytics = await analytics_manager.get_user_analytics(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "user_analytics": user_analytics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener analytics de usuario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_analytics_stats():
    """Obtener estadísticas del sistema de analytics."""
    try:
        stats = await analytics_manager.get_analytics_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def analytics_health_check():
    """Verificar salud del sistema de analytics."""
    try:
        health = await analytics_manager.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de utilidad
@router.get("/event-types")
async def get_event_types():
    """Obtener tipos de eventos disponibles."""
    return {
        "event_types": [
            {
                "value": event_type.value,
                "name": event_type.name,
                "description": f"Evento de tipo {event_type.value}"
            }
            for event_type in AnalyticsEventType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/metric-types")
async def get_metric_types():
    """Obtener tipos de métricas disponibles."""
    return {
        "metric_types": [
            {
                "value": metric_type.value,
                "name": metric_type.name,
                "description": f"Métrica de tipo {metric_type.value}"
            }
            for metric_type in MetricType
        ],
        "timestamp": datetime.now().isoformat()
    }


# Rutas de ejemplo y testing
@router.post("/examples/track-page-view")
async def track_page_view_example(
    page: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Ejemplo: Rastrear visualización de página."""
    try:
        event_id = await analytics_manager.track_event(
            event_type=AnalyticsEventType.USER_ACTION,
            name="page_view",
            user_id=user_id,
            session_id=session_id,
            properties={
                "page": page,
                "url": f"/pages/{page}",
                "timestamp": datetime.now().isoformat()
            },
            tags={
                "category": "navigation",
                "type": "page_view"
            }
        )
        
        return {
            "event_id": event_id,
            "success": True,
            "message": f"Visualización de página '{page}' rastreada",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al rastrear visualización de página: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/examples/track-conversion")
async def track_conversion_example(
    conversion_type: str,
    value: float,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Ejemplo: Rastrear conversión."""
    try:
        # Rastrear evento de conversión
        event_id = await analytics_manager.track_event(
            event_type=AnalyticsEventType.BUSINESS_METRIC,
            name="conversion",
            user_id=user_id,
            session_id=session_id,
            properties={
                "conversion_type": conversion_type,
                "value": value,
                "currency": "USD"
            },
            metrics={
                "conversion_value": value,
                "conversion_count": 1
            },
            tags={
                "category": "business",
                "type": "conversion"
            }
        )
        
        # Rastrear métrica de conversión
        metric_id = await analytics_manager.track_metric(
            name="conversion_value",
            value=value,
            metric_type=MetricType.COUNTER,
            dimensions={
                "conversion_type": conversion_type,
                "user_id": user_id or "anonymous"
            }
        )
        
        return {
            "event_id": event_id,
            "metric_id": metric_id,
            "success": True,
            "message": f"Conversión '{conversion_type}' rastreada con valor ${value}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al rastrear conversión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/examples/track-performance")
async def track_performance_example(
    operation: str,
    duration_ms: float,
    success: bool = True
):
    """Ejemplo: Rastrear métrica de rendimiento."""
    try:
        # Rastrear evento de rendimiento
        event_id = await analytics_manager.track_event(
            event_type=AnalyticsEventType.PERFORMANCE_METRIC,
            name="operation_performance",
            properties={
                "operation": operation,
                "duration_ms": duration_ms,
                "success": success
            },
            metrics={
                "duration_ms": duration_ms,
                "success_rate": 1.0 if success else 0.0
            },
            tags={
                "category": "performance",
                "operation": operation
            }
        )
        
        # Rastrear métrica de duración
        metric_id = await analytics_manager.track_metric(
            name="operation_duration_ms",
            value=duration_ms,
            metric_type=MetricType.TIMER,
            dimensions={
                "operation": operation,
                "success": str(success)
            }
        )
        
        return {
            "event_id": event_id,
            "metric_id": metric_id,
            "success": True,
            "message": f"Rendimiento de operación '{operation}' rastreado: {duration_ms}ms",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al rastrear rendimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))




