"""
Monitoring API Endpoints
========================

Endpoints para monitoreo avanzado.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

try:
    from ..core.performance.performance_monitor import get_performance_monitor
except ImportError:
    try:
        from ..core.monitoring.performance_monitor import get_performance_monitor
    except ImportError:
        def get_performance_monitor():
            return None
try:
    from ..core.error_tracker import get_error_tracker
except ImportError:
    def get_error_tracker():
        return None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/performance")
async def get_performance_metrics(
    window_seconds: Optional[float] = Query(None, ge=1.0)
) -> Dict[str, Any]:
    """Obtener métricas de performance."""
    try:
        monitor = get_performance_monitor()
        metrics = monitor.get_performance_metrics(window_seconds=window_seconds)
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/snapshot")
async def take_performance_snapshot() -> Dict[str, Any]:
    """Tomar snapshot de performance."""
    try:
        monitor = get_performance_monitor()
        snapshot = monitor.take_snapshot()
        return {
            "timestamp": snapshot.timestamp,
            "cpu_usage": snapshot.cpu_usage,
            "memory_usage": snapshot.memory_usage,
            "active_requests": snapshot.active_requests,
            "response_time": snapshot.response_time,
            "throughput": snapshot.throughput,
            "error_rate": snapshot.error_rate
        }
    except Exception as e:
        logger.error(f"Error taking snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/anomalies")
async def get_performance_anomalies() -> Dict[str, Any]:
    """Obtener anomalías de performance."""
    try:
        monitor = get_performance_monitor()
        anomalies = monitor.detect_anomalies()
        return {
            "anomalies": anomalies,
            "count": len(anomalies)
        }
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors")
async def get_error_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de errores."""
    try:
        tracker = get_error_tracker()
        stats = tracker.get_error_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting error statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors/recent")
async def get_recent_errors(
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Obtener errores recientes."""
    try:
        tracker = get_error_tracker()
        errors = tracker.get_recent_errors(limit=limit)
        return {
            "errors": [
                {
                    "error_id": error.error_id,
                    "error_type": error.error_type,
                    "message": error.message,
                    "timestamp": error.timestamp,
                    "count": error.count,
                    "context": error.context
                }
                for error in errors
            ],
            "count": len(errors)
        }
    except Exception as e:
        logger.error(f"Error getting recent errors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors/{error_id}")
async def get_error_details(error_id: str) -> Dict[str, Any]:
    """Obtener detalles de error."""
    try:
        tracker = get_error_tracker()
        error = tracker.get_error_by_id(error_id)
        
        if not error:
            raise HTTPException(status_code=404, detail="Error not found")
        
        return {
            "error_id": error.error_id,
            "error_type": error.error_type,
            "message": error.message,
            "traceback": error.traceback,
            "count": error.count,
            "first_occurrence": error.first_occurrence,
            "last_occurrence": error.last_occurrence,
            "context": error.context
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting error details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/errors/clear")
async def clear_errors() -> Dict[str, Any]:
    """Limpiar todos los errores."""
    try:
        tracker = get_error_tracker()
        tracker.clear_errors()
        return {"message": "All errors cleared"}
    except Exception as e:
        logger.error(f"Error clearing errors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






