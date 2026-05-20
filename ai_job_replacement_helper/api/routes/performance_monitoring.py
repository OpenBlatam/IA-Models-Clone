"""
Performance Monitoring endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.performance_monitoring import PerformanceMonitoringService

router = APIRouter()
monitoring_service = PerformanceMonitoringService()


@router.get("/endpoint-stats")
async def get_endpoint_stats(
    endpoint: str,
    method: str = "GET"
) -> Dict[str, Any]:
    """Obtener estadísticas de endpoint"""
    try:
        stats = monitoring_service.get_endpoint_stats(endpoint, method)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshot")
async def get_performance_snapshot(
    minutes: int = 5
) -> Dict[str, Any]:
    """Obtener snapshot de rendimiento"""
    try:
        snapshot = monitoring_service.get_performance_snapshot(minutes)
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "summary": snapshot.summary,
            "metrics_count": len(snapshot.metrics),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/slow-endpoints")
async def detect_slow_endpoints(
    threshold_ms: float = 1000.0
) -> Dict[str, Any]:
    """Detectar endpoints lentos"""
    try:
        slow = monitoring_service.detect_slow_endpoints(threshold_ms)
        return {
            "threshold_ms": threshold_ms,
            "slow_endpoints": slow,
            "total": len(slow),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-health")
async def get_system_health_metrics() -> Dict[str, Any]:
    """Obtener métricas de salud del sistema"""
    try:
        metrics = monitoring_service.get_system_health_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




