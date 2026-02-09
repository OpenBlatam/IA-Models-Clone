"""
Health Router - Handles health monitoring and system status endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional

from ...api.services_locator import get_service
from ...utils.logger import logger

router = APIRouter(prefix="/dermatology", tags=["health"])


@router.get("/health")
async def health_check():
    """Health check básico"""
    return JSONResponse(content={
        "status": "healthy",
        "service": "Dermatology AI",
        "version": "5.3.0"
    })


@router.get("/health/detailed")
async def detailed_health_check():
    """Health check detallado"""
    try:
        health_monitor = get_service("health_monitor")
        health_status = health_monitor.get_detailed_health()
        return JSONResponse(content={
            "success": True,
            "health": health_status.to_dict() if hasattr(health_status, 'to_dict') else health_status
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/health/check/{check_name}")
async def specific_health_check(check_name: str):
    """Health check específico"""
    try:
        health_monitor = get_service("health_monitor")
        check_result = health_monitor.run_health_check(check_name)
        return JSONResponse(content={
            "success": True,
            "check": check_result.to_dict() if hasattr(check_result, 'to_dict') else check_result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/monitoring/health")
async def monitoring_health():
    """Estado de monitoreo"""
    try:
        advanced_monitoring = get_service("advanced_monitoring")
        health = advanced_monitoring.get_system_health()
        return JSONResponse(content={
            "success": True,
            "health": health.to_dict() if hasattr(health, 'to_dict') else health
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/monitoring/metrics/{metric_name}")
async def get_monitoring_metric(
    metric_name: str,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None)
):
    """Obtiene métrica de monitoreo"""
    try:
        advanced_monitoring = get_service("advanced_monitoring")
        metric = advanced_monitoring.get_metric(metric_name, start_time, end_time)
        return JSONResponse(content={
            "success": True,
            "metric": metric.to_dict() if hasattr(metric, 'to_dict') else metric
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/monitoring/alerts")
async def get_monitoring_alerts():
    """Obtiene alertas de monitoreo"""
    try:
        advanced_monitoring = get_service("advanced_monitoring")
        alerts = advanced_monitoring.get_alerts()
        return JSONResponse(content={
            "success": True,
            "alerts": [alert.to_dict() for alert in alerts]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/metrics/realtime")
async def get_realtime_metrics():
    """Obtiene métricas en tiempo real"""
    try:
        realtime_metrics = get_service("realtime_metrics")
        metrics = realtime_metrics.get_current_metrics()
        return JSONResponse(content={
            "success": True,
            "metrics": [m.to_dict() for m in metrics]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")




