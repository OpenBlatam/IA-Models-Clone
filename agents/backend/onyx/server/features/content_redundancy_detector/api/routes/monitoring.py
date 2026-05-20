"""
Monitoring Router - System monitoring and alerting endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

try:
    from monitoring_advanced import monitoring_system, AlertLevel
except ImportError:
    logging.warning("monitoring_advanced module not available")
    monitoring_system = None
    AlertLevel = None

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/metrics", response_model=Dict[str, Any])
async def get_monitoring_metrics() -> JSONResponse:
    """Get monitoring metrics"""
    logger.info("Monitoring metrics requested")
    
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        metrics = monitoring_system.get_metrics()
        return JSONResponse(content={
            "success": True,
            "data": metrics,
            "error": None
        })
    except Exception as e:
        logger.error(f"Get monitoring metrics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/alerts", response_model=Dict[str, Any])
async def get_monitoring_alerts(limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
    """Get monitoring alerts"""
    logger.info("Monitoring alerts requested")
    
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        alerts = monitoring_system.get_alerts(limit)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "alerts": [
                    {
                        "id": alert.id,
                        "name": alert.name,
                        "level": alert.level.value if hasattr(alert.level, 'value') else str(alert.level),
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                        "resolved": alert.resolved,
                        "metadata": alert.metadata
                    }
                    for alert in alerts
                ],
                "count": len(alerts)
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get monitoring alerts error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/alert/create", response_model=Dict[str, Any])
async def create_monitoring_alert(alert_data: Dict[str, Any]) -> JSONResponse:
    """Create monitoring alert"""
    logger.info("Monitoring alert creation requested")
    
    if not monitoring_system or not AlertLevel:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        name = alert_data.get("name")
        level = alert_data.get("level", "info")
        message = alert_data.get("message")
        metadata = alert_data.get("metadata", {})
        
        if not all([name, message]):
            raise ValueError("Name and message are required")
        
        level_mapping = {
            "info": AlertLevel.INFO,
            "warning": AlertLevel.WARNING,
            "error": AlertLevel.ERROR,
            "critical": AlertLevel.CRITICAL
        }
        
        alert_level = level_mapping.get(level, AlertLevel.INFO)
        alert = monitoring_system.create_alert(name, alert_level, message, metadata)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Alert created successfully",
                "alert_id": alert.id,
                "name": alert.name,
                "level": alert.level.value if hasattr(alert.level, 'value') else str(alert.level),
                "message": alert.message
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create monitoring alert error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/alert/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_monitoring_alert(alert_id: str) -> JSONResponse:
    """Resolve monitoring alert"""
    logger.info(f"Monitoring alert resolution requested: {alert_id}")
    
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        success = monitoring_system.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Alert resolved successfully",
                "alert_id": alert_id
            },
            "error": None
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolve monitoring alert error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health", response_model=Dict[str, Any])
async def get_monitoring_health() -> JSONResponse:
    """Get monitoring health status"""
    logger.info("Monitoring health status requested")
    
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        health_status = monitoring_system.get_health_status()
        return JSONResponse(content={
            "success": True,
            "data": health_status,
            "error": None
        })
    except Exception as e:
        logger.error(f"Get monitoring health error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/performance", response_model=Dict[str, Any])
async def get_monitoring_performance() -> JSONResponse:
    """Get monitoring performance metrics"""
    logger.info("Monitoring performance metrics requested")
    
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        performance_metrics = monitoring_system.get_performance_metrics()
        return JSONResponse(content={
            "success": True,
            "data": performance_metrics,
            "error": None
        })
    except Exception as e:
        logger.error(f"Get monitoring performance error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/system", response_model=Dict[str, Any])
async def get_monitoring_system_metrics() -> JSONResponse:
    """Get monitoring system metrics"""
    logger.info("Monitoring system metrics requested")
    
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        system_metrics = monitoring_system.get_system_metrics()
        return JSONResponse(content={
            "success": True,
            "data": system_metrics,
            "error": None
        })
    except Exception as e:
        logger.error(f"Get monitoring system metrics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=Dict[str, Any])
async def get_monitoring_stats() -> JSONResponse:
    """Get monitoring statistics"""
    logger.info("Monitoring stats requested")
    
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        stats = monitoring_system.get_stats()
        return JSONResponse(content={
            "success": True,
            "data": stats,
            "error": None
        })
    except Exception as e:
        logger.error(f"Get monitoring stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






