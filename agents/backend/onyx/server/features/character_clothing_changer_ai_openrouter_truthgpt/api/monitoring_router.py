"""
Monitoring Router
=================

FastAPI router for monitoring and system information.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from utils.monitoring import get_alert_manager, get_system_monitor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/monitoring/alerts",
    status_code=status.HTTP_200_OK,
    summary="Get Recent Alerts",
    description="Get recent system alerts"
)
async def get_alerts(
    level: Optional[str] = None,
    limit: int = 10,
    alert_manager = Depends(get_alert_manager)
) -> Dict[str, Any]:
    """
    Get recent alerts.
    
    Args:
        level: Optional alert level filter
        limit: Maximum number of alerts to return
        alert_manager: AlertManager instance (injected)
        
    Returns:
        Dictionary with alerts
    """
    try:
        alerts = alert_manager.get_recent_alerts(level=level, limit=limit)
        return {
            "count": len(alerts),
            "alerts": [
                {
                    "level": a.level,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                    "metadata": a.metadata
                }
                for a in alerts
            ]
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alerts: {str(e)}"
        )


@router.get(
    "/monitoring/resources",
    status_code=status.HTTP_200_OK,
    summary="Get System Resources",
    description="Get current system resource usage"
)
async def get_resources(
    system_monitor = Depends(get_system_monitor)
) -> Dict[str, Any]:
    """
    Get system resource usage.
    
    Args:
        system_monitor: SystemMonitor instance (injected)
        
    Returns:
        Dictionary with resource information
    """
    try:
        resources = system_monitor.check_resources()
        return {
            "resources": resources,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting resources: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resources: {str(e)}"
        )


@router.post(
    "/monitoring/alerts/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear Alerts",
    description="Clear all stored alerts"
)
async def clear_alerts(
    alert_manager = Depends(get_alert_manager)
) -> Dict[str, Any]:
    """
    Clear all alerts.
    
    Args:
        alert_manager: AlertManager instance (injected)
        
    Returns:
        Dictionary with result
    """
    try:
        count = alert_manager.clear_alerts()
        return {
            "success": True,
            "cleared_count": count,
            "message": f"Cleared {count} alerts"
        }
    except Exception as e:
        logger.error(f"Error clearing alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear alerts: {str(e)}"
        )

