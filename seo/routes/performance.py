from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from ..dependencies import (
from ..core import PerformanceMetricsManager
        from fastapi.responses import StreamingResponse
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Performance routes for Ultra-Optimized SEO Service v15.

This module contains performance monitoring endpoints including:
- Performance metrics retrieval
- Endpoint-specific metrics
- Performance alerts
- Real-time monitoring
- Threshold management
"""


    get_performance_manager,
    get_logger
)

# Create router with prefix and tags
router = APIRouter(
    prefix="/performance",
    tags=["Performance Monitoring"],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"}
    }
)

@router.get("/metrics")
async def get_performance_metrics(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """
    Get comprehensive performance metrics.
    
    Returns detailed performance metrics including:
    - Response time statistics
    - Throughput metrics
    - System resource usage
    - Cache performance
    - Error rates
    """
    try:
        metrics = performance_manager.get_current_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")

@router.get("/endpoints")
async def get_endpoint_metrics(
    endpoint: Optional[str] = None,
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """
    Get endpoint-specific performance metrics.
    
    Returns performance metrics for specific endpoints or all endpoints.
    Includes response times, error rates, and throughput per endpoint.
    """
    try:
        metrics = performance_manager.get_endpoint_metrics(endpoint)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve endpoint metrics")

@router.get("/alerts")
async def get_performance_alerts(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """
    Get performance alerts and warnings.
    
    Returns active performance alerts based on configured thresholds.
    Includes severity levels and recommended actions.
    """
    try:
        alerts = performance_manager.get_performance_alerts()
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve performance alerts")

@router.get("/summary")
async def get_performance_summary(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """
    Get performance summary and trends.
    
    Returns a comprehensive performance summary including:
    - Current performance status
    - Historical trends
    - Performance recommendations
    - System health indicators
    """
    try:
        summary = performance_manager.get_performance_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve performance summary")

@router.post("/reset")
async def reset_performance_metrics(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager),
    logger = Depends(get_logger)
):
    """
    Reset performance metrics.
    
    Clears all performance metrics and starts fresh collection.
    Useful for testing and performance baseline establishment.
    """
    try:
        performance_manager.reset_metrics()
        logger.info("Performance metrics reset successfully")
        return {"success": True, "message": "Performance metrics reset successfully"}
    except Exception as e:
        logger.error("Failed to reset performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to reset performance metrics")

@router.get("/thresholds")
async def get_performance_thresholds(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """
    Get performance thresholds configuration.
    
    Returns current performance thresholds used for alerting.
    Includes response time, throughput, and error rate thresholds.
    """
    try:
        thresholds = performance_manager.thresholds
        return thresholds
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve performance thresholds")

@router.get("/real-time")
async def get_real_time_performance(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """
    Get real-time performance data stream.
    
    Returns a Server-Sent Events stream of real-time performance metrics.
    Useful for live monitoring dashboards.
    """
    try:
        
        async def generate_real_time_metrics():
            """Generate real-time performance metrics stream."""
            while True:
                try:
                    metrics = performance_manager.get_current_metrics()
                    yield f"data: {metrics.model_dump_json()}\n\n"
                    await asyncio.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                    break
        
        return StreamingResponse(
            generate_real_time_metrics(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to start real-time monitoring")

@router.post("/thresholds/update")
async def update_performance_thresholds(
    thresholds: dict,
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager),
    logger = Depends(get_logger)
):
    """
    Update performance thresholds.
    
    Updates performance thresholds for alerting and monitoring.
    Requires proper validation of threshold values.
    """
    try:
        # Update thresholds (implementation would be in PerformanceMetricsManager)
        performance_manager.update_thresholds(thresholds)
        logger.info("Performance thresholds updated successfully")
        return {"success": True, "message": "Performance thresholds updated successfully"}
    except Exception as e:
        logger.error("Failed to update performance thresholds", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update performance thresholds")

@router.get("/health")
async def get_performance_health(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """
    Get performance health status.
    
    Returns overall performance health status including:
    - System performance score
    - Health indicators
    - Performance recommendations
    """
    try:
        health_status = performance_manager.get_health_status()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve performance health") 