"""
Metrics routes for Logistics AI Platform

This module provides endpoints for Prometheus metrics and monitoring.
"""

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from utils.metrics import get_metrics_collector

try:
    from prometheus_client import CONTENT_TYPE_LATEST
except ImportError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"],
    responses={
        200: {"description": "Metrics data"},
        503: {"description": "Metrics service unavailable"}
    }
)


@router.get(
    "",
    summary="Prometheus metrics",
    description="Get metrics in Prometheus format for scraping",
    response_class=PlainTextResponse
)
async def get_prometheus_metrics() -> Response:
    """
    Get metrics in Prometheus format
    
    Returns:
        Metrics in Prometheus text format
        
    This endpoint exposes metrics in Prometheus format for scraping.
    Compatible with Prometheus server and other monitoring tools.
    """
    collector = get_metrics_collector()
    metrics_data = collector.get_metrics()
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST if CONTENT_TYPE_LATEST else "text/plain"
    )


@router.get(
    "/info",
    summary="Metrics information",
    description="Get information about available metrics"
)
async def get_metrics_info():
    """
    Get metrics information
    
    Returns:
        Dictionary with metrics information
    """
    collector = get_metrics_collector()
    return {
        "enabled": collector.enabled,
        "metrics": collector.get_metrics_dict()
    }

