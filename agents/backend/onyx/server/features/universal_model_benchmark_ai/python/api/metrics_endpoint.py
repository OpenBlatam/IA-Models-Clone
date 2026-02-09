"""
Metrics Endpoint - Prometheus metrics endpoint for FastAPI.

Provides:
- /metrics endpoint
- Prometheus format export
- Health metrics
"""

from fastapi import APIRouter
from core.metrics import metrics_collector

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """
    Get Prometheus metrics.
    
    Returns:
        Metrics in Prometheus text format
    """
    return metrics_collector.export_metrics()












