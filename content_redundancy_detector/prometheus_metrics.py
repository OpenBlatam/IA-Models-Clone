"""
Prometheus Metrics Endpoint and Integration
Exposes /metrics endpoint for Prometheus scraping
"""

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    Exposes all Prometheus metrics for scraping
    """
    metrics_data = generate_latest()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)






