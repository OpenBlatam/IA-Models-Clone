"""
Prometheus Metrics Endpoint
============================

Exposes Prometheus metrics for monitoring.
"""

from fastapi import APIRouter
from fastapi.responses import Response

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

router = APIRouter(prefix="/metrics", tags=["monitoring"])


@router.get("")
async def metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content="Prometheus client not available",
            status_code=503,
            media_type="text/plain",
        )

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )




