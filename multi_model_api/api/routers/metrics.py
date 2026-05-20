"""
Metrics router for Multi-Model API
Handles Prometheus metrics endpoint
"""

from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/multi-model", tags=["Metrics"])


@router.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint
    
    Returns metrics in Prometheus format for scraping
    
    Returns:
        Prometheus metrics in text format
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Prometheus client not available. Install prometheus-client to enable metrics."
        )




