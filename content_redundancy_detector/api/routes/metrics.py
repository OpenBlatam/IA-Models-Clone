"""
Metrics Router - System metrics endpoints
"""

from typing import Dict, Any
from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from ...core.logging_config import get_logger

# Prefer optional imports from webhooks package with graceful degradation
try:
    from ...webhooks import (
        metrics_collector,
        health_checker,
        HEALTH_MONITORING_AVAILABLE,
        METRICS_AVAILABLE,
    )
except Exception:
    metrics_collector = None  # type: ignore
    health_checker = None  # type: ignore
    HEALTH_MONITORING_AVAILABLE = False  # type: ignore
    METRICS_AVAILABLE = False  # type: ignore

logger = get_logger(__name__)

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def get_metrics() -> Dict[str, Any]:
    """Get overall metrics summary (if available)."""
    data: Dict[str, Any] = {}
    if METRICS_AVAILABLE and metrics_collector is not None:
        data["webhooks"] = metrics_collector.get_metrics_summary()
    else:
        data["webhooks"] = {"available": False}
    return {"success": True, "data": data, "error": None}


@router.get("/webhooks", response_model=Dict[str, Any])
async def get_webhook_metrics() -> Dict[str, Any]:
    """Get webhook metrics and health (if available)."""
    result: Dict[str, Any] = {"metrics": {}, "health": {}}

    if METRICS_AVAILABLE and metrics_collector is not None:
        result["metrics"] = metrics_collector.get_metrics_summary()
    else:
        result["metrics"] = {"available": False}

    if HEALTH_MONITORING_AVAILABLE and health_checker is not None:
        health_report = await health_checker.get_health_report()
        result["health"] = health_report
    else:
        result["health"] = {"available": False}

    return {"success": True, "data": result, "error": None}


@router.get("/prometheus")
async def get_prometheus_metrics() -> Response:
    """Expose Prometheus-formatted metrics if `prometheus_client` is available."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # type: ignore
        # If we later register custom metrics, they'll be included automatically
        payload = generate_latest()  # bytes
        resp = Response(content=payload, media_type=CONTENT_TYPE_LATEST)
        # Prometheus endpoints should not be cached
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return resp
    except Exception:
        # Graceful degradation when prometheus_client is not installed
        return PlainTextResponse(
            content="# Prometheus client not available or metrics not configured\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
            status_code=503,
            headers={"Cache-Control": "no-store"}
        )

