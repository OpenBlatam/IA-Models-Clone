"""Metrics endpoints"""
from fastapi import APIRouter, Response
from utils.metrics import get_metrics
from utils.prometheus_metrics import get_prometheus_metrics

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("")
async def get_metrics():
    """Get service metrics"""
    metrics = get_metrics()
    return metrics.get_all_metrics()


@router.post("/reset")
async def reset_metrics():
    """Reset all metrics"""
    metrics = get_metrics()
    metrics.reset()
    return {
        "status": "success",
        "message": "Metrics reset"
    }


@router.get("/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    prom_metrics = get_prometheus_metrics()
    metrics_data = prom_metrics.get_metrics()
    
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

