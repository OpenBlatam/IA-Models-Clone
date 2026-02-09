"""Metrics endpoints."""

from fastapi import APIRouter
from utils.metrics import metrics_collector

router = APIRouter()


@router.get("/")
async def get_metrics():
    """Get current metrics summary."""
    return metrics_collector.get_metrics_summary()


@router.get("/counters")
async def get_counters():
    """Get counter metrics."""
    return {
        "counters": dict(metrics_collector.counters),
        "timestamp": metrics_collector.get_metrics_summary()["timestamp"]
    }


@router.get("/timings")
async def get_timings():
    """Get timing metrics."""
    return {
        "timings": metrics_collector.get_metrics_summary()["timings"],
        "timestamp": metrics_collector.get_metrics_summary()["timestamp"]
    }

