"""
Performance router for Multi-Model API
Provides performance metrics and optimization endpoints
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional

from ...core.services import get_performance_service
from ...core.services.performance_service import PerformanceService

router = APIRouter(prefix="/multi-model/performance", tags=["Performance"])


@router.get("/metrics")
async def get_performance_metrics(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Get current performance metrics
    
    Returns:
    - Requests per second
    - Average latency
    - P95 and P99 latency
    - Error rate
    - Cache hit rate
    """
    return performance_service.get_current_metrics()


@router.get("/snapshots")
async def get_performance_snapshots(
    last_n: int = Query(10, ge=1, le=100, description="Number of recent snapshots"),
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Get recent performance snapshots
    
    Returns historical performance data for analysis
    """
    snapshots = performance_service.get_snapshots(last_n=last_n)
    return {
        "snapshots": [
            {
                "timestamp": s.timestamp,
                "requests_per_second": s.requests_per_second,
                "avg_latency_ms": s.avg_latency_ms,
                "p95_latency_ms": s.p95_latency_ms,
                "p99_latency_ms": s.p99_latency_ms,
                "error_rate": s.error_rate,
                "cache_hit_rate": s.cache_hit_rate
            }
            for s in snapshots
        ],
        "count": len(snapshots)
    }


@router.get("/issues")
async def detect_performance_issues(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Detect potential performance issues
    
    Returns list of detected issues with recommendations
    """
    issues = performance_service.detect_performance_issues()
    return {
        "issues": issues,
        "count": len(issues),
        "has_issues": len(issues) > 0
    }


@router.post("/snapshot")
async def take_performance_snapshot(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Take a snapshot of current performance metrics
    
    Useful for tracking performance at specific points in time
    """
    snapshot = performance_service.take_snapshot()
    return {
        "snapshot": {
            "timestamp": snapshot.timestamp,
            "requests_per_second": snapshot.requests_per_second,
            "avg_latency_ms": snapshot.avg_latency_ms,
            "p95_latency_ms": snapshot.p95_latency_ms,
            "p99_latency_ms": snapshot.p99_latency_ms,
            "error_rate": snapshot.error_rate,
            "cache_hit_rate": snapshot.cache_hit_rate
        },
        "message": "Snapshot taken successfully"
    }


@router.post("/reset")
async def reset_performance_metrics(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Reset all performance metrics
    
    WARNING: This will clear all collected performance data
    """
    performance_service.reset()
    return {"message": "Performance metrics reset successfully"}




