"""
Advanced metrics router for Multi-Model API
Provides detailed metrics and analytics endpoints
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional

from ...core.services import get_metrics_service
from ...core.services.metrics_service import MetricsService

router = APIRouter(prefix="/multi-model/metrics", tags=["Advanced Metrics"])


@router.get("/requests")
async def get_request_metrics(
    last_n: int = Query(100, ge=1, le=1000, description="Number of recent requests to analyze"),
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Get statistics for recent requests
    
    Returns:
    - Total requests
    - Average latency
    - Success rate
    - Cache hit rate
    - Total model executions
    """
    return metrics_service.get_request_stats(last_n=last_n)


@router.get("/models")
async def get_model_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Get statistics for all models
    
    Returns detailed metrics for each model including:
    - Call count
    - Success/failure counts
    - Success rate
    - Average latency
    - Last called timestamp
    """
    return metrics_service.get_model_stats()


@router.get("/strategies")
async def get_strategy_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Get statistics by execution strategy
    
    Returns metrics for each strategy:
    - Request count
    - Average latency
    - Total model executions
    - Success rate
    """
    return metrics_service.get_strategy_stats()


@router.post("/reset")
async def reset_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Reset all metrics
    
    WARNING: This will clear all collected metrics
    """
    metrics_service.reset()
    return {"message": "Metrics reset successfully"}




