"""
Health Check Router
==================

Health check and root endpoints.
"""

from fastapi import APIRouter
import torch
import psutil
import os
from ..config import settings
from ..services import model_store
from ..metrics import metrics_collector
from ..cache import cache
from ..constants import BYTES_PER_MB

router = APIRouter()


@router.get("/")
async def root():
    """
    Root endpoint.
    
    Returns basic API information.
    """
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@router.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns detailed health information including:
    - System status
    - CUDA availability
    - Memory usage
    - Model store status
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        stats = metrics_collector.get_stats()
        
        health_status = {
            "status": "healthy",
            "version": settings.app_version,
            "cuda_available": torch.cuda.is_available(),
            "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            "system": {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": memory_info.rss / BYTES_PER_MB,
                "memory_percent": process.memory_percent()
            },
            "models": {
                "total": len(all_models := model_store.list_all()),
                "compiled": sum(1 for m in all_models.values() if m.get("compiled", False))
            },
            "uptime_seconds": stats.get("uptime_seconds", 0),
            "total_requests": stats.get("total_requests", 0)
        }
        
        if torch.cuda.is_available():
            health_status["cuda"] = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": torch.cuda.memory_allocated(0) / BYTES_PER_MB,
                "memory_reserved_mb": torch.cuda.memory_reserved(0) / BYTES_PER_MB
            }
        
        return health_status
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/metrics")
async def get_metrics():
    """
    Get API metrics.
    
    Returns detailed metrics including:
    - Request statistics
    - Error rates
    - Performance metrics
    - Model operation counts
    """
    return metrics_collector.get_stats()


@router.get("/metrics/cache")
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns cache information including:
    - Cache size
    - Entry details
    - TTL settings
    """
    return cache.get_stats()


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Reset all metrics.
    
    Clears all collected metrics and resets counters.
    """
    metrics_collector.reset()
    return {
        "status": "success",
        "message": "Metrics reset successfully"
    }

