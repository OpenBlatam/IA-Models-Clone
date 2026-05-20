from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sys
import time
from ..types import (
from ..utils import validate_api_key, validate_performance_thresholds
from ..core.optimized_engine import optimized_engine, performance_monitor
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v14.0 - Performance Monitoring Routes
FastAPI router for performance monitoring and metrics endpoints
"""


    PerformanceStats,
    PerformanceSummary,
    HealthCheckResponse,
    APIInfoResponse,
    CacheStatsResponse,
    AIPerformanceResponse,
    PerformanceStatusResponse,
    OptimizationResponse,
    ErrorResponse
)

# Router configuration
router = APIRouter(
    prefix="/performance",
    tags=["Performance Monitoring"],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

# Security
security = HTTPBearer()

# Dependency for API key validation
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key - async function for dependency injection"""
    if not validate_api_key(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Optimized health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        version="14.0.0",
        timestamp=time.time(),
        optimizations={
            "jit_enabled": True,
            "cache_enabled": True,
            "batching_enabled": True,
            "mixed_precision": True
        }
    )

@router.get("/metrics", response_model=dict)
async def get_metrics() -> dict:
    """Real-time performance metrics endpoint"""
    return {
        "engine_stats": optimized_engine.get_stats(),
        "performance_summary": performance_monitor.get_performance_summary(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "optimization_level": "ultra_fast"
        }
    }

@router.get("/status", response_model=PerformanceStatusResponse)
async def performance_status() -> PerformanceStatusResponse:
    """Current performance status endpoint"""
    stats = optimized_engine.get_stats()
    perf_summary = performance_monitor.get_performance_summary()
    
    # Determine performance grade
    avg_time = perf_summary["avg_response_time"]
    if avg_time < 0.015:
        grade = "ULTRA_FAST"
    elif avg_time < 0.025:
        grade = "FAST"
    elif avg_time < 0.050:
        grade = "NORMAL"
    else:
        grade = "SLOW"
    
    return PerformanceStatusResponse(
        performance_grade=grade,
        average_response_time=avg_time,
        cache_hit_rate=stats["cache_hit_rate"],
        total_requests=stats["total_requests"],
        uptime=perf_summary["uptime"]
    )

@router.get("/cache", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    """Cache performance statistics endpoint"""
    stats = optimized_engine.get_stats()
    return CacheStatsResponse(
        cache_size=stats["cache_size"],
        cache_hit_rate=stats["cache_hit_rate"],
        total_requests=stats["total_requests"],
        cache_hits=stats["cache_hits"],
        cache_misses=stats["total_requests"] - stats["cache_hits"]
    )

@router.get("/ai", response_model=AIPerformanceResponse)
async def ai_performance() -> AIPerformanceResponse:
    """AI provider performance metrics endpoint"""
    stats = optimized_engine.get_stats()
    return AIPerformanceResponse(
        device=stats["device"],
        average_processing_time=stats["average_processing_time"],
        optimizations_enabled=stats["optimizations_enabled"],
        model_loaded=optimized_engine.model is not None,
        tokenizer_loaded=optimized_engine.tokenizer is not None
    )

@router.post("/optimize", response_model=OptimizationResponse)
async def trigger_optimization(
    api_key: str = Depends(verify_api_key)
) -> OptimizationResponse:
    """Trigger performance optimization endpoint"""
    try:
        # Clear cache for fresh start
        optimized_engine.cache.clear()
        
        # Reinitialize models if needed
        if not optimized_engine.model:
            await optimized_engine._initialize_models()
        
        return OptimizationResponse(
            status="optimization_completed",
            cache_cleared=True,
            models_reinitialized=optimized_engine.model is not None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/thresholds", response_model=dict)
async def performance_thresholds() -> dict:
    """Performance thresholds and grading endpoint"""
    stats = optimized_engine.get_stats()
    perf_summary = performance_monitor.get_performance_summary()
    
    thresholds = validate_performance_thresholds(
        avg_response_time=perf_summary["avg_response_time"],
        cache_hit_rate=stats["cache_hit_rate"],
        success_rate=perf_summary["success_rate"]
    )
    
    return {
        "current_metrics": {
            "avg_response_time": perf_summary["avg_response_time"],
            "cache_hit_rate": stats["cache_hit_rate"],
            "success_rate": perf_summary["success_rate"]
        },
        "grades": thresholds,
        "recommendations": {
            "response_time": "Consider optimization if grade is SLOW",
            "cache": "Increase cache size if grade is POOR",
            "success_rate": "Check error logs if grade is POOR"
        }
    } 