"""
Health and stats router for Multi-Model API
Handles health checks and statistics endpoints
"""

import time
from datetime import datetime
from fastapi import APIRouter, Depends

from ...api.dependencies import get_model_registry, get_cache_instance
from ...core.models import ModelRegistry
from ...core.cache import EnhancedCache

router = APIRouter(prefix="/multi-model", tags=["Health"])


@router.get("/health")
async def health_check(
    registry: ModelRegistry = Depends(get_model_registry),
    cache: EnhancedCache = Depends(get_cache_instance)
):
    """
    Comprehensive health check
    
    Returns system health status including:
    - Overall system status (healthy/degraded/unhealthy)
    - Cache health
    - Model availability
    - Circuit breaker states
    """
    cache_stats = await cache.get_stats()
    available_models = len(registry.get_available_models())
    total_models = len(registry.models)
    
    circuit_breakers_open = sum(
        1 for m in registry.models.values()
        if m.circuit_breaker.state == "open"
    )
    
    error_rate = 0.0
    if total_models > 0:
        total_calls = sum(m.call_count for m in registry.models.values())
        total_errors = sum(m.error_count for m in registry.models.values())
        if total_calls > 0:
            error_rate = (total_errors / total_calls) * 100
    
    is_healthy = (
        available_models > 0 and
        error_rate < 10.0 and
        circuit_breakers_open < total_models / 2
    )
    
    status = "healthy" if is_healthy else "degraded"
    if available_models == 0:
        status = "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "cache": {
            "enabled": True,
            "hit_rate": round(cache_stats.get("hit_rate", 0), 2),
            "l1_size": cache_stats.get("l1_size", 0),
            "l1_max_size": cache_stats.get("l1_max_size", 0),
            "l1_utilization": round(cache_stats.get("l1_utilization", 0), 2),
            "l2_enabled": cache_stats.get("l2_enabled", False),
            "l3_enabled": cache_stats.get("l3_enabled", False)
        },
        "models": {
            "total_available": available_models,
            "total_registered": total_models,
            "circuit_breakers_open": circuit_breakers_open,
            "error_rate": round(error_rate, 2)
        },
        "system": {
            "uptime_seconds": time.time(),
            "version": "2.7.0"
        }
    }


@router.get("/stats")
async def get_stats(
    registry: ModelRegistry = Depends(get_model_registry),
    cache: EnhancedCache = Depends(get_cache_instance)
):
    """
    Get comprehensive statistics
    
    Returns detailed metrics including:
    - Cache performance (hit rate, size, latency)
    - Model performance (success rates, latencies, circuit breaker states)
    - System health metrics
    - Rate limiting configuration
    """
    from ...core.rate_limiter import get_rate_limiter
    
    cache_stats = await cache.get_stats()
    
    model_stats = []
    total_calls = 0
    total_success = 0
    total_errors = 0
    total_latency = 0.0
    circuit_breakers_open = 0
    
    for model_type, model_meta in registry.models.items():
        model_stats.append({
            "model_type": model_type.value,
            "name": model_meta.name,
            "call_count": model_meta.call_count,
            "success_count": model_meta.success_count,
            "error_count": model_meta.error_count,
            "success_rate": round(model_meta.success_rate, 2),
            "avg_latency_ms": round(model_meta.avg_latency_ms, 2),
            "p95_latency_ms": round(model_meta.p95_latency_ms, 2),
            "circuit_breaker_state": model_meta.circuit_breaker.state,
            "circuit_breaker_failures": model_meta.circuit_breaker.failures,
            "is_available": model_meta.is_available,
            "last_used": model_meta.last_used.isoformat() if model_meta.last_used else None
        })
        
        total_calls += model_meta.call_count
        total_success += model_meta.success_count
        total_errors += model_meta.error_count
        total_latency += model_meta.total_latency_ms
        if model_meta.circuit_breaker.state == "open":
            circuit_breakers_open += 1
    
    overall_success_rate = (total_success / total_calls * 100) if total_calls > 0 else 0.0
    overall_avg_latency = (total_latency / total_calls) if total_calls > 0 else 0.0
    
    rate_limiter = get_rate_limiter()
    
    return {
        "cache": cache_stats,
        "models": model_stats,
        "summary": {
            "total_calls": total_calls,
            "total_success": total_success,
            "total_errors": total_errors,
            "overall_success_rate": round(overall_success_rate, 2),
            "overall_avg_latency_ms": round(overall_avg_latency, 2),
            "circuit_breakers_open": circuit_breakers_open,
            "total_models": len(registry.models),
            "available_models": len(registry.get_available_models())
        },
        "rate_limiting": {
            "default_limit": rate_limiter.default_limit,
            "default_window": rate_limiter.default_window,
            "burst_limit": rate_limiter.burst_limit
        },
        "timestamp": datetime.now().isoformat()
    }




