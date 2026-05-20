"""
Monitoring Routes
================

API routes for monitoring and statistics.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..dependencies import get_agent, get_dashboard
from ...utils.memory_optimizer import MemoryOptimizer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


@router.get("/stats")
async def get_stats():
    """Get agent statistics."""
    agent = get_agent()
    
    try:
        stats = agent.get_stats()
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/metrics")
async def get_dashboard_metrics():
    """Get dashboard metrics."""
    dashboard = get_dashboard()
    
    try:
        metrics = dashboard.get_metrics_dict()
        return JSONResponse(metrics)
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/health")
async def get_dashboard_health():
    """Get system health."""
    dashboard = get_dashboard()
    
    try:
        health = dashboard.get_system_health()
        return JSONResponse(health)
    except Exception as e:
        logger.error(f"Error getting dashboard health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/trends")
async def get_dashboard_trends():
    """Get performance trends."""
    dashboard = get_dashboard()
    
    try:
        trends = dashboard.get_performance_trends()
        return JSONResponse(trends)
    except Exception as e:
        logger.error(f"Error getting trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/usage")
async def get_memory_usage():
    """Get memory usage information."""
    try:
        usage = MemoryOptimizer.get_memory_usage()
        recommendations = MemoryOptimizer.get_memory_recommendations()
        return JSONResponse({
            "usage": usage,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/optimize")
async def optimize_memory(aggressive: bool = False):
    """Optimize memory usage."""
    try:
        collected = MemoryOptimizer.optimize_memory(aggressive=aggressive)
        return JSONResponse({
            "success": True,
            "objects_collected": collected,
            "aggressive": aggressive
        })
    except Exception as e:
        logger.error(f"Error optimizing memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    from ..dependencies import _agent, _dashboard
    
    return JSONResponse({
        "status": "healthy",
        "agent_initialized": _agent is not None,
        "dashboard_initialized": _dashboard is not None
    })




