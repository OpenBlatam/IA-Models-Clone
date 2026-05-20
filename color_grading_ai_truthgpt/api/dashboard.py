"""
Dashboard API for Color Grading AI
===================================

Dashboard endpoints for monitoring and statistics.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


async def get_dashboard_stats(agent) -> Dict[str, Any]:
    """
    Get dashboard statistics.
    
    Returns:
        Dashboard statistics
    """
    # Get metrics
    metrics = agent.get_metrics()
    
    # Get resource stats
    resources = agent.get_resource_stats()
    
    # Get queue status
    queue_status = {}
    try:
        import asyncio
        if asyncio.iscoroutinefunction(agent.task_queue.get_queue_status):
            queue_status = await agent.task_queue.get_queue_status()
        else:
            queue_status = agent.task_queue.get_queue_status()
    except:
        pass
    
    # Get recent history
    recent_history = agent.get_history(limit=10)
    
    # Get template stats
    template_stats = agent.metrics_collector.get_template_stats()
    
    # Calculate totals
    overall = metrics.get("overall", {})
    total_operations = overall.get("total_operations", 0)
    successful = overall.get("successful", 0)
    failed = overall.get("failed", 0)
    success_rate = overall.get("success_rate", 0)
    
    return {
        "summary": {
            "total_operations": total_operations,
            "successful": successful,
            "failed": failed,
            "success_rate": success_rate,
            "queue_size": queue_status.get("queue_size", 0),
        },
        "performance": {
            "avg_duration": metrics.get("grade_video_video", {}).get("avg_duration", 0),
            "total_duration": metrics.get("grade_video_video", {}).get("total_duration", 0),
        },
        "resources": resources,
        "queue": queue_status,
        "templates": template_stats,
        "recent_operations": recent_history[:5],
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/stats")
async def get_stats():
    """Get dashboard statistics."""
    from .color_grading_api import agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return await get_dashboard_stats(agent)


@router.get("/metrics")
async def get_metrics():
    """Get processing metrics."""
    from .color_grading_api import agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return agent.get_metrics()


@router.get("/resources")
async def get_resources():
    """Get resource statistics."""
    from .color_grading_api import agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return agent.get_resource_stats()


@router.get("/queue")
async def get_queue_status():
    """Get queue status."""
    from .color_grading_api import agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        return await agent.task_queue.get_queue_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/recent")
async def get_recent_history(limit: int = 10):
    """Get recent processing history."""
    from .color_grading_api import agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return agent.get_history(limit=limit)


@router.get("/templates/stats")
async def get_template_stats():
    """Get template usage statistics."""
    from .color_grading_api import agent
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return agent.metrics_collector.get_template_stats()

