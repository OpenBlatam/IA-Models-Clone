"""
Metrics Routes
==============

API routes for metrics and events.
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..dependencies import get_metrics_collector, get_event_bus
from ...core.event_bus import EventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/{metric_name}")
async def get_metric(
    metric_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get metric data."""
    try:
        metrics_collector = get_metrics_collector()
        
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        points = metrics_collector.get_metric(metric_name, start, end)
        
        return JSONResponse({
            "metric_name": metric_name,
            "points": [p.to_dict() for p in points],
            "count": len(points)
        })
    except Exception as e:
        logger.error(f"Error getting metric: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{metric_name}/stats")
async def get_metric_stats(
    metric_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get metric statistics."""
    try:
        metrics_collector = get_metrics_collector()
        
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        stats = metrics_collector.get_statistics(metric_name, start, end)
        
        return JSONResponse({
            "metric_name": metric_name,
            "statistics": stats
        })
    except Exception as e:
        logger.error(f"Error getting metric stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{histogram_name}/percentiles")
async def get_histogram_percentiles(
    histogram_name: str,
    percentiles: Optional[str] = None
):
    """Get histogram percentiles."""
    try:
        metrics_collector = get_metrics_collector()
        
        if percentiles:
            p_list = [float(p) for p in percentiles.split(",")]
        else:
            p_list = [50, 75, 90, 95, 99]
        
        result = metrics_collector.get_percentiles(histogram_name, p_list)
        
        return JSONResponse({
            "histogram_name": histogram_name,
            "percentiles": result
        })
    except Exception as e:
        logger.error(f"Error getting percentiles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/history")
async def get_event_history(
    event_type: Optional[str] = None,
    limit: int = 100
):
    """Get event history."""
    try:
        event_bus = get_event_bus()
        
        etype = EventType(event_type) if event_type else None
        events = event_bus.get_history(etype, limit)
        
        return JSONResponse({
            "events": [e.to_dict() for e in events],
            "count": len(events)
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    except Exception as e:
        logger.error(f"Error getting event history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




