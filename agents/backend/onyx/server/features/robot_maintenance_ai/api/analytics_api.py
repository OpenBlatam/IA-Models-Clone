"""
Analytics API endpoints for reporting and statistics.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from .base_router import BaseRouter
from ..utils.metrics import metrics_collector
from ..utils.data_helpers import count_by_key, parse_time_range, count_by_function, filter_by_fields, round_decimal, get_nested_value, increment_dict_value, count_matching
from ..utils.file_helpers import datetime_to_iso, get_iso_timestamp, parse_iso_date, extract_date_from_iso

# Create base router instance
base = BaseRouter(
    prefix="/api/analytics",
    tags=["Analytics"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class AnalyticsTimeRange(BaseModel):
    """Time range for analytics queries."""
    start_date: Optional[str] = Field(None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")


@router.get("/overview")
@base.timed_endpoint("analytics_overview")
async def get_analytics_overview(
    time_range: Optional[str] = Query("7d", description="Time range: 1d, 7d, 30d, 90d, all"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get comprehensive analytics overview.
    """
    base.log_request("analytics_overview", time_range=time_range)
    
    # Calculate date range using helper
    start_date, end_date = parse_time_range(time_range or "7d")
    
    # Get conversation statistics
    conversations = base.database.get_conversations_by_date_range(
        datetime_to_iso(start_date),
        datetime_to_iso(end_date)
    )
    
    # Get message statistics
    messages = base.database.get_messages_by_date_range(
        datetime_to_iso(start_date),
        datetime_to_iso(end_date)
    )
    
    # Calculate metrics
    total_conversations = len(conversations)
    total_messages = len(messages)
    avg_messages_per_conversation = total_messages / total_conversations if total_conversations > 0 else 0
    
    # Robot type distribution
    robot_types = count_by_key(conversations, "robot_type")
    
    # Maintenance type distribution
    maintenance_types = count_by_key(conversations, "maintenance_type")
    
    # Daily activity
    daily_activity = count_by_function(
        messages,
        lambda msg: extract_date_from_iso(msg.get("timestamp"))
    )
    
    return base.success({
        "time_range": {
            "start": datetime_to_iso(start_date),
            "end": datetime_to_iso(end_date),
            "range": time_range
        },
        "conversations": {
            "total": total_conversations,
            "avg_messages": round_decimal(avg_messages_per_conversation)
        },
        "messages": {
            "total": total_messages
        },
        "distribution": {
            "robot_types": robot_types,
            "maintenance_types": maintenance_types
        },
        "daily_activity": daily_activity,
        "api_metrics": metrics_collector.get_stats()
    })


@router.get("/conversations/stats")
@base.timed_endpoint("conversation_stats")
async def get_conversation_stats(
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    maintenance_type: Optional[str] = Query(None, description="Filter by maintenance type"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get detailed conversation statistics.
    """
    base.log_request("conversation_stats", robot_type=robot_type, maintenance_type=maintenance_type)
    
    conversations = base.database.get_all_conversations()
    
    # Apply filters using helper
    conversations = filter_by_fields(
        conversations,
        {
            "robot_type": robot_type,
            "maintenance_type": maintenance_type
        }
    )
    
    # Calculate statistics
    total = len(conversations)
    avg_message_count = sum(c.get("message_count", 0) for c in conversations) / total if total > 0 else 0
    
    # Time-based statistics
    created_dates = [c.get("created_at", "") for c in conversations]
    now = datetime.now()
    recent_7d = sum(1 for d in created_dates if d and (now - parse_iso_date(d, now)).days <= 7)
    recent_30d = sum(1 for d in created_dates if d and (now - parse_iso_date(d, now)).days <= 30)
    
    return base.success({
        "total_conversations": total,
        "average_messages": round_decimal(avg_message_count),
        "recent_activity": {
            "last_7_days": recent_7d,
            "last_30_days": recent_30d
        },
        "filters": {
            "robot_type": robot_type,
            "maintenance_type": maintenance_type
        }
    })


@router.get("/performance")
@base.timed_endpoint("performance_metrics")
async def get_performance_metrics(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get system performance metrics.
    """
    base.log_request("performance_metrics")
    
    metrics = metrics_collector.get_stats()
    
    # Calculate additional performance metrics
    endpoint_stats = metrics.get("endpoint_stats", {})
    
    # Find fastest and slowest endpoints
    fastest = None
    slowest = None
    fastest_time = float('inf')
    slowest_time = 0
    
    for endpoint, stats in endpoint_stats.items():
        avg_time = float(stats.get("avg_response_time_seconds", "0"))
        if avg_time < fastest_time and avg_time > 0:
            fastest_time = avg_time
            fastest = endpoint
        if avg_time > slowest_time:
            slowest_time = avg_time
            slowest = endpoint
    
    return base.success({
        "overall": {
            "uptime_seconds": metrics.get("uptime_seconds", "0"),
            "total_requests": metrics.get("total_requests", 0),
            "cache_hit_rate": get_nested_value(metrics, "cache_stats", "hit_rate", default="0%")
        },
        "endpoints": {
            "fastest": {
                "endpoint": fastest,
                "avg_time": f"{fastest_time:.4f}s" if fastest else "N/A"
            },
            "slowest": {
                "endpoint": slowest,
                "avg_time": f"{slowest_time:.4f}s" if slowest else "N/A"
            },
            "all": endpoint_stats
        },
        "cache": metrics.get("cache_stats", {})
    })


@router.get("/trends")
@base.timed_endpoint("trends")
async def get_trends(
    metric: str = Query("conversations", description="Metric: conversations, messages, api_calls"),
    days: int = Query(30, description="Number of days to analyze"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get trend analysis for specified metric.
    """
    base.log_request("trends", metric=metric, days=days)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    trends = []
    
    if metric == "conversations":
        conversations = base.database.get_conversations_by_date_range(
            datetime_to_iso(start_date),
            datetime_to_iso(end_date)
        )
        # Group by day
        daily_counts = {}
        for conv in conversations:
            date = extract_date_from_iso(conv.get("created_at"))
            if date:
                increment_dict_value(daily_counts, date)
        
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            trends.append({
                "date": date,
                "count": daily_counts.get(date, 0)
            })
    
    elif metric == "messages":
        messages = base.database.get_messages_by_date_range(
            datetime_to_iso(start_date),
            datetime_to_iso(end_date)
        )
        # Group by day
        daily_counts = {}
        for msg in messages:
            date = extract_date_from_iso(msg.get("timestamp"))
            if date:
                increment_dict_value(daily_counts, date)
        
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            trends.append({
                "date": date,
                "count": daily_counts.get(date, 0)
            })
    
    elif metric == "api_calls":
        # Use metrics collector
        metrics_data = metrics_collector.get_stats()
        # For now, return overall stats
        trends = [{
            "date": end_date.strftime("%Y-%m-%d"),
            "count": metrics_data.get("total_requests", 0)
        }]
    
    return base.success({
        "metric": metric,
        "days": days,
        "trends": trends
    })




