"""
Real-time Dashboard API for monitoring and visualization.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from .base_router import BaseRouter
from ..utils.metrics import metrics_collector
from ..utils.file_helpers import get_iso_timestamp, datetime_to_iso, parse_iso_date, extract_date_from_iso
from ..utils.data_helpers import round_decimal, get_nested_value, ensure_minimum, increment_dict_value

# Create base router instance
base = BaseRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


@router.get("/overview")
@base.timed_endpoint("get_dashboard_overview")
async def get_dashboard_overview(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get comprehensive dashboard overview with real-time metrics.
    """
    base.log_request("get_dashboard_overview")
    
    # Get recent activity (last 24 hours)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    recent_conversations = base.database.get_conversations_by_date_range(
        datetime_to_iso(start_date),
        datetime_to_iso(end_date)
    )
    
    recent_messages = base.database.get_messages_by_date_range(
        datetime_to_iso(start_date),
        datetime_to_iso(end_date)
    )
    
    # Get all-time stats
    all_conversations = base.database.get_all_conversations()
    all_history = base.database.get_maintenance_history(limit=1000)
    
    # Calculate metrics
    active_robots = len(set(c.get("robot_type") for c in all_conversations if c.get("robot_type")))
    
    # Maintenance stats
    maint_by_type = {}
    for record in all_history:
        maint_type = record.get("maintenance_type", "unknown")
        increment_dict_value(maint_by_type, maint_type)
    
    # Recent activity trend (last 7 days)
    daily_activity = {}
    for i in range(7):
        date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_activity[date] = sum(
            1 for msg in recent_messages
            if msg.get("timestamp", "").startswith(date)
        )
    
    # API metrics
    api_metrics = metrics_collector.get_stats()
    
    overview = {
        "timestamp": get_iso_timestamp(),
        "summary": {
            "total_conversations": len(all_conversations),
            "active_robots": active_robots,
            "total_maintenances": len(all_history),
            "recent_activity_24h": {
                "conversations": len(recent_conversations),
                "messages": len(recent_messages)
            }
        },
        "maintenance": {
            "by_type": maint_by_type,
            "most_common": max(maint_by_type.items(), key=lambda x: x[1])[0] if maint_by_type else None
        },
        "activity": {
            "daily_trend": daily_activity,
            "peak_day": max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else None
        },
        "api_performance": {
            "total_requests": api_metrics.get("total_requests", 0),
            "uptime_seconds": api_metrics.get("uptime_seconds", "0"),
            "cache_hit_rate": get_nested_value(api_metrics, "cache_stats", "hit_rate", default="0%")
        },
        "alerts": {
            "active": 0,  # Would come from alerts system
            "critical": 0,
            "warning": 0
        }
    }
    
    return base.success(overview)


@router.get("/metrics/realtime")
@base.timed_endpoint("get_realtime_metrics")
async def get_realtime_metrics(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get real-time system metrics.
    """
    base.log_request("get_realtime_metrics")
    
    metrics = metrics_collector.get_stats()
    
    # Get endpoint performance
    endpoint_stats = metrics.get("endpoint_stats", {})
    
    # Calculate current rate
    current_rate = {}
    for endpoint, stats in endpoint_stats.items():
        total = stats.get("total_requests", 0)
        uptime = float(metrics.get("uptime_seconds", "1"))
        current_rate[endpoint] = {
            "requests_per_minute": round_decimal(total / ensure_minimum(uptime / 60)),
            "avg_response_time": stats.get("avg_response_time_seconds", "0"),
            "error_rate": stats.get("error_rate", 0)
        }
    
    return base.success({
        "timestamp": get_iso_timestamp(),
        "uptime_seconds": metrics.get("uptime_seconds", "0"),
        "total_requests": metrics.get("total_requests", 0),
        "endpoint_rates": current_rate,
        "cache": metrics.get("cache_stats", {}),
        "system_health": "healthy"  # Could be calculated based on error rates
    })


@router.get("/widgets/maintenance-status")
@base.timed_endpoint("get_maintenance_status_widget")
async def get_maintenance_status_widget(
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get maintenance status widget data.
    """
    base.log_request("get_maintenance_status_widget", robot_type=robot_type)
    
    history = base.database.get_maintenance_history(robot_type=robot_type, limit=100)
    
    # Calculate status
    if not history:
        status = "no_data"
        last_maintenance = None
    else:
        last_maintenance = max(
            (h for h in history if h.get("created_at")),
            key=lambda x: parse_iso_date(x["created_at"], datetime.min) or datetime.min,
            default=None
        )
        
        if last_maintenance:
            last_date = parse_iso_date(last_maintenance["created_at"])
            if last_date:
                days_since = (datetime.now() - last_date).days
            else:
                days_since = 999
            
            if days_since < 7:
                status = "recent"
            elif days_since < 30:
                status = "due_soon"
            elif days_since < 60:
                status = "overdue"
            else:
                status = "critical"
        else:
            status = "no_data"
    
    return base.success({
        "status": status,
        "last_maintenance": last_maintenance.get("created_at") if last_maintenance else None,
        "total_maintenances": len(history),
        "robot_type": robot_type or "all"
    })


@router.get("/widgets/activity-chart")
@base.timed_endpoint("get_activity_chart_data")
async def get_activity_chart_data(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get activity chart data for visualization.
    """
    base.log_request("get_activity_chart_data", days=days)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    messages = base.database.get_messages_by_date_range(
        datetime_to_iso(start_date),
        datetime_to_iso(end_date)
    )
    
    # Group by day
    daily_data = {}
    for msg in messages:
        date = extract_date_from_iso(msg.get("timestamp"))
        if date:
            increment_dict_value(daily_data, date)
    
    # Fill missing days with 0
    chart_data = []
    for i in range(days):
        date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        chart_data.append({
            "date": date,
            "count": daily_data.get(date, 0)
        })
    
    return base.success({
        "chart_data": chart_data,
        "total": len(messages),
        "average_per_day": round_decimal(len(messages) / days),
        "peak_day": max(chart_data, key=lambda x: x["count"]) if chart_data else None
    })


@router.get("/widgets/robot-distribution")
@base.timed_endpoint("get_robot_distribution_widget")
async def get_robot_distribution_widget(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get robot type distribution widget data.
    """
    base.log_request("get_robot_distribution_widget")
    
    conversations = base.database.get_all_conversations()
    
    # Count by robot type
    distribution = {}
    for conv in conversations:
        robot_type = conv.get("robot_type", "unknown")
        increment_dict_value(distribution, robot_type)
    
    # Calculate percentages
    total = sum(distribution.values())
    distribution_with_percent = {
        robot_type: {
            "count": count,
            "percentage": round_decimal(count / total * 100) if total > 0 else 0
        }
        for robot_type, count in distribution.items()
    }
    
    return base.success({
        "distribution": distribution_with_percent,
        "total": total,
        "unique_types": len(distribution)
    })




