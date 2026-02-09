"""
Comparison API for benchmarking and comparing robots and maintenance strategies.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .base_router import BaseRouter
from ..utils.data_helpers import (
    parse_time_range, filter_by_date_range, extract_sorted_dates,
    count_by_field, get_most_common_key,
    calculate_frequency_per_month, calculate_intervals, safe_average,
    find_max_by_key, find_min_by_key, round_decimal, increment_dict_value
)
from ..utils.file_helpers import datetime_to_iso, get_iso_timestamp

# Create base router instance
base = BaseRouter(
    prefix="/api/comparison",
    tags=["Comparison & Benchmarking"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class ComparisonRequest(BaseModel):
    """Request for comparison."""
    robot_types: List[str] = Field(..., min_items=2, max_items=10, description="Robot types to compare")
    metric: str = Field("maintenance_frequency", description="Metric to compare")
    time_range: Optional[str] = Field("30d", description="Time range: 7d, 30d, 90d, 1y, all")


@router.post("/robots")
@base.timed_endpoint("compare_robots")
async def compare_robots(
    request: ComparisonRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Compare multiple robot types across various metrics.
    """
    base.log_request("compare_robots", robot_types_count=len(request.robot_types), time_range=request.time_range)
    
    # Calculate date range using helper
    start_date, end_date = parse_time_range(request.time_range or "30d")
    
    comparison_data = {}
    
    for robot_type in request.robot_types:
        # Get maintenance history
        history = base.database.get_maintenance_history(robot_type=robot_type, limit=1000)
        
        # Filter by date range using helper
        filtered_history = filter_by_date_range(history, start_date, end_date)
        
        # Calculate metrics
        total_maintenances = len(filtered_history)
        days_span = (end_date - start_date).days
        frequency = calculate_frequency_per_month(total_maintenances, days_span)
        
        # Maintenance types distribution
        maint_types = count_by_field(filtered_history, "maintenance_type", "unknown")
        
        # Average time between maintenances
        dates = extract_sorted_dates(filtered_history)
        intervals = calculate_intervals(dates)
        avg_interval = safe_average(intervals, 0)
        
        comparison_data[robot_type] = {
            "total_maintenances": total_maintenances,
            "maintenance_frequency_per_month": round_decimal(frequency),
            "average_interval_days": round_decimal(avg_interval),
            "maintenance_types": maint_types,
            "most_common_type": get_most_common_key(maint_types)
        }
    
    # Calculate rankings using helpers
    rankings = {
        "lowest_frequency": find_min_by_key(
            comparison_data,
            key_func=lambda x: x["maintenance_frequency_per_month"]
        ),
        "highest_frequency": find_max_by_key(
            comparison_data,
            key_func=lambda x: x["maintenance_frequency_per_month"]
        ),
        "longest_interval": find_max_by_key(
            comparison_data,
            key_func=lambda x: x["average_interval_days"]
        ),
        "shortest_interval": find_min_by_key(
            comparison_data,
            key_func=lambda x: x["average_interval_days"]
        )
    }
    
    return base.success({
        "comparison": comparison_data,
        "rankings": {
            "lowest_frequency": {
                "robot_type": rankings["lowest_frequency"][0],
                "value": rankings["lowest_frequency"][1]["maintenance_frequency_per_month"]
            },
            "highest_frequency": {
                "robot_type": rankings["highest_frequency"][0],
                "value": rankings["highest_frequency"][1]["maintenance_frequency_per_month"]
            },
            "longest_interval": {
                "robot_type": rankings["longest_interval"][0],
                "value": rankings["longest_interval"][1]["average_interval_days"]
            },
            "shortest_interval": {
                "robot_type": rankings["shortest_interval"][0],
                "value": rankings["shortest_interval"][1]["average_interval_days"]
            }
        },
        "time_range": request.time_range,
        "timestamp": get_iso_timestamp()
    })


@router.get("/benchmark")
@base.timed_endpoint("get_benchmark_data")
async def get_benchmark_data(
    robot_type: str = Query(..., description="Robot type to benchmark"),
    metric: str = Query("maintenance_frequency", description="Metric to benchmark"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get benchmark data for a specific robot type.
    """
    base.log_request("get_benchmark_data", robot_type=robot_type, metric=metric)
    
    history = base.database.get_maintenance_history(robot_type=robot_type, limit=1000)
    
    if not history:
        return base.success({
            "robot_type": robot_type,
            "message": "Insufficient data for benchmarking",
            "recommendations": [
                "Collect more maintenance data",
                "Ensure regular maintenance logging"
            ]
        })
    
    # Calculate benchmarks
    total = len(history)
    maint_types = count_by_field(history, "maintenance_type", "unknown")
    
    # Calculate time-based metrics
    dates = extract_sorted_dates(history)
    
    if len(dates) > 1:
        time_span = (dates[-1] - dates[0]).days
        frequency = calculate_frequency_per_month(total, time_span)
        
        intervals = calculate_intervals(dates)
        avg_interval = safe_average(intervals, 0)
        min_interval = min(intervals) if intervals else 0
        max_interval = max(intervals) if intervals else 0
    else:
        frequency = 0
        avg_interval = 0
        min_interval = 0
        max_interval = 0
    
    benchmark = {
        "robot_type": robot_type,
        "total_maintenances": total,
        "maintenance_frequency_per_month": round_decimal(frequency),
        "average_interval_days": round_decimal(avg_interval),
        "min_interval_days": round_decimal(min_interval),
        "max_interval_days": round_decimal(max_interval),
        "maintenance_types_distribution": maint_types,
        "data_quality": {
            "total_records": total,
            "time_span_days": (dates[-1] - dates[0]).days if len(dates) > 1 else 0,
            "completeness": "good" if total > 10 else "needs_more_data"
        }
    }
    
    # Add recommendations
    recommendations = []
    if frequency > 4:
        recommendations.append("High maintenance frequency detected. Consider preventive maintenance optimization.")
    if avg_interval < 15:
        recommendations.append("Very short maintenance intervals. Review maintenance procedures.")
    if max(maint_types.values()) / total > 0.7:
        recommendations.append("One maintenance type dominates. Consider diversifying maintenance approach.")
    
    benchmark["recommendations"] = recommendations
    
    return base.success(benchmark)


@router.get("/trends")
@base.timed_endpoint("compare_trends")
async def compare_trends(
    robot_types: List[str] = Query(..., description="Robot types to compare trends"),
    days: int = Query(90, ge=7, le=365, description="Number of days to analyze"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Compare trends across multiple robot types.
    """
    base.log_request("compare_trends", robot_types_count=len(robot_types), days=days)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    trends = {}
    
    for robot_type in robot_types:
        history = base.database.get_maintenance_history(robot_type=robot_type, limit=1000)
        
        # Filter by date range using helper
        filtered = filter_by_date_range(history, start_date, end_date)
        
        # Group by week
        weekly_counts = {}
        for record in filtered:
            date_str = record.get("created_at")
            if date_str:
                from ..utils.file_helpers import parse_iso_date
                date = parse_iso_date(date_str)
                if date:
                    week = date.strftime("%Y-W%W")
                    increment_dict_value(weekly_counts, week)
        
        trends[robot_type] = {
            "weekly_counts": weekly_counts,
            "total": len(filtered),
            "trend": "increasing" if len(filtered) > 10 else "stable"
        }
    
    return base.success({
        "trends": trends,
        "period": {
            "start": datetime_to_iso(start_date),
            "end": datetime_to_iso(end_date),
            "days": days
        }
    })




