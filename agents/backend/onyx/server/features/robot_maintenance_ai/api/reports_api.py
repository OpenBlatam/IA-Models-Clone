"""
Advanced Reports API for generating comprehensive maintenance reports.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from .base_router import BaseRouter
from .exceptions import ValidationError
from ..utils.file_helpers import get_iso_timestamp, get_date_range, parse_iso_date, datetime_to_iso
from ..utils.data_helpers import (
    count_by_key, count_by_field, count_by_function,
    get_most_common_key, find_most_common,
    calculate_average_interval, calculate_intervals, calculate_frequency_per_month,
    safe_average, safe_divide, round_decimal, ensure_minimum, accumulate_dict_value
)

# Create base router instance
base = BaseRouter(
    prefix="/api/reports",
    tags=["Reports"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class ReportRequest(BaseModel):
    """Request for generating a report."""
    report_type: str = Field(..., description="Type of report: summary, detailed, predictive, cost")
    robot_type: Optional[str] = Field(None, description="Filter by robot type")
    start_date: Optional[str] = Field(None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")
    include_recommendations: bool = Field(True, description="Include recommendations in report")


@router.post("/generate")
@base.timed_endpoint("generate_report")
async def generate_report(
    request: ReportRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Generate a comprehensive maintenance report.
    """
    base.log_request("generate_report", report_type=request.report_type, robot_type=request.robot_type)
    
    # Set date range using helper
    start_date, end_date = get_date_range(
        start_date=request.start_date,
        end_date=request.end_date,
        default_days=30
    )
    
    # Get data
    history = base.database.get_maintenance_history(robot_type=request.robot_type, limit=10000)
    filtered_history = [
        h for h in history
        if h.get("created_at") and 
        (parsed_date := parse_iso_date(h["created_at"])) and
        start_date <= parsed_date <= end_date
    ]
    
    # Generate report based on type
    if request.report_type == "summary":
        report = _generate_summary_report(filtered_history, start_date, end_date, request.robot_type)
    elif request.report_type == "detailed":
        report = _generate_detailed_report(filtered_history, start_date, end_date, request.robot_type)
    elif request.report_type == "predictive":
        report = _generate_predictive_report(filtered_history, start_date, end_date, request.robot_type)
    elif request.report_type == "cost":
        report = _generate_cost_report(filtered_history, start_date, end_date, request.robot_type)
    else:
        raise ValidationError(f"Invalid report type: {request.report_type}")
    
    # Add recommendations if requested
    if request.include_recommendations:
        report["recommendations"] = _generate_recommendations(filtered_history)
    
    report["metadata"] = {
        "generated_at": get_iso_timestamp(),
        "report_type": request.report_type,
        "period": {
            "start": datetime_to_iso(start_date),
            "end": datetime_to_iso(end_date)
        },
        "robot_type": request.robot_type or "all"
    }
    
    return base.success(report)


def _generate_summary_report(history: List[Dict], start_date: datetime, end_date: datetime, robot_type: Optional[str]) -> Dict[str, Any]:
    """Generate summary report."""
    total = len(history)
    
    # Maintenance types
    maint_types = count_by_field(history, "maintenance_type", "unknown")
    
    # Time metrics
    dates = sorted([
        parsed_date
        for h in history
        if h.get("created_at") and (parsed_date := parse_iso_date(h["created_at"]))
    ])
    
    days_span = (end_date - start_date).days
    frequency = safe_divide(total, ensure_minimum(days_span), 0) * 30
    
    intervals = calculate_intervals(dates)
    avg_interval = safe_average(intervals, 0)
    
    return {
        "type": "summary",
        "summary": {
            "total_maintenances": total,
            "period_days": (end_date - start_date).days,
            "maintenance_frequency_per_month": round_decimal(frequency),
            "average_interval_days": round_decimal(avg_interval)
        },
        "breakdown": {
            "by_type": maint_types,
            "most_common": get_most_common_key(maint_types)
        }
    }


def _generate_detailed_report(history: List[Dict], start_date: datetime, end_date: datetime, robot_type: Optional[str]) -> Dict[str, Any]:
    """Generate detailed report."""
    summary = _generate_summary_report(history, start_date, end_date, robot_type)
    
    # Add detailed breakdown
    monthly_breakdown = count_by_function(
        [h for h in history if h.get("created_at")],
        lambda h: parse_iso_date(h["created_at"]).strftime("%Y-%m") if parse_iso_date(h["created_at"]) else None
    )
    
    return {
        "type": "detailed",
        **summary,
        "monthly_breakdown": monthly_breakdown,
        "records": history[:100]  # Limit to first 100 for size
    }


def _generate_predictive_report(history: List[Dict], start_date: datetime, end_date: datetime, robot_type: Optional[str]) -> Dict[str, Any]:
    """Generate predictive report."""
    summary = _generate_summary_report(history, start_date, end_date, robot_type)
    
    # Predictions
    dates = sorted([
        parsed_date
        for h in history
        if h.get("created_at") and (parsed_date := parse_iso_date(h["created_at"]))
    ])
    
    predictions = {}
    if len(dates) > 1:
        intervals = calculate_intervals(dates)
        avg_interval = safe_average(intervals, 0)
        
        if dates and avg_interval > 0:
            last_maintenance = dates[-1]
            next_predicted = last_maintenance + timedelta(days=avg_interval)
            predictions = {
                "next_maintenance_predicted": datetime_to_iso(next_predicted),
                "days_until_next": round((next_predicted - datetime.now()).days, 1),
                "confidence": "medium" if len(intervals) > 5 else "low"
            }
    
    return {
        "type": "predictive",
        **summary,
        "predictions": predictions
    }


def _generate_cost_report(history: List[Dict], start_date: datetime, end_date: datetime, robot_type: Optional[str]) -> Dict[str, Any]:
    """Generate cost analysis report."""
    summary = _generate_summary_report(history, start_date, end_date, robot_type)
    
    # Cost estimates (placeholder - would need actual cost data)
    cost_estimates = {
        "preventive": 500,
        "corrective": 1500,
        "predictive": 300,
        "emergency": 3000,
        "calibration": 200
    }
    
    total_estimated_cost = 0
    cost_by_type = {}
    
    for record in history:
        maint_type = record.get("maintenance_type", "preventive")
        cost = cost_estimates.get(maint_type, 500)
        total_estimated_cost += cost
        accumulate_dict_value(cost_by_type, maint_type, cost)
    
    return {
        "type": "cost",
        **summary,
        "cost_analysis": {
            "total_estimated_cost": round_decimal(total_estimated_cost),
            "average_cost_per_maintenance": round_decimal(total_estimated_cost / len(history)) if history else 0,
            "cost_by_type": cost_by_type,
            "monthly_cost_estimate": round_decimal(total_estimated_cost / ensure_minimum((end_date - start_date).days) * 30)
        }
    }


def _generate_recommendations(history: List[Dict]) -> List[str]:
    """Generate recommendations based on history."""
    recommendations = []
    
    if len(history) > 20:
        recommendations.append("Consider implementing predictive maintenance to reduce frequency")
    
    maint_types = count_by_field(history, "maintenance_type", "unknown")
    
    if maint_types.get("corrective", 0) > maint_types.get("preventive", 0):
        recommendations.append("High corrective maintenance ratio. Increase preventive maintenance.")
    
    if len(history) < 5:
        recommendations.append("Insufficient data for accurate analysis. Continue logging maintenance activities.")
    
    return recommendations


@router.get("/types")
@base.timed_endpoint("get_report_types")
async def get_report_types(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get available report types and their descriptions.
    """
    base.log_request("get_report_types")
    
    return base.success({
        "report_types": [
            {
                "type": "summary",
                "description": "High-level summary with key metrics",
                "includes": ["total maintenances", "frequency", "breakdown by type"]
            },
            {
                "type": "detailed",
                "description": "Comprehensive report with full details",
                "includes": ["summary", "monthly breakdown", "detailed records"]
            },
            {
                "type": "predictive",
                "description": "Report with predictions and forecasts",
                "includes": ["summary", "next maintenance prediction", "trends"]
            },
            {
                "type": "cost",
                "description": "Cost analysis and financial insights",
                "includes": ["summary", "cost breakdown", "cost estimates"]
            }
        ]
    })




