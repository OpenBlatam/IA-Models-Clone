"""
Analytics API Endpoints
=======================

REST API endpoints for analytics and reporting.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..services.analytics_service import AnalyticsService, ReportType, MetricType
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/analytics", tags=["Analytics"])

# Pydantic models
class MetricRequest(BaseModel):
    name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}

class ReportRequest(BaseModel):
    report_type: ReportType
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    include_charts: bool = True

class ReportResponse(BaseModel):
    id: str
    name: str
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    data: Dict[str, Any]
    charts: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

# Global analytics service instance
analytics_service = None

def get_analytics_service() -> AnalyticsService:
    """Get global analytics service instance."""
    global analytics_service
    if analytics_service is None:
        analytics_service = AnalyticsService({})
    return analytics_service

# API Endpoints

@router.post("/metrics")
async def record_metric(
    request: MetricRequest,
    current_user: User = Depends(require_permission("analytics:create"))
):
    """Record a new metric."""
    
    analytics_service = get_analytics_service()
    
    await analytics_service.record_metric(
        name=request.name,
        value=request.value,
        metric_type=request.metric_type,
        tags=request.tags,
        metadata=request.metadata
    )
    
    return {"message": "Metric recorded successfully"}

@router.get("/metrics")
async def list_metrics(
    metric_name: Optional[str] = Query(None, description="Filter by metric name"),
    metric_type: Optional[MetricType] = Query(None, description="Filter by metric type"),
    limit: int = Query(100, description="Maximum number of metrics to return"),
    current_user: User = Depends(require_permission("analytics:view"))
):
    """List metrics with optional filters."""
    
    analytics_service = get_analytics_service()
    metrics = analytics_service.metrics
    
    # Apply filters
    if metric_name:
        metrics = [m for m in metrics if m.name == metric_name]
    
    if metric_type:
        metrics = [m for m in metrics if m.metric_type == metric_type]
    
    # Limit results
    metrics = metrics[-limit:] if limit else metrics
    
    return {
        "metrics": [
            {
                "name": m.name,
                "value": m.value,
                "metric_type": m.metric_type.value,
                "timestamp": m.timestamp.isoformat(),
                "tags": m.tags,
                "metadata": m.metadata
            }
            for m in metrics
        ],
        "total_count": len(metrics)
    }

@router.get("/performance")
async def get_performance_metrics(
    period_start: Optional[datetime] = Query(None, description="Start of period"),
    period_end: Optional[datetime] = Query(None, description="End of period"),
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Get performance metrics."""
    
    analytics_service = get_analytics_service()
    
    metrics = await analytics_service.get_performance_metrics(
        period_start=period_start,
        period_end=period_end
    )
    
    return {
        "total_workflows": metrics.total_workflows,
        "completed_workflows": metrics.completed_workflows,
        "failed_workflows": metrics.failed_workflows,
        "average_execution_time": metrics.average_execution_time,
        "success_rate": metrics.success_rate,
        "total_documents_generated": metrics.total_documents_generated,
        "average_document_size": metrics.average_document_size,
        "agent_utilization": metrics.agent_utilization,
        "business_area_distribution": metrics.business_area_distribution
    }

@router.post("/reports/performance", response_model=ReportResponse)
async def generate_performance_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Generate performance report."""
    
    analytics_service = get_analytics_service()
    
    report = await analytics_service.generate_performance_report(
        period_start=request.period_start,
        period_end=request.period_end
    )
    
    return ReportResponse(
        id=report.id,
        name=report.name,
        report_type=report.report_type.value,
        period_start=report.period_start,
        period_end=report.period_end,
        generated_at=report.generated_at,
        data=report.data,
        charts=report.charts if request.include_charts else [],
        insights=report.insights,
        recommendations=report.recommendations,
        metadata=report.metadata
    )

@router.post("/reports/usage", response_model=ReportResponse)
async def generate_usage_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Generate usage report."""
    
    analytics_service = get_analytics_service()
    
    report = await analytics_service.generate_usage_report(
        period_start=request.period_start,
        period_end=request.period_end
    )
    
    return ReportResponse(
        id=report.id,
        name=report.name,
        report_type=report.report_type.value,
        period_start=report.period_start,
        period_end=report.period_end,
        generated_at=report.generated_at,
        data=report.data,
        charts=report.charts if request.include_charts else [],
        insights=report.insights,
        recommendations=report.recommendations,
        metadata=report.metadata
    )

@router.get("/reports")
async def list_reports(
    report_type: Optional[ReportType] = Query(None, description="Filter by report type"),
    limit: int = Query(50, description="Maximum number of reports to return"),
    current_user: User = Depends(require_permission("analytics:view"))
):
    """List analytics reports."""
    
    analytics_service = get_analytics_service()
    
    reports = analytics_service.list_reports(report_type=report_type)
    
    # Sort by generation time (newest first)
    reports.sort(key=lambda x: x.generated_at, reverse=True)
    
    # Limit results
    reports = reports[:limit] if limit else reports
    
    return {
        "reports": [
            {
                "id": r.id,
                "name": r.name,
                "report_type": r.report_type.value,
                "period_start": r.period_start.isoformat(),
                "period_end": r.period_end.isoformat(),
                "generated_at": r.generated_at.isoformat(),
                "insights_count": len(r.insights),
                "recommendations_count": len(r.recommendations),
                "charts_count": len(r.charts)
            }
            for r in reports
        ],
        "total_count": len(reports)
    }

@router.get("/reports/{report_id}")
async def get_report(
    report_id: str,
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Get specific report by ID."""
    
    analytics_service = get_analytics_service()
    
    report = analytics_service.get_report(report_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Report {report_id} not found"
        )
    
    return ReportResponse(
        id=report.id,
        name=report.name,
        report_type=report.report_type.value,
        period_start=report.period_start,
        period_end=report.period_end,
        generated_at=report.generated_at,
        data=report.data,
        charts=report.charts,
        insights=report.insights,
        recommendations=report.recommendations,
        metadata=report.metadata
    )

@router.get("/reports/{report_id}/export")
async def export_report(
    report_id: str,
    format: str = Query("json", description="Export format (json, csv)"),
    current_user: User = Depends(require_permission("analytics:export"))
):
    """Export report data."""
    
    analytics_service = get_analytics_service()
    
    report = analytics_service.get_report(report_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Report {report_id} not found"
        )
    
    try:
        export_data = await analytics_service.export_report_data(report_id, format)
        
        return {
            "report_id": report_id,
            "format": format,
            "data": export_data,
            "exported_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.get("/health-score")
async def get_system_health_score(
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Get overall system health score."""
    
    analytics_service = get_analytics_service()
    
    health_score = await analytics_service.get_system_health_score()
    
    # Determine health status
    if health_score >= 90:
        status = "excellent"
    elif health_score >= 80:
        status = "good"
    elif health_score >= 70:
        status = "fair"
    elif health_score >= 60:
        status = "poor"
    else:
        status = "critical"
    
    return {
        "health_score": health_score,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "recommendations": [
            "Monitor system performance regularly",
            "Review failed workflows and optimize",
            "Ensure proper resource allocation",
            "Maintain system documentation"
        ] if health_score < 80 else [
            "System is performing well",
            "Continue monitoring key metrics",
            "Consider optimization opportunities"
        ]
    }

@router.get("/dashboard")
async def get_analytics_dashboard(
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Get analytics dashboard data."""
    
    analytics_service = get_analytics_service()
    
    # Get recent performance metrics
    performance_metrics = await analytics_service.get_performance_metrics()
    
    # Get system health score
    health_score = await analytics_service.get_system_health_score()
    
    # Get recent reports
    recent_reports = analytics_service.list_reports()
    recent_reports.sort(key=lambda x: x.generated_at, reverse=True)
    recent_reports = recent_reports[:5]
    
    # Calculate trends (simplified)
    total_metrics = len(analytics_service.metrics)
    recent_metrics = len([
        m for m in analytics_service.metrics 
        if m.timestamp >= datetime.now() - timedelta(hours=24)
    ])
    
    return {
        "overview": {
            "total_workflows": performance_metrics.total_workflows,
            "success_rate": performance_metrics.success_rate,
            "total_documents": performance_metrics.total_documents_generated,
            "health_score": health_score
        },
        "performance": {
            "completed_workflows": performance_metrics.completed_workflows,
            "failed_workflows": performance_metrics.failed_workflows,
            "average_execution_time": performance_metrics.average_execution_time,
            "agent_utilization": performance_metrics.agent_utilization
        },
        "business_areas": performance_metrics.business_area_distribution,
        "recent_reports": [
            {
                "id": r.id,
                "name": r.name,
                "type": r.report_type.value,
                "generated_at": r.generated_at.isoformat()
            }
            for r in recent_reports
        ],
        "activity": {
            "total_metrics": total_metrics,
            "recent_metrics_24h": recent_metrics,
            "metrics_trend": "increasing" if recent_metrics > total_metrics * 0.1 else "stable"
        }
    }

@router.get("/metrics/summary")
async def get_metrics_summary(
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Get metrics summary."""
    
    analytics_service = get_analytics_service()
    
    # Group metrics by name
    metric_groups = {}
    for metric in analytics_service.metrics:
        if metric.name not in metric_groups:
            metric_groups[metric.name] = []
        metric_groups[metric.name].append(metric)
    
    summary = {}
    for name, metrics in metric_groups.items():
        values = [m.value for m in metrics]
        summary[name] = {
            "count": len(metrics),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "avg": sum(values) / len(values) if values else 0,
            "latest": values[-1] if values else 0,
            "latest_timestamp": metrics[-1].timestamp.isoformat() if metrics else None
        }
    
    return {
        "metric_summary": summary,
        "total_metrics": len(analytics_service.metrics),
        "unique_metric_names": len(metric_groups),
        "generated_at": datetime.now().isoformat()
    }





























