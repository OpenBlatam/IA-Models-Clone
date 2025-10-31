"""
Advanced Analytics API Endpoints
===============================

Comprehensive API endpoints for analytics and reporting in business agents system.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import redis

from ....schemas import (
    AgentAnalytics, WorkflowAnalytics, CollaborationAnalytics,
    SystemAnalytics, AnalyticsRequest, AnalyticsResponse,
    PerformanceMetrics, TrendAnalysis, InsightGeneration,
    ErrorResponse
)
from ....exceptions import (
    AnalyticsError, AnalyticsNotFoundError, AnalyticsValidationError,
    AnalyticsPermissionDeniedError, AnalyticsSystemError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ....services import AnalyticsService
from ....middleware.auth_middleware import get_current_user, require_permissions, require_roles
from ....config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])
security = HTTPBearer()


async def get_db_session() -> AsyncSession:
    """Get database session dependency"""
    pass


async def get_redis_client() -> redis.Redis:
    """Get Redis client dependency"""
    settings = get_settings()
    return redis.Redis(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password,
        db=settings.redis.db
    )


async def get_analytics_service(
    db: AsyncSession = Depends(get_db_session),
    redis: redis.Redis = Depends(get_redis_client)
) -> AnalyticsService:
    """Get analytics service dependency"""
    return AnalyticsService(db, redis)


# System Analytics Endpoints
@router.get("/system", response_model=SystemAnalytics)
async def get_system_analytics(
    time_period: str = Query("30d", description="Time period for analytics"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get comprehensive system analytics"""
    try:
        result = await analytics_service.get_system_analytics(time_period)
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/dashboard", response_model=Dict[str, Any])
async def get_dashboard_data(
    time_period: str = Query("7d", description="Time period for dashboard"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get real-time dashboard data"""
    try:
        result = await analytics_service.get_dashboard_data(time_period)
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/trends", response_model=TrendAnalysis)
async def get_performance_trends(
    metric_type: str = Query("all", description="Type of metrics to analyze"),
    time_period: str = Query("90d", description="Time period for trend analysis"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get performance trends analysis"""
    try:
        result = await analytics_service.get_performance_trends(metric_type, time_period)
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


# Agent Analytics Endpoints
@router.get("/agents", response_model=Dict[str, Any])
async def get_agents_analytics(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    time_period: str = Query("30d", description="Time period for analytics"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get comprehensive agents analytics"""
    try:
        result = await analytics_service.get_agents_analytics(
            agent_type, category, time_period, page, per_page
        )
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/agents/{agent_id}", response_model=AgentAnalytics)
async def get_agent_analytics(
    agent_id: str = Path(..., description="Agent ID"),
    time_period: str = Query("30d", description="Time period for analytics"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get detailed analytics for specific agent"""
    try:
        result = await analytics_service.get_agent_analytics(agent_id, time_period)
        return result
        
    except AnalyticsNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content=get_error_response(e)
        )
    except Exception as e:
        error = handle_agent_error(e, agent_id=agent_id, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/agents/{agent_id}/performance", response_model=PerformanceMetrics)
async def get_agent_performance(
    agent_id: str = Path(..., description="Agent ID"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get agent performance metrics"""
    try:
        result = await analytics_service.get_agent_performance(agent_id)
        return result
        
    except AnalyticsNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content=get_error_response(e)
        )
    except Exception as e:
        error = handle_agent_error(e, agent_id=agent_id, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


# Workflow Analytics Endpoints
@router.get("/workflows", response_model=Dict[str, Any])
async def get_workflows_analytics(
    workflow_type: Optional[str] = Query(None, description="Filter by workflow type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    time_period: str = Query("30d", description="Time period for analytics"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get comprehensive workflows analytics"""
    try:
        result = await analytics_service.get_workflows_analytics(
            workflow_type, category, time_period, page, per_page
        )
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/workflows/{workflow_id}", response_model=WorkflowAnalytics)
async def get_workflow_analytics(
    workflow_id: str = Path(..., description="Workflow ID"),
    time_period: str = Query("30d", description="Time period for analytics"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get detailed analytics for specific workflow"""
    try:
        result = await analytics_service.get_workflow_analytics(workflow_id, time_period)
        return result
        
    except AnalyticsNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content=get_error_response(e)
        )
    except Exception as e:
        error = handle_agent_error(e, workflow_id=workflow_id, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


# Collaboration Analytics Endpoints
@router.get("/collaboration", response_model=CollaborationAnalytics)
async def get_collaboration_analytics(
    time_period: str = Query("30d", description="Time period for analytics"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get collaboration analytics"""
    try:
        result = await analytics_service.get_collaboration_analytics(time_period)
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/collaboration/teams", response_model=Dict[str, Any])
async def get_team_analytics(
    team_id: Optional[str] = Query(None, description="Filter by team ID"),
    time_period: str = Query("30d", description="Time period for analytics"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get team collaboration analytics"""
    try:
        result = await analytics_service.get_team_analytics(team_id, time_period)
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


# Insights and Recommendations
@router.get("/insights", response_model=InsightGeneration)
async def get_insights(
    insight_type: str = Query("all", description="Type of insights to generate"),
    time_period: str = Query("30d", description="Time period for insights"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get AI-powered insights and recommendations"""
    try:
        result = await analytics_service.get_insights(insight_type, time_period)
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.post("/insights/generate", response_model=InsightGeneration)
async def generate_custom_insights(
    insight_request: AnalyticsRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:create"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Generate custom insights based on specific criteria"""
    try:
        result = await analytics_service.generate_custom_insights(insight_request)
        
        # Background tasks
        background_tasks.add_task(
            log_insight_generation,
            current_user["user_id"],
            insight_request.insight_type
        )
        background_tasks.add_task(
            cache_insights,
            result.insight_id,
            result.insights
        )
        
        return result
        
    except AnalyticsValidationError as e:
        return JSONResponse(
            status_code=400,
            content=get_error_response(e)
        )
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


# Export and Reporting
@router.get("/export", response_model=Dict[str, Any])
async def export_analytics(
    export_type: str = Query("json", description="Export format (json, csv, excel)"),
    analytics_type: str = Query("system", description="Type of analytics to export"),
    time_period: str = Query("30d", description="Time period for export"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:export"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Export analytics data"""
    try:
        result = await analytics_service.export_analytics(analytics_type, export_type, time_period)
        return result
        
    except AnalyticsValidationError as e:
        return JSONResponse(
            status_code=400,
            content=get_error_response(e)
        )
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.post("/reports/generate", response_model=Dict[str, Any])
async def generate_report(
    report_request: AnalyticsRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:create"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Generate comprehensive analytics report"""
    try:
        result = await analytics_service.generate_report(report_request)
        
        # Background tasks
        background_tasks.add_task(
            log_report_generation,
            current_user["user_id"],
            report_request.report_type
        )
        background_tasks.add_task(
            store_report,
            result.report_id,
            result.report_data
        )
        background_tasks.add_task(
            notify_report_ready,
            current_user["user_id"],
            result.report_id
        )
        
        return result
        
    except AnalyticsValidationError as e:
        return JSONResponse(
            status_code=400,
            content=get_error_response(e)
        )
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/reports/{report_id}", response_model=Dict[str, Any])
async def get_report(
    report_id: str = Path(..., description="Report ID"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get generated report"""
    try:
        result = await analytics_service.get_report(report_id)
        return result
        
    except AnalyticsNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content=get_error_response(e)
        )
    except Exception as e:
        error = handle_agent_error(e, report_id=report_id, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/reports", response_model=Dict[str, Any])
async def list_reports(
    report_type: Optional[str] = Query(None, description="Filter by report type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """List generated reports"""
    try:
        result = await analytics_service.list_reports(report_type, status, page, per_page)
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


# Real-time Analytics
@router.get("/realtime", response_model=Dict[str, Any])
async def get_realtime_analytics(
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get real-time analytics data"""
    try:
        result = await analytics_service.get_realtime_analytics()
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/realtime/stream")
async def stream_realtime_analytics(
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Stream real-time analytics data"""
    try:
        async def generate_analytics_stream():
            while True:
                try:
                    data = await analytics_service.get_realtime_analytics()
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Error in analytics stream: {e}")
                    break
        
        return StreamingResponse(
            generate_analytics_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


# Benchmark and Comparison
@router.get("/benchmark", response_model=Dict[str, Any])
async def get_benchmark_analytics(
    benchmark_type: str = Query("industry", description="Type of benchmark"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get benchmark comparison analytics"""
    try:
        result = await analytics_service.get_benchmark_analytics(benchmark_type)
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


@router.get("/comparison", response_model=Dict[str, Any])
async def get_comparison_analytics(
    comparison_type: str = Query("period", description="Type of comparison"),
    time_period_1: str = Query("30d", description="First time period"),
    time_period_2: str = Query("60d", description="Second time period"),
    current_user: Dict[str, Any] = Depends(require_permissions(["analytics:read"])),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get comparative analytics"""
    try:
        result = await analytics_service.get_comparison_analytics(
            comparison_type, time_period_1, time_period_2
        )
        return result
        
    except Exception as e:
        error = handle_agent_error(e, user_id=current_user["user_id"])
        log_agent_error(error)
        return JSONResponse(
            status_code=500,
            content=get_error_response(error)
        )


# Health Check
@router.get("/health")
async def analytics_health_check():
    """Analytics service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "analytics-api",
        "version": "1.0.0"
    }


# Background Tasks
async def log_insight_generation(user_id: str, insight_type: str):
    """Log insight generation"""
    try:
        logger.info(f"Insights generated for user: {user_id}, type: {insight_type}")
    except Exception as e:
        logger.error(f"Failed to log insight generation: {e}")


async def log_report_generation(user_id: str, report_type: str):
    """Log report generation"""
    try:
        logger.info(f"Report generated for user: {user_id}, type: {report_type}")
    except Exception as e:
        logger.error(f"Failed to log report generation: {e}")


async def cache_insights(insight_id: str, insights: List[Dict[str, Any]]):
    """Cache insights"""
    try:
        logger.info(f"Caching insights: {insight_id}")
    except Exception as e:
        logger.error(f"Failed to cache insights: {e}")


async def store_report(report_id: str, report_data: Dict[str, Any]):
    """Store report"""
    try:
        logger.info(f"Storing report: {report_id}")
    except Exception as e:
        logger.error(f"Failed to store report: {e}")


async def notify_report_ready(user_id: str, report_id: str):
    """Notify report ready"""
    try:
        logger.info(f"Notifying user {user_id} that report {report_id} is ready")
    except Exception as e:
        logger.error(f"Failed to notify report ready: {e}")





























