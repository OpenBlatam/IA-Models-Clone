"""
Admin API endpoints for system management and monitoring
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ....services.content_moderation_service import ContentModerationService
from ....services.performance_service import PerformanceService
from ....services.analytics_service import AnalyticsService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError
from ....core.security import require_admin_role

router = APIRouter()


class ModerationRequest(BaseModel):
    """Request model for content moderation."""
    content: str = Field(..., min_length=1, description="Content to moderate")
    content_type: str = Field(default="post", description="Type of content")
    user_id: Optional[str] = Field(None, description="User ID for context")


class CacheClearRequest(BaseModel):
    """Request model for cache clearing."""
    pattern: str = Field(default="*", description="Cache key pattern to clear")


class DatabaseOptimizationRequest(BaseModel):
    """Request model for database optimization."""
    analyze_tables: bool = Field(default=True, description="Analyze tables")
    vacuum_tables: bool = Field(default=True, description="Vacuum tables")
    update_statistics: bool = Field(default=True, description="Update statistics")
    cleanup_old_data: bool = Field(default=True, description="Clean up old data")


async def get_moderation_service(session: DatabaseSessionDep) -> ContentModerationService:
    """Get content moderation service instance."""
    return ContentModerationService(session)


async def get_performance_service(session: DatabaseSessionDep) -> PerformanceService:
    """Get performance service instance."""
    return PerformanceService(session)


async def get_analytics_service(session: DatabaseSessionDep) -> AnalyticsService:
    """Get analytics service instance."""
    return AnalyticsService(session)


@router.get("/system/metrics", response_model=Dict[str, Any])
async def get_system_metrics(
    performance_service: PerformanceService = Depends(get_performance_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get comprehensive system performance metrics."""
    try:
        metrics = await performance_service.get_system_metrics()
        
        return {
            "success": True,
            "data": metrics,
            "message": "System metrics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system metrics"
        )


@router.get("/database/performance", response_model=Dict[str, Any])
async def get_database_performance(
    performance_service: PerformanceService = Depends(get_performance_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get database performance metrics."""
    try:
        performance = await performance_service.get_database_performance()
        
        return {
            "success": True,
            "data": performance,
            "message": "Database performance metrics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get database performance"
        )


@router.get("/cache/performance", response_model=Dict[str, Any])
async def get_cache_performance(
    performance_service: PerformanceService = Depends(get_performance_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get cache performance metrics."""
    try:
        performance = await performance_service.get_cache_performance()
        
        return {
            "success": True,
            "data": performance,
            "message": "Cache performance metrics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache performance"
        )


@router.post("/database/optimize", response_model=Dict[str, Any])
async def optimize_database(
    request: DatabaseOptimizationRequest = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    performance_service: PerformanceService = Depends(get_performance_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Optimize database performance."""
    try:
        # Run optimization in background
        background_tasks.add_task(performance_service.optimize_database)
        
        return {
            "success": True,
            "message": "Database optimization started in background"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start database optimization"
        )


@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_cache(
    request: CacheClearRequest = Depends(),
    performance_service: PerformanceService = Depends(get_performance_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Clear cache entries."""
    try:
        result = await performance_service.clear_cache(request.pattern)
        
        return {
            "success": True,
            "data": result,
            "message": "Cache cleared successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.get("/slow-queries", response_model=Dict[str, Any])
async def get_slow_queries(
    limit: int = Query(default=10, ge=1, le=100, description="Number of slow queries to return"),
    performance_service: PerformanceService = Depends(get_performance_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get slow query information."""
    try:
        slow_queries = await performance_service.get_slow_queries(limit)
        
        return {
            "success": True,
            "data": {
                "slow_queries": slow_queries,
                "total": len(slow_queries)
            },
            "message": "Slow queries retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get slow queries"
        )


@router.post("/moderate-content", response_model=Dict[str, Any])
async def moderate_content(
    request: ModerationRequest = Depends(),
    moderation_service: ContentModerationService = Depends(get_moderation_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Moderate content for safety and quality."""
    try:
        moderation_result = await moderation_service.moderate_content(
            content=request.content,
            content_type=request.content_type,
            user_id=request.user_id
        )
        
        return {
            "success": True,
            "data": moderation_result,
            "message": "Content moderation completed"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to moderate content"
        )


@router.get("/moderation/stats", response_model=Dict[str, Any])
async def get_moderation_stats(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    moderation_service: ContentModerationService = Depends(get_moderation_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get content moderation statistics."""
    try:
        stats = await moderation_service.get_moderation_stats(days)
        
        return {
            "success": True,
            "data": stats,
            "message": "Moderation statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get moderation statistics"
        )


@router.get("/analytics/overview", response_model=Dict[str, Any])
async def get_analytics_overview(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get analytics overview."""
    try:
        overview = await analytics_service.get_analytics_overview(days)
        
        return {
            "success": True,
            "data": overview,
            "message": "Analytics overview retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics overview"
        )


@router.get("/analytics/content", response_model=Dict[str, Any])
async def get_content_analytics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get content analytics."""
    try:
        analytics = await analytics_service.get_content_analytics(days)
        
        return {
            "success": True,
            "data": analytics,
            "message": "Content analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get content analytics"
        )


@router.get("/analytics/users", response_model=Dict[str, Any])
async def get_user_analytics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get user analytics."""
    try:
        analytics = await analytics_service.get_user_analytics(days)
        
        return {
            "success": True,
            "data": analytics,
            "message": "User analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user analytics"
        )


@router.get("/analytics/engagement", response_model=Dict[str, Any])
async def get_engagement_analytics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get engagement analytics."""
    try:
        analytics = await analytics_service.get_engagement_analytics(days)
        
        return {
            "success": True,
            "data": analytics,
            "message": "Engagement analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get engagement analytics"
        )


@router.get("/system/health", response_model=Dict[str, Any])
async def get_system_health(
    performance_service: PerformanceService = Depends(get_performance_service),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get comprehensive system health status."""
    try:
        # Get system metrics
        system_metrics = await performance_service.get_system_metrics()
        
        # Get database performance
        db_performance = await performance_service.get_database_performance()
        
        # Get cache performance
        cache_performance = await performance_service.get_cache_performance()
        
        # Determine overall health
        health_status = "healthy"
        issues = []
        
        # Check CPU usage
        cpu_usage = system_metrics.get("system", {}).get("cpu", {}).get("usage_percent", 0)
        if cpu_usage > 80:
            health_status = "warning"
            issues.append(f"High CPU usage: {cpu_usage}%")
        
        # Check memory usage
        memory_usage = system_metrics.get("system", {}).get("memory", {}).get("usage_percent", 0)
        if memory_usage > 85:
            health_status = "warning"
            issues.append(f"High memory usage: {memory_usage}%")
        
        # Check disk usage
        disk_usage = system_metrics.get("system", {}).get("disk", {}).get("usage_percent", 0)
        if disk_usage > 90:
            health_status = "critical"
            issues.append(f"High disk usage: {disk_usage}%")
        
        # Check database connections
        db_connections = db_performance.get("connection_pool", {}).get("checked_out", 0)
        if db_connections > 8:  # Assuming max 10 connections
            health_status = "warning"
            issues.append(f"High database connections: {db_connections}")
        
        return {
            "success": True,
            "data": {
                "status": health_status,
                "issues": issues,
                "system_metrics": system_metrics,
                "database_performance": db_performance,
                "cache_performance": cache_performance,
                "timestamp": system_metrics.get("timestamp")
            },
            "message": "System health status retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system health"
        )


@router.get("/logs", response_model=Dict[str, Any])
async def get_system_logs(
    level: str = Query(default="INFO", description="Log level filter"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of log entries"),
    current_user: CurrentUserDep = Depends(require_admin_role)
):
    """Get system logs."""
    try:
        # In a real implementation, you would read from log files or a log management system
        # For now, we'll return a mock structure
        logs = [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "level": "INFO",
                "message": "System started successfully",
                "source": "main.py"
            },
            {
                "timestamp": "2024-01-15T10:01:00Z",
                "level": "INFO",
                "message": "Database connection established",
                "source": "database.py"
            }
        ]
        
        return {
            "success": True,
            "data": {
                "logs": logs,
                "total": len(logs),
                "level": level
            },
            "message": "System logs retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system logs"
        )






























