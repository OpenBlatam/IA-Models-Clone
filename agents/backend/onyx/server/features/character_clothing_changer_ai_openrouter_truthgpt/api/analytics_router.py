"""
Analytics Router
================

FastAPI router for analytics endpoints.
"""

import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, Optional

from services.analytics_service import AnalyticsService
from services.optimization_service import OptimizationService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance"""
    return AnalyticsService()


def get_optimization_service() -> OptimizationService:
    """Get optimization service instance"""
    return OptimizationService()


@router.get(
    "/analytics/report",
    status_code=status.HTTP_200_OK,
    summary="Get Analytics Report",
    description="Generate comprehensive analytics report"
)
async def get_analytics_report(
    period: str = Query("7d", description="Period (e.g., 7d, 30d)"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> Dict[str, Any]:
    """
    Get analytics report.
    
    Args:
        period: Period string (e.g., "7d", "30d")
        analytics_service: AnalyticsService instance (injected)
        
    Returns:
        Analytics report
    """
    try:
        report = analytics_service.generate_report(period=period)
        return report
    except Exception as e:
        logger.error(f"Error generating analytics report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get(
    "/analytics/usage",
    status_code=status.HTTP_200_OK,
    summary="Get Usage Analytics",
    description="Get usage analytics for period"
)
async def get_usage_analytics(
    days: int = Query(7, description="Number of days"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> Dict[str, Any]:
    """
    Get usage analytics.
    
    Args:
        days: Number of days to analyze
        analytics_service: AnalyticsService instance (injected)
        
    Returns:
        Usage analytics
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        usage = analytics_service.analyze_usage(start_date, end_date)
        return usage
    except Exception as e:
        logger.error(f"Error getting usage analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage analytics: {str(e)}"
        )


@router.get(
    "/analytics/trends",
    status_code=status.HTTP_200_OK,
    summary="Get Trend Analysis",
    description="Get trend analysis"
)
async def get_trends(
    days: int = Query(7, description="Number of days"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> Dict[str, Any]:
    """
    Get trend analysis.
    
    Args:
        days: Number of days to analyze
        analytics_service: AnalyticsService instance (injected)
        
    Returns:
        Trend analysis
    """
    try:
        trends = analytics_service.analyze_trends(days=days)
        return trends
    except Exception as e:
        logger.error(f"Error getting trends: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trends: {str(e)}"
        )


@router.post(
    "/optimization/optimize",
    status_code=status.HTTP_200_OK,
    summary="Optimize Parameters",
    description="Optimize workflow parameters"
)
async def optimize_parameters(
    current_params: Dict[str, Any],
    target_quality: str = Query("balanced", description="Target quality (fast, balanced, high)"),
    optimization_service: OptimizationService = Depends(get_optimization_service)
) -> Dict[str, Any]:
    """
    Optimize workflow parameters.
    
    Args:
        current_params: Current workflow parameters
        target_quality: Target quality level
        optimization_service: OptimizationService instance (injected)
        
    Returns:
        Optimized parameters
    """
    try:
        optimized = optimization_service.optimize_workflow_parameters(
            current_params=current_params,
            target_quality=target_quality
        )
        return {
            "original": current_params,
            "optimized": optimized,
            "target_quality": target_quality
        }
    except Exception as e:
        logger.error(f"Error optimizing parameters: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize parameters: {str(e)}"
        )


@router.get(
    "/optimization/suggestions",
    status_code=status.HTTP_200_OK,
    summary="Get Optimization Suggestions",
    description="Get optimization suggestions based on metrics"
)
async def get_optimization_suggestions(
    metrics: Dict[str, Any],
    optimization_service: OptimizationService = Depends(get_optimization_service)
) -> Dict[str, Any]:
    """
    Get optimization suggestions.
    
    Args:
        metrics: Performance metrics
        optimization_service: OptimizationService instance (injected)
        
    Returns:
        Optimization suggestions
    """
    try:
        suggestions = optimization_service.suggest_improvements(metrics=metrics)
        return {
            "suggestions": suggestions,
            "count": len(suggestions)
        }
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get suggestions: {str(e)}"
        )


@router.get(
    "/analytics/reports",
    status_code=status.HTTP_200_OK,
    summary="Get Recent Reports",
    description="Get recent analytics reports"
)
async def get_recent_reports(
    limit: int = Query(5, description="Maximum number of reports"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> Dict[str, Any]:
    """
    Get recent analytics reports.
    
    Args:
        limit: Maximum number of reports
        analytics_service: AnalyticsService instance (injected)
        
    Returns:
        List of recent reports
    """
    try:
        reports = analytics_service.get_recent_reports(limit=limit)
        return {
            "reports": reports,
            "count": len(reports)
        }
    except Exception as e:
        logger.error(f"Error getting reports: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get reports: {str(e)}"
        )

