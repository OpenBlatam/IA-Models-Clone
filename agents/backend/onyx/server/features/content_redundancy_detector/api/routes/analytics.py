"""
Analytics Router - Analytics and reporting endpoints
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse

try:
    from analytics import (
        generate_performance_report,
        generate_content_insights_report,
        generate_similarity_insights_report,
        generate_quality_insights_report,
        get_analytics_report,
        get_all_analytics_reports
    )
except ImportError:
    logging.warning("analytics module not available")
    def generate_performance_report(): return type('obj', (object,), {'data': {}})()
    def generate_content_insights_report(): return type('obj', (object,), {'data': {}})()
    def generate_similarity_insights_report(): return type('obj', (object,), {'data': {}})()
    def generate_quality_insights_report(): return type('obj', (object,), {'data': {}})()
    def get_analytics_report(*args, **kwargs): return {}
    def get_all_analytics_reports(): return []

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_analytics() -> JSONResponse:
    """Get performance analytics report"""
    logger.info("Performance analytics requested")
    
    try:
        report = generate_performance_report()
        return JSONResponse(content={
            "success": True,
            "data": report.data if hasattr(report, 'data') else report,
            "error": None
        })
    except Exception as e:
        logger.error(f"Performance analytics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/content", response_model=Dict[str, Any])
async def get_content_analytics() -> JSONResponse:
    """Get content analytics report"""
    logger.info("Content analytics requested")
    
    try:
        report = generate_content_insights_report()
        return JSONResponse(content={
            "success": True,
            "data": report.data if hasattr(report, 'data') else report,
            "error": None
        })
    except Exception as e:
        logger.error(f"Content analytics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/similarity", response_model=Dict[str, Any])
async def get_similarity_analytics() -> JSONResponse:
    """Get similarity analytics report"""
    logger.info("Similarity analytics requested")
    
    try:
        report = generate_similarity_insights_report()
        return JSONResponse(content={
            "success": True,
            "data": report.data if hasattr(report, 'data') else report,
            "error": None
        })
    except Exception as e:
        logger.error(f"Similarity analytics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/quality", response_model=Dict[str, Any])
async def get_quality_analytics() -> JSONResponse:
    """Get quality analytics report"""
    logger.info("Quality analytics requested")
    
    try:
        report = generate_quality_insights_report()
        return JSONResponse(content={
            "success": True,
            "data": report.data if hasattr(report, 'data') else report,
            "error": None
        })
    except Exception as e:
        logger.error(f"Quality analytics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reports", response_model=Dict[str, Any])
async def get_all_analytics_reports_endpoint() -> JSONResponse:
    """Get all analytics reports"""
    logger.info("All analytics reports requested")
    
    try:
        reports = get_all_analytics_reports()
        return JSONResponse(content={
            "success": True,
            "data": {
                "reports": [
                    {
                        "report_id": report.id if hasattr(report, 'id') else str(i),
                        "report_type": report.report_type if hasattr(report, 'report_type') else "unknown",
                        "generated_at": report.generated_at if hasattr(report, 'generated_at') else None,
                        "period_start": report.period_start if hasattr(report, 'period_start') else None,
                        "period_end": report.period_end if hasattr(report, 'period_end') else None
                    }
                    for i, report in enumerate(reports)
                ]
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get analytics reports error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/query", response_model=Dict[str, Any])
async def query_analytics(query_params: Dict[str, Any]) -> JSONResponse:
    """Query analytics with custom parameters"""
    logger.info("Analytics query requested")
    
    try:
        from advanced_analytics_dashboard import analytics_dashboard, AnalyticsQuery
        
        query = AnalyticsQuery(**query_params)
        result = await analytics_dashboard.query(query)
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except Exception as e:
        logger.error(f"Analytics query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports", response_model=Dict[str, Any])
async def generate_analytics_report(report_config: Dict[str, Any]) -> JSONResponse:
    """Generate a new analytics report"""
    logger.info("Generate analytics report requested")
    
    try:
        from advanced_analytics_dashboard import analytics_dashboard, ReportConfig
        
        config = ReportConfig(**report_config)
        report = await analytics_dashboard.generate_report(config)
        
        return JSONResponse(content={
            "success": True,
            "data": report,
            "error": None
        })
    except Exception as e:
        logger.error(f"Generate analytics report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboards", response_model=Dict[str, Any])
async def create_dashboard(dashboard_data: Dict[str, Any]) -> JSONResponse:
    """Create analytics dashboard"""
    logger.info(f"Creating dashboard: {dashboard_data.get('name', 'unknown')}")
    
    try:
        from advanced_analytics_dashboard import analytics_dashboard
        
        name = dashboard_data.get("name")
        description = dashboard_data.get("description", "")
        created_by = dashboard_data.get("created_by", "system")
        is_public = dashboard_data.get("is_public", False)
        
        if not name:
            raise ValueError("Dashboard name is required")
        
        dashboard_id = await analytics_dashboard.create_dashboard(name, description, created_by, is_public)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "dashboard_id": dashboard_id,
                "name": name,
                "status": "created",
                "timestamp": time.time()
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dashboards", response_model=Dict[str, Any])
async def list_dashboards() -> JSONResponse:
    """List all dashboards"""
    logger.info("Listing dashboards")
    
    try:
        from advanced_analytics_dashboard import analytics_dashboard
        
        dashboards = await analytics_dashboard.list_dashboards()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "dashboards": [
                    dashboard.model_dump() if hasattr(dashboard, 'model_dump') else dashboard
                    for dashboard in dashboards
                ],
                "count": len(dashboards),
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error listing dashboards: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dashboards/{dashboard_id}", response_model=Dict[str, Any])
async def get_dashboard(dashboard_id: str = Path(...)) -> JSONResponse:
    """Get dashboard by ID"""
    logger.info(f"Getting dashboard: {dashboard_id}")
    
    try:
        from advanced_analytics_dashboard import analytics_dashboard
        
        dashboard = await analytics_dashboard.get_dashboard(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return JSONResponse(content={
            "success": True,
            "data": dashboard.model_dump() if hasattr(dashboard, 'model_dump') else dashboard,
            "error": None
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dashboards/{dashboard_id}/html", response_model=Dict[str, Any])
async def get_dashboard_html(dashboard_id: str = Path(...)) -> JSONResponse:
    """Get dashboard as HTML"""
    logger.info(f"Getting dashboard HTML: {dashboard_id}")
    
    try:
        from advanced_analytics_dashboard import analytics_dashboard
        
        html = await analytics_dashboard.generate_dashboard_html(dashboard_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "dashboard_id": dashboard_id,
                "html": html,
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error generating dashboard HTML: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reports/{report_id}/generate", response_model=Dict[str, Any])
async def generate_report(report_id: str = Path(...)) -> JSONResponse:
    """Generate report"""
    logger.info(f"Generating report: {report_id}")
    
    try:
        import base64
        from advanced_analytics_dashboard import analytics_dashboard
        
        report_data = await analytics_dashboard.generate_report(report_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "report_id": report_id,
                "data": base64.b64encode(report_data).decode() if isinstance(report_data, bytes) else report_data,
                "size": len(report_data) if isinstance(report_data, bytes) else None,
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

