"""
Monitoring Routes
Real-time monitoring and metrics endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import asyncio
from monitoring_system import monitoring_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/monitoring", tags=["Real-time Monitoring"])

@router.get("/system-metrics")
async def get_system_metrics():
    """Get real-time system metrics"""
    try:
        metrics = await monitoring_system.collect_system_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai-metrics")
async def get_ai_metrics():
    """Get AI processing metrics"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        
        metrics = await monitoring_system.collect_ai_metrics(
            real_working_processor, advanced_real_processor
        )
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting AI metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload-metrics")
async def get_upload_metrics():
    """Get document upload metrics"""
    try:
        from document_upload_processor import document_upload_processor
        
        metrics = await monitoring_system.collect_upload_metrics(document_upload_processor)
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting upload metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-metrics")
async def get_performance_metrics():
    """Get performance metrics"""
    try:
        metrics = await monitoring_system.collect_performance_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/comprehensive-metrics")
async def get_comprehensive_metrics():
    """Get comprehensive monitoring metrics"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        
        metrics = await monitoring_system.get_comprehensive_metrics(
            real_working_processor, advanced_real_processor, document_upload_processor
        )
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting comprehensive metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts():
    """Get current alerts"""
    try:
        alerts = await monitoring_system.check_alerts()
        return JSONResponse(content={
            "alerts": alerts,
            "alert_count": len(alerts),
            "timestamp": monitoring_system._calculate_overall_health()
        })
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-status")
async def get_health_status():
    """Get overall health status"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        
        # Get comprehensive metrics
        metrics = await monitoring_system.get_comprehensive_metrics(
            real_working_processor, advanced_real_processor, document_upload_processor
        )
        
        health_status = {
            "overall_health": metrics.get("overall_health", "unknown"),
            "system_health": "healthy" if metrics.get("system_metrics", {}).get("cpu", {}).get("usage_percent", 0) < 80 else "warning",
            "ai_health": "healthy" if metrics.get("ai_metrics", {}).get("combined", {}).get("overall_success_rate", 0) > 90 else "warning",
            "upload_health": "healthy" if metrics.get("upload_metrics", {}).get("success_rate", 0) > 90 else "warning",
            "performance_score": metrics.get("performance_metrics", {}).get("performance_score", 0),
            "alert_count": metrics.get("alert_count", 0),
            "timestamp": metrics.get("timestamp")
        }
        
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics-summary")
async def get_metrics_summary():
    """Get metrics summary"""
    try:
        summary = monitoring_system.get_metrics_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_dashboard():
    """Get monitoring dashboard data"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        
        # Get all metrics
        system_metrics = await monitoring_system.collect_system_metrics()
        ai_metrics = await monitoring_system.collect_ai_metrics(
            real_working_processor, advanced_real_processor
        )
        upload_metrics = await monitoring_system.collect_upload_metrics(document_upload_processor)
        performance_metrics = await monitoring_system.collect_performance_metrics()
        alerts = await monitoring_system.check_alerts()
        
        dashboard_data = {
            "timestamp": monitoring_system._calculate_overall_health(),
            "overview": {
                "overall_health": monitoring_system._calculate_overall_health(),
                "uptime": performance_metrics.get("uptime_hours", 0),
                "performance_score": performance_metrics.get("performance_score", 0),
                "alert_count": len(alerts)
            },
            "system": {
                "cpu_usage": system_metrics.get("cpu", {}).get("usage_percent", 0),
                "memory_usage": system_metrics.get("memory", {}).get("usage_percent", 0),
                "disk_usage": system_metrics.get("disk", {}).get("usage_percent", 0)
            },
            "ai_processing": {
                "total_requests": ai_metrics.get("combined", {}).get("total_requests", 0),
                "success_rate": ai_metrics.get("combined", {}).get("overall_success_rate", 0),
                "average_processing_time": (ai_metrics.get("basic_processor", {}).get("average_processing_time", 0) + 
                                          ai_metrics.get("advanced_processor", {}).get("average_processing_time", 0)) / 2
            },
            "document_upload": {
                "total_uploads": upload_metrics.get("total_uploads", 0),
                "success_rate": upload_metrics.get("success_rate", 0),
                "supported_formats": upload_metrics.get("supported_formats", {})
            },
            "alerts": alerts[-5:],  # Last 5 alerts
            "charts": {
                "cpu_usage": system_metrics.get("cpu", {}).get("usage_percent", 0),
                "memory_usage": system_metrics.get("memory", {}).get("usage_percent", 0),
                "disk_usage": system_metrics.get("disk", {}).get("usage_percent", 0),
                "ai_success_rate": ai_metrics.get("combined", {}).get("overall_success_rate", 0),
                "upload_success_rate": upload_metrics.get("success_rate", 0)
            }
        }
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-monitoring")
async def health_check_monitoring():
    """Monitoring system health check"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        
        # Get basic metrics
        system_metrics = await monitoring_system.collect_system_metrics()
        ai_metrics = await monitoring_system.collect_ai_metrics(
            real_working_processor, advanced_real_processor
        )
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Real-time Monitoring System",
            "version": "1.0.0",
            "features": {
                "system_monitoring": True,
                "ai_monitoring": True,
                "upload_monitoring": True,
                "performance_monitoring": True,
                "alert_system": True,
                "dashboard": True
            },
            "system_status": {
                "cpu_usage": system_metrics.get("cpu", {}).get("usage_percent", 0),
                "memory_usage": system_metrics.get("memory", {}).get("usage_percent", 0),
                "disk_usage": system_metrics.get("disk", {}).get("usage_percent", 0)
            },
            "ai_status": {
                "total_requests": ai_metrics.get("combined", {}).get("total_requests", 0),
                "success_rate": ai_metrics.get("combined", {}).get("overall_success_rate", 0)
            }
        })
    except Exception as e:
        logger.error(f"Error in monitoring health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













