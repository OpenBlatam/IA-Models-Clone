from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
import logging
import asyncio
from datetime import datetime
from ..routes.base import get_request_context, log_route_access
from ..schemas.base import BaseResponse, ErrorResponse
from ..async_database_api_operations import AsyncDatabaseManager
from ..caching_manager import CachingManager
from ..performance_metrics import PerformanceMonitor
from ..error_handling_middleware import ErrorMonitor
        import psutil
from typing import Any, List, Dict, Optional
"""
Health Router

This module contains routes for health checks, status monitoring,
and system diagnostics. Provides comprehensive system health information.
"""


# Import dependencies

# Import schemas

# Import services

# Initialize router
router = APIRouter(prefix="/health", tags=["health"])

# Logger
logger = logging.getLogger(__name__)

# Route dependencies
async def get_db_manager(
    context: Dict[str, Any] = Depends(get_request_context)
) -> AsyncDatabaseManager:
    """Get database manager from context."""
    return context["async_io_manager"].db_manager

async def get_cache_manager(
    context: Dict[str, Any] = Depends(get_request_context)
) -> CachingManager:
    """Get cache manager from context."""
    return context["cache_manager"]

async def get_performance_monitor(
    context: Dict[str, Any] = Depends(get_request_context)
) -> PerformanceMonitor:
    """Get performance monitor from context."""
    return context["performance_monitor"]

async def get_error_monitor(
    context: Dict[str, Any] = Depends(get_request_context)
) -> ErrorMonitor:
    """Get error monitor from context."""
    return context["error_monitor"]

# Health Check Routes
@router.get("/", response_model=BaseResponse)
async def health_check():
    """Basic health check endpoint."""
    return BaseResponse(
        status="success",
        message="Service is healthy",
        data={
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Product Descriptions API",
            "version": "1.0.0"
        }
    )

@router.get("/detailed", response_model=BaseResponse)
async def detailed_health_check(
    context: Dict[str, Any] = Depends(get_request_context)
):
    """Detailed health check with all system components."""
    try:
        log_route_access("detailed_health_check")
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Product Descriptions API",
            "version": "1.0.0",
            "components": {}
        }
        
        # Check database health
        try:
            db_manager = await get_db_manager(context)
            db_health = await db_manager.health_check()
            health_status["components"]["database"] = {
                "status": "healthy" if db_health else "unhealthy",
                "details": db_health
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check cache health
        try:
            cache_manager = await get_cache_manager(context)
            cache_health = await cache_manager.health_check()
            health_status["components"]["cache"] = {
                "status": "healthy" if cache_health else "unhealthy",
                "details": cache_health
            }
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            health_status["components"]["cache"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check performance monitor health
        try:
            perf_monitor = await get_performance_monitor(context)
            perf_health = await perf_monitor.health_check()
            health_status["components"]["performance_monitor"] = {
                "status": "healthy" if perf_health else "unhealthy",
                "details": perf_health
            }
        except Exception as e:
            logger.error(f"Performance monitor health check failed: {e}")
            health_status["components"]["performance_monitor"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check error monitor health
        try:
            error_monitor = await get_error_monitor(context)
            error_health = await error_monitor.health_check()
            health_status["components"]["error_monitor"] = {
                "status": "healthy" if error_health else "unhealthy",
                "details": error_health
            }
        except Exception as e:
            logger.error(f"Error monitor health check failed: {e}")
            health_status["components"]["error_monitor"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall health
        all_healthy = all(
            comp["status"] == "healthy" 
            for comp in health_status["components"].values()
        )
        
        overall_status = "healthy" if all_healthy else "degraded"
        
        return BaseResponse(
            status="success",
            message=f"Detailed health check completed - {overall_status}",
            data=health_status
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )

@router.get("/readiness", response_model=BaseResponse)
async def readiness_check(
    context: Dict[str, Any] = Depends(get_request_context)
):
    """Readiness check for Kubernetes deployments."""
    try:
        log_route_access("readiness_check")
        
        # Check critical dependencies
        db_manager = await get_db_manager(context)
        cache_manager = await get_cache_manager(context)
        
        # Test database connectivity
        db_ready = await db_manager.is_ready()
        
        # Test cache connectivity
        cache_ready = await cache_manager.is_ready()
        
        if db_ready and cache_ready:
            return BaseResponse(
                status="success",
                message="Service is ready",
                data={
                    "ready": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

@router.get("/liveness", response_model=BaseResponse)
async def liveness_check():
    """Liveness check for Kubernetes deployments."""
    return BaseResponse(
        status="success",
        message="Service is alive",
        data={
            "alive": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Component Health Checks
@router.get("/database", response_model=BaseResponse)
async def database_health_check(
    context: Dict[str, Any] = Depends(get_request_context)
):
    """Database-specific health check."""
    try:
        log_route_access("database_health_check")
        
        db_manager = await get_db_manager(context)
        health = await db_manager.health_check()
        
        return BaseResponse(
            status="success",
            message="Database health check completed",
            data=health
        )
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database health check failed"
        )

@router.get("/cache", response_model=BaseResponse)
async def cache_health_check(
    context: Dict[str, Any] = Depends(get_request_context)
):
    """Cache-specific health check."""
    try:
        log_route_access("cache_health_check")
        
        cache_manager = await get_cache_manager(context)
        health = await cache_manager.health_check()
        
        return BaseResponse(
            status="success",
            message="Cache health check completed",
            data=health
        )
        
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache health check failed"
        )

@router.get("/performance", response_model=BaseResponse)
async def performance_health_check(
    context: Dict[str, Any] = Depends(get_request_context)
):
    """Performance monitor health check."""
    try:
        log_route_access("performance_health_check")
        
        perf_monitor = await get_performance_monitor(context)
        health = await perf_monitor.health_check()
        
        return BaseResponse(
            status="success",
            message="Performance monitor health check completed",
            data=health
        )
        
    except Exception as e:
        logger.error(f"Performance monitor health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Performance monitor health check failed"
        )

@router.get("/errors", response_model=BaseResponse)
async def error_monitor_health_check(
    context: Dict[str, Any] = Depends(get_request_context)
):
    """Error monitor health check."""
    try:
        log_route_access("error_monitor_health_check")
        
        error_monitor = await get_error_monitor(context)
        health = await error_monitor.health_check()
        
        return BaseResponse(
            status="success",
            message="Error monitor health check completed",
            data=health
        )
        
    except Exception as e:
        logger.error(f"Error monitor health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Error monitor health check failed"
        )

# System Diagnostics
@router.get("/diagnostics", response_model=BaseResponse)
async def system_diagnostics(
    context: Dict[str, Any] = Depends(get_request_context)
):
    """Comprehensive system diagnostics."""
    try:
        log_route_access("system_diagnostics")
        
        diagnostics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": {},
            "component_status": {},
            "performance_metrics": {},
            "error_summary": {}
        }
        
        # Get system information
        diagnostics["system_info"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        # Get component status
        db_manager = await get_db_manager(context)
        cache_manager = await get_cache_manager(context)
        perf_monitor = await get_performance_monitor(context)
        error_monitor = await get_error_monitor(context)
        
        diagnostics["component_status"] = {
            "database": await db_manager.get_status(),
            "cache": await cache_manager.get_status(),
            "performance_monitor": await perf_monitor.get_status(),
            "error_monitor": await error_monitor.get_status()
        }
        
        # Get performance metrics
        diagnostics["performance_metrics"] = await perf_monitor.get_current_metrics()
        
        # Get error summary
        diagnostics["error_summary"] = await error_monitor.get_error_summary()
        
        return BaseResponse(
            status="success",
            message="System diagnostics completed",
            data=diagnostics
        )
        
    except Exception as e:
        logger.error(f"System diagnostics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System diagnostics failed"
        )

# Health Status Summary
@router.get("/summary", response_model=BaseResponse)
async def health_summary(
    context: Dict[str, Any] = Depends(get_request_context)
):
    """Get a summary of all health checks."""
    try:
        log_route_access("health_summary")
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "alerts": []
        }
        
        # Check each component
        components = ["database", "cache", "performance_monitor", "error_monitor"]
        
        for component in components:
            try:
                if component == "database":
                    db_manager = await get_db_manager(context)
                    status = await db_manager.health_check()
                elif component == "cache":
                    cache_manager = await get_cache_manager(context)
                    status = await cache_manager.health_check()
                elif component == "performance_monitor":
                    perf_monitor = await get_performance_monitor(context)
                    status = await perf_monitor.health_check()
                elif component == "error_monitor":
                    error_monitor = await get_error_monitor(context)
                    status = await error_monitor.health_check()
                
                summary["components"][component] = {
                    "status": "healthy" if status else "unhealthy",
                    "last_check": datetime.utcnow().isoformat()
                }
                
                if not status:
                    summary["alerts"].append(f"{component} is unhealthy")
                    
            except Exception as e:
                summary["components"][component] = {
                    "status": "error",
                    "error": str(e),
                    "last_check": datetime.utcnow().isoformat()
                }
                summary["alerts"].append(f"{component} check failed: {str(e)}")
        
        # Determine overall status
        if any(comp["status"] == "error" for comp in summary["components"].values()):
            summary["overall_status"] = "critical"
        elif any(comp["status"] == "unhealthy" for comp in summary["components"].values()):
            summary["overall_status"] = "degraded"
        
        return BaseResponse(
            status="success",
            message=f"Health summary: {summary['overall_status']}",
            data=summary
        )
        
    except Exception as e:
        logger.error(f"Health summary failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health summary failed"
        ) 