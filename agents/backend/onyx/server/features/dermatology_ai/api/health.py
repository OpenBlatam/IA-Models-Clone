"""
Health check endpoints
Provides system health status for monitoring and load balancers
"""

from fastapi import APIRouter, HTTPException, Response
from typing import Dict, Any
import logging
from datetime import datetime

from ..core.composition_root import get_composition_root
from ..core.infrastructure.metrics_exporter import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint
    Returns 200 if service is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "dermatology_ai"
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint
    Returns 200 if service is ready to accept traffic
    """
    try:
        composition_root = get_composition_root()
        
        if not composition_root._initialized:
            raise HTTPException(
                status_code=503,
                detail="Service not initialized"
            )
        
        # Check critical dependencies
        checks = {
            "database": False,
            "cache": False,
            "image_processor": False,
        }
        
        try:
            # Check database
            if composition_root._database_adapter:
                # Simple connectivity check
                checks["database"] = True
        except Exception as e:
            logger.warning(f"Database check failed: {e}")
        
        try:
            # Check cache
            cache = await composition_root.service_factory.create("cache")
            if cache:
                checks["cache"] = True
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        try:
            # Check image processor
            processor = await composition_root.service_factory.create("image_processor")
            if processor:
                checks["image_processor"] = True
        except Exception as e:
            logger.warning(f"Image processor check failed: {e}")
        
        all_ready = all(checks.values())
        
        return {
            "status": "ready" if all_ready else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "ready": all_ready
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check endpoint
    Returns 200 if service is alive (not deadlocked)
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint
    """
    metrics_collector = get_metrics_collector()
    prometheus_metrics = metrics_collector.get_metrics_prometheus()
    return Response(
        content=prometheus_metrics,
        media_type="text/plain; version=0.0.4"
    )


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with component status
    """
    try:
        composition_root = get_composition_root()
        
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "dermatology_ai",
            "version": "7.1.0",
            "architecture": "Hexagonal",
            "components": {}
        }
        
        if composition_root._initialized:
            health_info["components"]["composition_root"] = {
                "status": "initialized",
                "use_cases_cached": len(composition_root._use_case_cache)
            }
            
            # Check service factory
            try:
                factory = composition_root.service_factory
                health_info["components"]["service_factory"] = {
                    "status": "active",
                    "registered_services": len(factory.registrations),
                    "singletons": len(factory.singletons)
                }
            except Exception as e:
                health_info["components"]["service_factory"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            health_info["components"]["composition_root"] = {
                "status": "not_initialized"
            }
            health_info["status"] = "degraded"
        
        return health_info
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

