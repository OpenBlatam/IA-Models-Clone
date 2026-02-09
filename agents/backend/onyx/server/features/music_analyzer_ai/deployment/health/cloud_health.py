"""
Health check endpoints optimized for cloud deployments
Supports AWS Lambda, Azure Functions, and containerized deployments
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint
    Optimized for cloud load balancers and health probes
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "music-analyzer-ai"
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint
    Verifies that the service is ready to accept traffic
    """
    try:
        # Check critical dependencies
        checks = {
            "spotify_configured": bool(os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET")),
            "environment": os.getenv("ENVIRONMENT", "unknown")
        }
        
        # Check if all critical services are available
        all_ready = all(checks.values())
        
        if not all_ready:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "checks": checks
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check endpoint
    Verifies that the service is alive and responsive
    """
    return {
        "status": "alive",
        "timestamp": time.time()
    }


@router.get("/detailed")
async def detailed_health() -> Dict[str, Any]:
    """
    Detailed health check with system information
    Useful for monitoring and debugging
    """
    import platform
    import sys
    
    # Detect deployment platform
    platform_info = {
        "aws_lambda": bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME")),
        "azure_functions": bool(os.getenv("FUNCTIONS_WORKER_RUNTIME")),
        "container": bool(os.getenv("CONTAINER_NAME")),
        "local": not any([
            os.getenv("AWS_LAMBDA_FUNCTION_NAME"),
            os.getenv("FUNCTIONS_WORKER_RUNTIME"),
            os.getenv("CONTAINER_NAME")
        ])
    }
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "music-analyzer-ai",
        "version": "2.21.0",
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "platform": {k: v for k, v in platform_info.items() if v},
        "python_version": sys.version,
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor()
        },
        "configuration": {
            "cache_enabled": os.getenv("CACHE_ENABLED", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        }
    }




