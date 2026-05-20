from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import time
from typing import Dict, Any, Optional
from ..core.roro import (
from ..core.database import get_session
from ..utils.helpers import get_system_version, get_uptime
import asyncio
        import os
        import psutil
        from ..core.database import get_session
        import os
        import psutil
        import psutil
        import platform
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Health routes using RORO pattern
Provides health check and system status endpoints.
"""


    HealthCheckRequest,
    HealthCheckResponse,
    create_success_response,
    create_error_response,
    validate_roro_request
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.post("/check", response_model=HealthCheckResponse)
async def health_check_roro(
    request_data: Dict[str, Any],
    session: AsyncSession = Depends(get_session)
) -> HealthCheckResponse:
    """Health check using RORO pattern"""
    
    # Validate RORO request
    is_valid, request, validation_errors = validate_roro_request(
        request_data, HealthCheckRequest
    )
    
    if not is_valid:
        return create_error_response(
            request_data,
            "Invalid request data",
            "VALIDATION_ERROR",
            "ValidationError",
            validation_errors
        )
    
    try:
        # Get system information
        system_version = get_system_version()
        uptime_info = get_uptime()
        
        # Check core components
        components = await check_core_components(session, request.include_details)
        
        # Check external services if requested
        external_services = None
        if request.check_external_services:
            external_services = await check_external_services()
        
        # Determine overall status
        overall_status = determine_overall_status(components, external_services)
        
        # Create response data
        response_data = {
            "status": overall_status,
            "version": system_version,
            "uptime": uptime_info,
            "components": components,
            "external_services": external_services
        }
        
        return create_success_response(
            request,
            "Health check completed successfully",
            response_data
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return create_error_response(
            request,
            "Health check failed",
            "HEALTH_CHECK_ERROR",
            "HealthCheckError",
            details={"error": str(e)}
        )


@router.get("/ready")
async def readiness_probe() -> Dict[str, Any]:
    """Readiness probe for Kubernetes"""
    
    try:
        # Check if system is ready to receive traffic
        is_ready = await check_readiness()
        
        if is_ready:
            return {
                "status": "ready",
                "timestamp": time.time(),
                "message": "System is ready to receive traffic"
            }
        else:
            return {
                "status": "not_ready",
                "timestamp": time.time(),
                "message": "System is not ready to receive traffic"
            }
            
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "message": "Readiness probe failed",
            "error": str(e)
        }


@router.get("/live")
async def liveness_probe() -> Dict[str, Any]:
    """Liveness probe for Kubernetes"""
    
    try:
        # Check if system is alive
        is_alive = await check_liveness()
        
        if is_alive:
            return {
                "status": "alive",
                "timestamp": time.time(),
                "message": "System is alive"
            }
        else:
            return {
                "status": "dead",
                "timestamp": time.time(),
                "message": "System is not responding"
            }
            
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "message": "Liveness probe failed",
            "error": str(e)
        }


@router.get("/status")
async def system_status() -> Dict[str, Any]:
    """Get detailed system status"""
    
    try:
        # Get comprehensive system status
        status_info = await get_system_status()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "data": status_info
        }
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "message": "System status check failed",
            "error": str(e)
        }


# Helper functions for RORO pattern
async def check_core_components(
    session: AsyncSession,
    include_details: bool = False
) -> Dict[str, bool]:
    """Check core system components"""
    
    components = {
        "database": False,
        "file_system": False,
        "memory": False,
        "cpu": False
    }
    
    try:
        # Check database connectivity
        await session.execute("SELECT 1")
        components["database"] = True
        
        # Check file system
        test_file = "/tmp/health_check_test"
        try:
            with open(test_file, "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("test")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            os.remove(test_file)
            components["file_system"] = True
        except Exception:
            pass
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent < 90:  # Less than 90% memory usage
            components["memory"] = True
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent < 90:  # Less than 90% CPU usage
            components["cpu"] = True
            
    except Exception as e:
        logger.error(f"Error checking core components: {e}")
    
    return components


async def check_external_services() -> Dict[str, Any]:
    """Check external service connectivity"""
    
    services = {
        "ai_models": {"status": False, "response_time": None},
        "storage_service": {"status": False, "response_time": None},
        "notification_service": {"status": False, "response_time": None}
    }
    
    try:
        # Check AI models service
        start_time = time.time()
        # Simulate AI model check
        await asyncio.sleep(0.1)
        services["ai_models"]["status"] = True
        services["ai_models"]["response_time"] = time.time() - start_time
        
        # Check storage service
        start_time = time.time()
        # Simulate storage check
        await asyncio.sleep(0.05)
        services["storage_service"]["status"] = True
        services["storage_service"]["response_time"] = time.time() - start_time
        
        # Check notification service
        start_time = time.time()
        # Simulate notification check
        await asyncio.sleep(0.03)
        services["notification_service"]["status"] = True
        services["notification_service"]["response_time"] = time.time() - start_time
        
    except Exception as e:
        logger.error(f"Error checking external services: {e}")
    
    return services


def determine_overall_status(
    components: Dict[str, bool],
    external_services: Optional[Dict[str, Any]] = None
) -> str:
    """Determine overall system health status"""
    
    # Check if all core components are healthy
    all_components_healthy = all(components.values())
    
    if not all_components_healthy:
        return "unhealthy"
    
    # Check external services if provided
    if external_services:
        critical_services = ["ai_models", "storage_service"]
        critical_services_healthy = all(
            external_services.get(service, {}).get("status", False)
            for service in critical_services
        )
        
        if not critical_services_healthy:
            return "degraded"
    
    return "healthy"


async def check_readiness() -> bool:
    """Check if system is ready to receive traffic"""
    
    try:
        # Check database connectivity
        async with get_session() as session:
            await session.execute("SELECT 1")
        
        # Check if models are loaded
        # In production, check if AI models are loaded and ready
        
        # Check if file system is accessible
        test_dir = "/tmp"
        if not os.access(test_dir, os.W_OK):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return False


async def check_liveness() -> bool:
    """Check if system is alive and responding"""
    
    try:
        # Basic liveness check - if we can reach this point, system is alive
        # In production, add more comprehensive checks
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 95:  # More than 95% memory usage
            return False
        
        # Check if main process is responsive
        # In production, check if main application loop is running
        
        return True
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return False


async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status information"""
    
    try:
        
        # System information
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0]
        }
        
        # Resource usage
        resource_usage = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        # Process information
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "memory_info": process.memory_info()._asdict(),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "create_time": process.create_time()
        }
        
        # Network information
        network_info = {
            "connections": len(process.connections()),
            "network_io": process.io_counters()._asdict() if process.io_counters() else None
        }
        
        return {
            "system_info": system_info,
            "resource_usage": resource_usage,
            "process_info": process_info,
            "network_info": network_info,
            "uptime": get_uptime(),
            "version": get_system_version()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {"error": str(e)}


# Named exports
__all__ = [
    "router",
    "health_check_roro",
    "readiness_probe",
    "liveness_probe",
    "system_status"
] 