"""
Microservices Routes
Real, working microservices orchestration endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from microservices_system import microservices_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/microservices", tags=["Microservices Orchestration"])

@router.post("/start-service")
async def start_service(
    service_id: str = Form(...)
):
    """Start a microservice"""
    try:
        result = await microservices_system.start_service(service_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error starting service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-service")
async def stop_service(
    service_id: str = Form(...)
):
    """Stop a microservice"""
    try:
        result = await microservices_system.stop_service(service_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error stopping service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restart-service")
async def restart_service(
    service_id: str = Form(...)
):
    """Restart a microservice"""
    try:
        result = await microservices_system.restart_service(service_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error restarting service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service-health/{service_id}")
async def check_service_health(service_id: str):
    """Check service health"""
    try:
        result = await microservices_system.check_service_health(service_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error checking service health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service-status/{service_id}")
async def get_service_status(service_id: str):
    """Get service status"""
    try:
        result = await microservices_system.get_service_status(service_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-all-services")
async def start_all_services():
    """Start all services in dependency order"""
    try:
        result = await microservices_system.start_all_services()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error starting all services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-all-services")
async def stop_all_services():
    """Stop all services"""
    try:
        result = await microservices_system.stop_all_services()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error stopping all services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/services")
async def get_all_services():
    """Get all services"""
    try:
        result = microservices_system.get_all_services()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting all services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service-health-summary")
async def get_service_health_summary():
    """Get service health summary"""
    try:
        result = microservices_system.get_service_health_summary()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting service health summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/microservices-stats")
async def get_microservices_stats():
    """Get microservices statistics"""
    try:
        result = microservices_system.get_microservices_stats()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting microservices stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/microservices-dashboard")
async def get_microservices_dashboard():
    """Get comprehensive microservices dashboard"""
    try:
        # Get all microservices data
        services = microservices_system.get_all_services()
        health_summary = microservices_system.get_service_health_summary()
        stats = microservices_system.get_microservices_stats()
        
        # Get individual service statuses
        service_statuses = {}
        for service_id in services["services"]:
            status = await microservices_system.get_service_status(service_id)
            service_statuses[service_id] = status
        
        # Calculate additional metrics
        total_services = health_summary["total_services"]
        running_services = health_summary["running_services"]
        stopped_services = health_summary["stopped_services"]
        failed_services = health_summary["failed_services"]
        
        # Calculate service availability
        availability = (running_services / total_services * 100) if total_services > 0 else 0
        
        # Get services by status
        services_by_status = {
            "running": [],
            "stopped": [],
            "failed": [],
            "starting": [],
            "stopping": []
        }
        
        for service_id, status in service_statuses.items():
            if "error" not in status:
                services_by_status[status["status"]].append(service_id)
        
        dashboard_data = {
            "timestamp": stats["uptime_seconds"],
            "overview": {
                "total_services": total_services,
                "running_services": running_services,
                "stopped_services": stopped_services,
                "failed_services": failed_services,
                "availability_percent": round(availability, 2),
                "uptime_hours": stats["uptime_hours"]
            },
            "service_metrics": {
                "total_services": stats["stats"]["total_services"],
                "running_services": stats["stats"]["running_services"],
                "stopped_services": stats["stats"]["stopped_services"],
                "failed_services": stats["stats"]["failed_services"],
                "service_restarts": stats["stats"]["service_restarts"]
            },
            "services_by_status": services_by_status,
            "service_details": service_statuses,
            "health_summary": health_summary
        }
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error getting microservices dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service-dependencies")
async def get_service_dependencies():
    """Get service dependency graph"""
    try:
        services = microservices_system.get_all_services()
        dependencies = {}
        
        for service_id, service_config in services["services"].items():
            dependencies[service_id] = {
                "service_id": service_id,
                "name": service_config["name"],
                "dependencies": service_config["dependencies"],
                "dependents": []
            }
        
        # Find dependents for each service
        for service_id, dep_info in dependencies.items():
            for other_service_id, other_dep_info in dependencies.items():
                if service_id in other_dep_info["dependencies"]:
                    dep_info["dependents"].append(other_service_id)
        
        return JSONResponse(content={
            "dependencies": dependencies,
            "total_services": len(dependencies)
        })
    except Exception as e:
        logger.error(f"Error getting service dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service-topology")
async def get_service_topology():
    """Get service topology and relationships"""
    try:
        services = microservices_system.get_all_services()
        health_summary = microservices_system.get_service_health_summary()
        
        # Build topology
        topology = {
            "nodes": [],
            "edges": [],
            "clusters": {
                "ai_services": [],
                "data_services": [],
                "infrastructure_services": []
            }
        }
        
        # Add nodes (services)
        for service_id, service_config in services["services"].items():
            service_health = health_summary["service_health"][service_id]
            
            node = {
                "id": service_id,
                "name": service_config["name"],
                "port": service_config["port"],
                "status": service_health["status"],
                "dependencies": service_config["dependencies"],
                "auto_start": service_config["auto_start"],
                "restart_on_failure": service_config["restart_on_failure"]
            }
            
            topology["nodes"].append(node)
            
            # Categorize services
            if "ai" in service_id.lower():
                topology["clusters"]["ai_services"].append(service_id)
            elif any(keyword in service_id.lower() for keyword in ["upload", "backup", "analytics"]):
                topology["clusters"]["data_services"].append(service_id)
            else:
                topology["clusters"]["infrastructure_services"].append(service_id)
        
        # Add edges (dependencies)
        for service_id, service_config in services["services"].items():
            for dependency in service_config["dependencies"]:
                edge = {
                    "from": dependency,
                    "to": service_id,
                    "type": "dependency"
                }
                topology["edges"].append(edge)
        
        return JSONResponse(content=topology)
    except Exception as e:
        logger.error(f"Error getting service topology: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-microservices")
async def health_check_microservices():
    """Microservices system health check"""
    try:
        stats = microservices_system.get_microservices_stats()
        health_summary = microservices_system.get_service_health_summary()
        services = microservices_system.get_all_services()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Microservices System",
            "version": "1.0.0",
            "features": {
                "service_orchestration": True,
                "dependency_management": True,
                "health_monitoring": True,
                "auto_restart": True,
                "service_discovery": True,
                "load_balancing": True,
                "circuit_breaker": True,
                "service_mesh": True
            },
            "microservices_stats": stats["stats"],
            "system_status": {
                "total_services": health_summary["total_services"],
                "running_services": health_summary["running_services"],
                "stopped_services": health_summary["stopped_services"],
                "failed_services": health_summary["failed_services"],
                "uptime_hours": stats["uptime_hours"]
            },
            "available_services": list(services["services"].keys())
        })
    except Exception as e:
        logger.error(f"Error in microservices health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













