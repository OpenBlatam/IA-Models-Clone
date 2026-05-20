"""
API Gateway Routes
Real, working API gateway endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from api_gateway_system import api_gateway_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/gateway", tags=["API Gateway"])

@router.post("/register-route")
async def register_route(
    path: str = Form(...),
    service: str = Form(...),
    endpoint: str = Form(...),
    methods: List[str] = Form(["GET", "POST"]),
    rate_limit: int = Form(100),
    timeout: int = Form(30)
):
    """Register a new route"""
    try:
        result = await api_gateway_system.register_route(
            path, service, endpoint, methods, rate_limit, timeout
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error registering route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/unregister-route/{path:path}")
async def unregister_route(path: str):
    """Unregister a route"""
    try:
        result = await api_gateway_system.unregister_route(path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error unregistering route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/routes")
async def get_routes():
    """Get all registered routes"""
    try:
        result = api_gateway_system.get_routes()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-rate-limit")
async def check_rate_limit(
    client_ip: str = Form(...),
    route: str = Form(...)
):
    """Check rate limit for client"""
    try:
        result = await api_gateway_system.check_rate_limit(client_ip, route)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-api-key")
async def generate_api_key(
    service: str = Form(...),
    expires_hours: int = Form(24)
):
    """Generate API key for service access"""
    try:
        result = await api_gateway_system.generate_api_key(service, expires_hours)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error generating API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-api-key")
async def validate_api_key(
    api_key: str = Form(...)
):
    """Validate API key"""
    try:
        result = await api_gateway_system.validate_api_key(api_key)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/revoke-api-key")
async def revoke_api_key(
    api_key: str = Form(...)
):
    """Revoke API key"""
    try:
        result = await api_gateway_system.revoke_api_key(api_key)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api-keys")
async def get_api_keys(
    active_only: bool = True
):
    """Get API keys"""
    try:
        result = api_gateway_system.get_api_keys(active_only)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/circuit-breaker/{service}")
async def check_circuit_breaker(service: str):
    """Check circuit breaker status for service"""
    try:
        result = await api_gateway_system.check_circuit_breaker(service)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error checking circuit breaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/circuit-breaker/{service}/failure")
async def record_circuit_breaker_failure(service: str):
    """Record circuit breaker failure"""
    try:
        result = await api_gateway_system.record_circuit_breaker_failure(service)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error recording circuit breaker failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/circuit-breaker/{service}/success")
async def record_circuit_breaker_success(service: str):
    """Record circuit breaker success"""
    try:
        result = await api_gateway_system.record_circuit_breaker_success(service)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error recording circuit breaker success: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/circuit-breakers")
async def get_circuit_breakers():
    """Get all circuit breaker status"""
    try:
        result = api_gateway_system.get_circuit_breakers()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/request-logs")
async def get_request_logs(
    limit: int = 100
):
    """Get recent request logs"""
    try:
        result = api_gateway_system.get_request_logs(limit)
        return JSONResponse(content={
            "request_logs": result,
            "total_logs": len(api_gateway_system.request_logs)
        })
    except Exception as e:
        logger.error(f"Error getting request logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gateway-stats")
async def get_gateway_stats():
    """Get gateway statistics"""
    try:
        result = api_gateway_system.get_gateway_stats()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting gateway stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/middleware")
async def get_middleware():
    """Get all middleware"""
    try:
        result = api_gateway_system.get_middleware()
        return JSONResponse(content={
            "middleware": result,
            "middleware_count": len(result)
        })
    except Exception as e:
        logger.error(f"Error getting middleware: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/log-request")
async def log_request(
    client_ip: str = Form(...),
    method: str = Form(...),
    path: str = Form(...),
    status_code: int = Form(...),
    response_time: float = Form(...),
    service: Optional[str] = Form(None)
):
    """Log API request"""
    try:
        result = await api_gateway_system.log_request(
            client_ip, method, path, status_code, response_time, service
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error logging request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gateway-dashboard")
async def get_gateway_dashboard():
    """Get comprehensive gateway dashboard"""
    try:
        # Get all gateway data
        routes = api_gateway_system.get_routes()
        middleware = api_gateway_system.get_middleware()
        api_keys = api_gateway_system.get_api_keys()
        circuit_breakers = api_gateway_system.get_circuit_breakers()
        request_logs = api_gateway_system.get_request_logs(50)
        stats = api_gateway_system.get_gateway_stats()
        
        # Calculate additional metrics
        recent_requests = len(request_logs)
        successful_requests = len([log for log in request_logs if 200 <= log["status_code"] < 300])
        failed_requests = len([log for log in request_logs if log["status_code"] >= 400])
        
        # Calculate average response time
        if request_logs:
            avg_response_time = sum(log["response_time"] for log in request_logs) / len(request_logs)
        else:
            avg_response_time = 0
        
        # Count services by status
        open_circuits = len([cb for cb in circuit_breakers["circuit_breakers"].values() if cb["state"] == "open"])
        closed_circuits = len([cb for cb in circuit_breakers["circuit_breakers"].values() if cb["state"] == "closed"])
        half_open_circuits = len([cb for cb in circuit_breakers["circuit_breakers"].values() if cb["state"] == "half-open"])
        
        dashboard_data = {
            "timestamp": stats["uptime_seconds"],
            "overview": {
                "total_routes": routes["route_count"],
                "total_middleware": len(middleware),
                "active_api_keys": api_keys["active_count"],
                "total_services": circuit_breakers["total_services"],
                "uptime_hours": stats["uptime_hours"]
            },
            "request_metrics": {
                "total_requests": stats["stats"]["total_requests"],
                "successful_requests": stats["stats"]["successful_requests"],
                "failed_requests": stats["stats"]["failed_requests"],
                "rate_limited_requests": stats["stats"]["rate_limited_requests"],
                "success_rate": (stats["stats"]["successful_requests"] / stats["stats"]["total_requests"] * 100) if stats["stats"]["total_requests"] > 0 else 0,
                "average_response_time": round(avg_response_time, 3)
            },
            "circuit_breakers": {
                "total_services": circuit_breakers["total_services"],
                "open_circuits": open_circuits,
                "closed_circuits": closed_circuits,
                "half_open_circuits": half_open_circuits,
                "circuit_breaker_trips": stats["stats"]["circuit_breaker_trips"]
            },
            "routes": routes["routes"],
            "recent_requests": request_logs,
            "circuit_breaker_details": circuit_breakers["circuit_breakers"]
        }
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error getting gateway dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-gateway")
async def health_check_gateway():
    """API Gateway system health check"""
    try:
        stats = api_gateway_system.get_gateway_stats()
        routes = api_gateway_system.get_routes()
        middleware = api_gateway_system.get_middleware()
        api_keys = api_gateway_system.get_api_keys()
        circuit_breakers = api_gateway_system.get_circuit_breakers()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "API Gateway System",
            "version": "1.0.0",
            "features": {
                "route_registration": True,
                "rate_limiting": True,
                "api_key_management": True,
                "circuit_breaker": True,
                "request_logging": True,
                "middleware_support": True,
                "service_discovery": True,
                "load_balancing": True
            },
            "gateway_stats": stats["stats"],
            "system_status": {
                "total_routes": routes["route_count"],
                "middleware_count": len(middleware),
                "active_api_keys": api_keys["active_count"],
                "circuit_breakers": circuit_breakers["total_services"],
                "uptime_hours": stats["uptime_hours"]
            }
        })
    except Exception as e:
        logger.error(f"Error in gateway health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













