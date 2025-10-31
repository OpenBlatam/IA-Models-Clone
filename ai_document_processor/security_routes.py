"""
Security Routes
Real, working security endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import asyncio
from security_system import security_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/security", tags=["Security"])

async def get_client_ip(request: Request) -> str:
    """Get client IP address"""
    return request.client.host

@router.post("/validate-request")
async def validate_request(
    request_data: dict,
    client_ip: str = Depends(get_client_ip),
    api_key: Optional[str] = None
):
    """Validate request for security"""
    try:
        result = await security_system.validate_request(request_data, client_ip, api_key)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error validating request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-file")
async def validate_file_upload(
    file_content: bytes,
    filename: str,
    client_ip: str = Depends(get_client_ip)
):
    """Validate file upload for security"""
    try:
        result = await security_system.validate_file_upload(file_content, filename, client_ip)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error validating file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-api-key")
async def generate_api_key(
    expires_hours: int = 24
):
    """Generate new API key"""
    try:
        api_key = await security_system.generate_api_key(expires_hours)
        if not api_key:
            raise HTTPException(status_code=500, detail="Failed to generate API key")
        
        return JSONResponse(content={
            "api_key": api_key,
            "expires_hours": expires_hours,
            "message": "API key generated successfully"
        })
    except Exception as e:
        logger.error(f"Error generating API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/block-ip")
async def block_ip(
    client_ip: str,
    reason: str = "Security violation"
):
    """Block IP address"""
    try:
        await security_system.block_ip(client_ip, reason)
        return JSONResponse(content={
            "message": f"IP {client_ip} blocked successfully",
            "reason": reason
        })
    except Exception as e:
        logger.error(f"Error blocking IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unblock-ip")
async def unblock_ip(
    client_ip: str
):
    """Unblock IP address"""
    try:
        await security_system.unblock_ip(client_ip)
        return JSONResponse(content={
            "message": f"IP {client_ip} unblocked successfully"
        })
    except Exception as e:
        logger.error(f"Error unblocking IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/security-stats")
async def get_security_stats():
    """Get security statistics"""
    try:
        stats = security_system.get_security_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting security stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/security-config")
async def get_security_config():
    """Get security configuration"""
    try:
        config = security_system.get_security_config()
        return JSONResponse(content=config)
    except Exception as e:
        logger.error(f"Error getting security config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/security-logs")
async def get_security_logs():
    """Get recent security logs"""
    try:
        logs = security_system.security_logs[-50:]  # Last 50 events
        return JSONResponse(content={
            "security_logs": logs,
            "total_events": len(security_system.security_logs)
        })
    except Exception as e:
        logger.error(f"Error getting security logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/blocked-ips")
async def get_blocked_ips():
    """Get list of blocked IP addresses"""
    try:
        blocked_ips = list(security_system.blocked_ips)
        return JSONResponse(content={
            "blocked_ips": blocked_ips,
            "count": len(blocked_ips)
        })
    except Exception as e:
        logger.error(f"Error getting blocked IPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rate-limits")
async def get_rate_limits():
    """Get current rate limit status"""
    try:
        rate_limits = {}
        for ip, data in security_system.rate_limits.items():
            current_time = time.time()
            minute_requests = len([
                req_time for req_time in data["requests"]
                if current_time - req_time < 60
            ])
            hour_requests = len([
                req_time for req_time in data["requests"]
                if current_time - req_time < 3600
            ])
            
            rate_limits[ip] = {
                "minute_requests": minute_requests,
                "hour_requests": hour_requests,
                "total_requests": len(data["requests"])
            }
        
        return JSONResponse(content={
            "rate_limits": rate_limits,
            "max_per_minute": security_system.max_requests_per_minute,
            "max_per_hour": security_system.max_requests_per_hour
        })
    except Exception as e:
        logger.error(f"Error getting rate limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-security")
async def health_check_security():
    """Security system health check"""
    try:
        stats = security_system.get_security_stats()
        config = security_system.get_security_config()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Security System",
            "version": "1.0.0",
            "features": {
                "request_validation": True,
                "file_validation": True,
                "rate_limiting": True,
                "ip_blocking": True,
                "api_key_management": True,
                "security_logging": True,
                "malicious_content_detection": True
            },
            "security_stats": stats["stats"],
            "security_config": {
                "max_requests_per_minute": config["max_requests_per_minute"],
                "max_requests_per_hour": config["max_requests_per_hour"],
                "max_file_size": config["max_file_size"],
                "allowed_file_types": config["allowed_file_types"],
                "blocked_ips_count": len(config["blocked_ips"])
            }
        })
    except Exception as e:
        logger.error(f"Error in security health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













