"""
Security Router - Security and authentication endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

try:
    from security_advanced import security_manager
except ImportError:
    logging.warning("security_advanced module not available")
    security_manager = None

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/security", tags=["Security"])


@router.post("/api-key/create", response_model=Dict[str, Any])
async def create_api_key(key_data: Dict[str, Any]) -> JSONResponse:
    """Create API key"""
    logger.info("API key creation requested")
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        user_id = key_data.get("user_id")
        name = key_data.get("name")
        permissions = key_data.get("permissions", [])
        rate_limit = key_data.get("rate_limit", 1000)
        expires_days = key_data.get("expires_days")
        
        if not all([user_id, name]):
            raise ValueError("User ID and name are required")
        
        api_key = security_manager.create_api_key(user_id, name, permissions, rate_limit, expires_days)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "API key created successfully",
                "api_key": api_key,
                "user_id": user_id,
                "name": name,
                "permissions": permissions,
                "rate_limit": rate_limit
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create API key error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/api-key/validate", response_model=Dict[str, Any])
async def validate_api_key(validation_data: Dict[str, Any]) -> JSONResponse:
    """Validate API key"""
    logger.info("API key validation requested")
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        api_key = validation_data.get("api_key")
        
        if not api_key:
            raise ValueError("API key is required")
        
        key_record = security_manager.validate_api_key(api_key)
        
        if not key_record:
            raise HTTPException(status_code=401, detail="Invalid or expired API key")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "valid": True,
                "key_id": key_record.id,
                "name": key_record.name,
                "user_id": key_record.user_id,
                "permissions": key_record.permissions,
                "rate_limit": key_record.rate_limit,
                "is_active": key_record.is_active
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validate API key error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/session/create", response_model=Dict[str, Any])
async def create_session(session_data: Dict[str, Any]) -> JSONResponse:
    """Create user session"""
    logger.info("User session creation requested")
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        user_id = session_data.get("user_id")
        ip_address = session_data.get("ip_address", "127.0.0.1")
        user_agent = session_data.get("user_agent", "Unknown")
        
        if not user_id:
            raise ValueError("User ID is required")
        
        session_id = security_manager.create_user_session(user_id, ip_address, user_agent)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Session created successfully",
                "session_id": session_id,
                "user_id": user_id,
                "ip_address": ip_address
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create session error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/session/validate", response_model=Dict[str, Any])
async def validate_session(validation_data: Dict[str, Any]) -> JSONResponse:
    """Validate user session"""
    logger.info("Session validation requested")
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        session_id = validation_data.get("session_id")
        
        if not session_id:
            raise ValueError("Session ID is required")
        
        session = security_manager.validate_session(session_id)
        
        if not session:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "valid": True,
                "session_id": session.id,
                "user_id": session.user_id,
                "ip_address": session.ip_address,
                "is_active": session.is_active,
                "created_at": session.created_at,
                "last_activity": session.last_activity
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validate session error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/events", response_model=Dict[str, Any])
async def get_security_events(limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
    """Get security events"""
    logger.info("Security events requested")
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        events = security_manager.get_security_events(limit)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "events": [
                    {
                        "id": event.id,
                        "event_type": event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                        "severity": event.severity.value if hasattr(event.severity, 'value') else str(event.severity),
                        "source_ip": event.source_ip,
                        "user_id": event.user_id,
                        "description": event.description,
                        "timestamp": event.timestamp,
                        "resolved": event.resolved
                    }
                    for event in events
                ],
                "count": len(events)
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get security events error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/audit", response_model=Dict[str, Any])
async def get_audit_log(limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
    """Get audit log"""
    logger.info("Audit log requested")
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        audit_logs = security_manager.get_audit_log(limit)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "logs": [
                    {
                        "id": log.id,
                        "user_id": log.user_id,
                        "action": log.action,
                        "resource": log.resource,
                        "ip_address": log.ip_address,
                        "timestamp": log.timestamp,
                        "success": log.success
                    }
                    for log in audit_logs
                ],
                "count": len(audit_logs)
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get audit log error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=Dict[str, Any])
async def get_security_stats() -> JSONResponse:
    """Get security statistics"""
    logger.info("Security stats requested")
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        stats = security_manager.get_stats()
        return JSONResponse(content={
            "success": True,
            "data": stats,
            "error": None
        })
    except Exception as e:
        logger.error(f"Get security stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






