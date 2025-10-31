"""
Security API routes for Facebook Posts API
Advanced security, authentication, authorization, and threat protection
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..core.config import get_settings
from ..api.schemas import ErrorResponse
from ..api.dependencies import get_request_id
from ..services.security_service import (
    get_security_service, SecurityLevel, ThreatType, AuthenticationMethod,
    SecurityEvent, UserSession, SecurityPolicy
)
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/security", tags=["Security"])

# Security scheme
security = HTTPBearer()


# Authentication Routes

@router.post(
    "/auth/login",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Authentication successful"},
        401: {"description": "Authentication failed"},
        429: {"description": "Too many failed attempts"},
        500: {"description": "Authentication error"}
    },
    summary="User authentication",
    description="Authenticate user with username and password"
)
@timed("security_authentication")
async def authenticate_user(
    username: str = Query(..., description="Username"),
    password: str = Query(..., description="Password"),
    request: Request = None,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Authenticate user"""
    
    # Extract request information
    ip_address = request.client.host if request else "unknown"
    user_agent = request.headers.get("user-agent", "unknown") if request else "unknown"
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Authenticate user
        auth_result = await security_service.authenticate_user(
            username=username,
            password=password,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if not auth_result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
        
        logger.info(
            "User authenticated successfully",
            username=username,
            ip_address=ip_address,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Authentication successful",
            "data": auth_result,
            "request_id": request_id,
            "authenticated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Authentication failed",
            username=username,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )


@router.post(
    "/auth/logout",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Logout successful"},
        401: {"description": "Unauthorized"},
        500: {"description": "Logout error"}
    },
    summary="User logout",
    description="Logout user and invalidate session"
)
@timed("security_logout")
async def logout_user(
    session_id: str = Query(..., description="Session ID"),
    token: Optional[str] = Query(None, description="Access token"),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Logout user"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Logout user
        await security_service.logout_user(session_id=session_id, token=token)
        
        logger.info(
            "User logged out successfully",
            session_id=session_id,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Logout successful",
            "request_id": request_id,
            "logged_out_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Logout failed",
            session_id=session_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )


@router.post(
    "/auth/refresh",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Token refresh successful"},
        401: {"description": "Invalid refresh token"},
        500: {"description": "Token refresh error"}
    },
    summary="Refresh access token",
    description="Refresh access token using refresh token"
)
@timed("security_token_refresh")
async def refresh_access_token(
    refresh_token: str = Query(..., description="Refresh token"),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Refresh access token"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Verify refresh token
        payload = security_service.token_manager.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Generate new access token
        user_id = payload.get("user_id")
        new_access_token = security_service.token_manager.generate_access_token(user_id)
        
        logger.info(
            "Access token refreshed successfully",
            user_id=user_id,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Token refresh successful",
            "access_token": new_access_token,
            "request_id": request_id,
            "refreshed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Token refresh failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )


# Authorization Routes

@router.get(
    "/auth/verify",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Token verification successful"},
        401: {"description": "Invalid token"},
        500: {"description": "Token verification error"}
    },
    summary="Verify access token",
    description="Verify access token and get user information"
)
@timed("security_token_verification")
async def verify_access_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Verify access token"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Verify token
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Check if token is blacklisted
        if security_service.token_manager.is_token_blacklisted(credentials.credentials):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        
        logger.info(
            "Token verified successfully",
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Token verification successful",
            "user_id": payload.get("user_id"),
            "token_payload": payload,
            "request_id": request_id,
            "verified_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Token verification failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token verification failed: {str(e)}"
        )


@router.post(
    "/auth/authorize",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Authorization successful"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        500: {"description": "Authorization error"}
    },
    summary="Authorize request",
    description="Authorize request with required permissions"
)
@timed("security_authorization")
async def authorize_request(
    required_permissions: List[str] = Query(..., description="Required permissions"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Authorize request"""
    
    # Extract request information
    ip_address = request.client.host if request else "unknown"
    user_agent = request.headers.get("user-agent", "unknown") if request else "unknown"
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Authorize request
        auth_result = await security_service.authorize_request(
            token=credentials.credentials,
            required_permissions=required_permissions,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if not auth_result:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        logger.info(
            "Request authorized successfully",
            user_id=auth_result.get("user_id"),
            permissions=required_permissions,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Authorization successful",
            "user_id": auth_result.get("user_id"),
            "permissions": auth_result.get("permissions"),
            "request_id": request_id,
            "authorized_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Authorization failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authorization failed: {str(e)}"
        )


# Password Management Routes

@router.post(
    "/password/validate",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Password validation successful"},
        400: {"description": "Invalid password"},
        500: {"description": "Password validation error"}
    },
    summary="Validate password",
    description="Validate password against security policies"
)
@timed("security_password_validation")
async def validate_password(
    password: str = Query(..., description="Password to validate"),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Validate password"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Validate password
        validation_result = await security_service.validate_password(password)
        
        logger.info(
            "Password validation completed",
            is_valid=validation_result["is_valid"],
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Password validation completed",
            "validation_result": validation_result,
            "request_id": request_id,
            "validated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Password validation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password validation failed: {str(e)}"
        )


@router.get(
    "/password/generate",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Password generated successfully"},
        500: {"description": "Password generation error"}
    },
    summary="Generate secure password",
    description="Generate a secure random password"
)
@timed("security_password_generation")
async def generate_secure_password(
    length: int = Query(16, description="Password length", ge=8, le=64),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Generate secure password"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Generate password
        password = security_service.password_manager.generate_secure_password(length)
        
        # Validate generated password
        validation_result = await security_service.validate_password(password)
        
        logger.info(
            "Secure password generated",
            length=length,
            is_valid=validation_result["is_valid"],
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Secure password generated",
            "password": password,
            "length": length,
            "validation_result": validation_result,
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Password generation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password generation failed: {str(e)}"
        )


# Encryption Routes

@router.post(
    "/encrypt",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Data encrypted successfully"},
        400: {"description": "Invalid data"},
        500: {"description": "Encryption error"}
    },
    summary="Encrypt sensitive data",
    description="Encrypt sensitive data using advanced encryption"
)
@timed("security_encryption")
async def encrypt_data(
    data: str = Query(..., description="Data to encrypt"),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Encrypt sensitive data"""
    
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data is required"
        )
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Encrypt data
        encrypted_data = await security_service.encrypt_sensitive_data(data)
        
        if not encrypted_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Encryption failed"
            )
        
        logger.info(
            "Data encrypted successfully",
            data_length=len(data),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Data encrypted successfully",
            "encrypted_data": encrypted_data,
            "original_length": len(data),
            "request_id": request_id,
            "encrypted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Data encryption failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data encryption failed: {str(e)}"
        )


@router.post(
    "/decrypt",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Data decrypted successfully"},
        400: {"description": "Invalid encrypted data"},
        500: {"description": "Decryption error"}
    },
    summary="Decrypt sensitive data",
    description="Decrypt sensitive data using advanced decryption"
)
@timed("security_decryption")
async def decrypt_data(
    encrypted_data: str = Query(..., description="Encrypted data to decrypt"),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Decrypt sensitive data"""
    
    if not encrypted_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Encrypted data is required"
        )
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Decrypt data
        decrypted_data = await security_service.decrypt_sensitive_data(encrypted_data)
        
        if not decrypted_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Decryption failed"
            )
        
        logger.info(
            "Data decrypted successfully",
            encrypted_length=len(encrypted_data),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Data decrypted successfully",
            "decrypted_data": decrypted_data,
            "encrypted_length": len(encrypted_data),
            "request_id": request_id,
            "decrypted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Data decryption failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data decryption failed: {str(e)}"
        )


# Session Management Routes

@router.get(
    "/sessions",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Sessions retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Session retrieval error"}
    },
    summary="Get active sessions",
    description="Get all active user sessions"
)
@timed("security_session_retrieval")
async def get_active_sessions(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get active sessions"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Verify token
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get active sessions
        active_sessions = []
        for session_id, session in security_service.active_sessions.items():
            if session.user_id == payload.get("user_id"):
                active_sessions.append({
                    "session_id": session.session_id,
                    "ip_address": session.ip_address,
                    "user_agent": session.user_agent,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "expires_at": session.expires_at.isoformat(),
                    "is_active": session.is_active
                })
        
        logger.info(
            "Active sessions retrieved",
            user_id=payload.get("user_id"),
            session_count=len(active_sessions),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Active sessions retrieved successfully",
            "sessions": active_sessions,
            "total_count": len(active_sessions),
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Session retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session retrieval failed: {str(e)}"
        )


@router.delete(
    "/sessions/{session_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Session terminated successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Session not found"},
        500: {"description": "Session termination error"}
    },
    summary="Terminate session",
    description="Terminate a specific user session"
)
@timed("security_session_termination")
async def terminate_session(
    session_id: str = Path(..., description="Session ID to terminate"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Terminate session"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Verify token
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Check if session exists and belongs to user
        session = security_service.active_sessions.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session.user_id != payload.get("user_id"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Terminate session
        del security_service.active_sessions[session_id]
        
        logger.info(
            "Session terminated successfully",
            session_id=session_id,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Session terminated successfully",
            "session_id": session_id,
            "request_id": request_id,
            "terminated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Session termination failed",
            session_id=session_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session termination failed: {str(e)}"
        )


# Security Policy Routes

@router.get(
    "/policies",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Security policies retrieved successfully"},
        500: {"description": "Policy retrieval error"}
    },
    summary="Get security policies",
    description="Get all security policies"
)
@timed("security_policy_retrieval")
async def get_security_policies(
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get security policies"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Get policies
        policies = []
        for policy_id, policy in security_service.policy_engine.policies.items():
            policies.append({
                "id": policy.id,
                "name": policy.name,
                "description": policy.description,
                "rules": policy.rules,
                "enabled": policy.enabled,
                "priority": policy.priority,
                "created_at": policy.created_at.isoformat(),
                "updated_at": policy.updated_at.isoformat()
            })
        
        logger.info(
            "Security policies retrieved",
            policies_count=len(policies),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Security policies retrieved successfully",
            "policies": policies,
            "total_count": len(policies),
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Security policy retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security policy retrieval failed: {str(e)}"
        )


@router.post(
    "/policies/evaluate",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Policy evaluation successful"},
        400: {"description": "Invalid policy or context"},
        500: {"description": "Policy evaluation error"}
    },
    summary="Evaluate security policy",
    description="Evaluate security policy against context"
)
@timed("security_policy_evaluation")
async def evaluate_security_policy(
    policy_id: str = Query(..., description="Policy ID to evaluate"),
    context: Dict[str, Any] = Query(..., description="Context for policy evaluation"),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Evaluate security policy"""
    
    try:
        # Get security service
        security_service = get_security_service()
        
        # Evaluate policy
        is_compliant = await security_service.policy_engine.evaluate_policy(policy_id, context)
        
        logger.info(
            "Security policy evaluated",
            policy_id=policy_id,
            is_compliant=is_compliant,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Policy evaluation completed",
            "policy_id": policy_id,
            "is_compliant": is_compliant,
            "context": context,
            "request_id": request_id,
            "evaluated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Security policy evaluation failed",
            policy_id=policy_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security policy evaluation failed: {str(e)}"
        )


# Export router
__all__ = ["router"]






























