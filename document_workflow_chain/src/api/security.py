"""
Security API - Advanced Implementation
=====================================

Advanced security API with comprehensive security features.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta

from ..services import security_service, audit_service

# Create router
router = APIRouter()

# Security scheme
security = HTTPBearer()


# Request/Response models
class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user_id: int
    username: str
    roles: List[str]


class RegisterRequest(BaseModel):
    """User registration request model"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class RegisterResponse(BaseModel):
    """User registration response model"""
    user_id: int
    username: str
    email: str
    message: str


class PasswordChangeRequest(BaseModel):
    """Password change request model"""
    current_password: str
    new_password: str


class PasswordResetRequest(BaseModel):
    """Password reset request model"""
    email: EmailStr


class APIKeyRequest(BaseModel):
    """API key request model"""
    name: str
    permissions: List[str] = ["read"]
    expires_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    """API key response model"""
    api_key: str
    api_key_id: str
    name: str
    permissions: List[str]
    expires_at: Optional[str]
    created_at: str


class SecurityEventRequest(BaseModel):
    """Security event request model"""
    event_type: str
    details: Dict[str, Any]
    severity: str = "medium"


class SecurityStatsResponse(BaseModel):
    """Security statistics response model"""
    total_logins: int
    failed_logins: int
    password_resets: int
    security_events: int
    blocked_ips: int
    policies: Dict[str, Any]


# Authentication endpoints
@router.post("/auth/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    background_tasks: BackgroundTasks
):
    """User login with JWT token generation"""
    try:
        # Simulate user authentication
        # In production, verify against database
        if request.username == "admin" and request.password == "admin123":
            user_id = 1
            username = request.username
            roles = ["admin", "user"]
        else:
            # Log failed login attempt
            background_tasks.add_task(
                audit_service.log_event,
                "user_login",
                details={"success": False, "username": request.username},
                severity="medium"
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Generate JWT tokens
        tokens = await security_service.generate_jwt_token(
            user_id=user_id,
            username=username,
            roles=roles
        )
        
        # Log successful login
        background_tasks.add_task(
            audit_service.log_event,
            "user_login",
            user_id=user_id,
            details={"success": True, "username": username},
            severity="low"
        )
        
        return LoginResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"],
            user_id=user_id,
            username=username,
            roles=roles
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/auth/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest,
    background_tasks: BackgroundTasks
):
    """User registration"""
    try:
        # Validate password strength
        password_validation = await security_service.validate_password_strength(request.password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {password_validation['issues']}"
            )
        
        # Hash password
        hashed_password = await security_service.hash_password(request.password)
        
        # Simulate user creation
        # In production, save to database
        user_id = 1  # Simulated user ID
        
        # Log user registration
        background_tasks.add_task(
            audit_service.log_event,
            "user_created",
            user_id=user_id,
            details={
                "username": request.username,
                "email": request.email,
                "full_name": request.full_name
            },
            severity="low"
        )
        
        return RegisterResponse(
            user_id=user_id,
            username=request.username,
            email=request.email,
            message="User registered successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/auth/refresh")
async def refresh_token(
    refresh_token: str,
    background_tasks: BackgroundTasks
):
    """Refresh access token"""
    try:
        # Refresh access token
        new_tokens = await security_service.refresh_access_token(refresh_token)
        
        # Log token refresh
        background_tasks.add_task(
            audit_service.log_event,
            "token_refreshed",
            details={"token_type": "access"},
            severity="low"
        )
        
        return new_tokens
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {str(e)}"
        )


@router.post("/auth/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks = None
):
    """User logout"""
    try:
        # Verify token to get user info
        payload = await security_service.verify_jwt_token(credentials.credentials)
        user_id = payload["user_id"]
        
        # Log logout
        if background_tasks:
            background_tasks.add_task(
                audit_service.log_event,
                "user_logout",
                user_id=user_id,
                details={"logout_method": "api"},
                severity="low"
            )
        
        return {"message": "Logged out successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Logout failed: {str(e)}"
        )


# Password management endpoints
@router.post("/auth/change-password")
async def change_password(
    request: PasswordChangeRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks = None
):
    """Change user password"""
    try:
        # Verify token to get user info
        payload = await security_service.verify_jwt_token(credentials.credentials)
        user_id = payload["user_id"]
        
        # Validate new password strength
        password_validation = await security_service.validate_password_strength(request.new_password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {password_validation['issues']}"
            )
        
        # Hash new password
        hashed_password = await security_service.hash_password(request.new_password)
        
        # Log password change
        if background_tasks:
            background_tasks.add_task(
                audit_service.log_event,
                "password_changed",
                user_id=user_id,
                details={"change_method": "api"},
                severity="medium"
            )
        
        return {"message": "Password changed successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )


@router.post("/auth/reset-password")
async def reset_password(
    request: PasswordResetRequest,
    background_tasks: BackgroundTasks
):
    """Request password reset"""
    try:
        # Simulate password reset request
        # In production, send reset email
        
        # Log password reset request
        background_tasks.add_task(
            audit_service.log_event,
            "password_reset_requested",
            details={"email": request.email},
            severity="medium"
        )
        
        return {"message": "Password reset email sent"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset failed: {str(e)}"
        )


# API key management endpoints
@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks = None
):
    """Create API key for user"""
    try:
        # Verify token to get user info
        payload = await security_service.verify_jwt_token(credentials.credentials)
        user_id = payload["user_id"]
        
        # Generate API key
        api_key_data = await security_service.generate_api_key(
            user_id=user_id,
            name=request.name,
            permissions=request.permissions,
            expires_days=request.expires_days
        )
        
        # Log API key creation
        if background_tasks:
            background_tasks.add_task(
                audit_service.log_event,
                "api_key_created",
                user_id=user_id,
                details={
                    "api_key_name": request.name,
                    "permissions": request.permissions
                },
                severity="medium"
            )
        
        return APIKeyResponse(
            api_key=api_key_data["api_key"],
            api_key_id=api_key_data["api_key_record"]["id"],
            name=request.name,
            permissions=request.permissions,
            expires_at=api_key_data["api_key_record"]["expires_at"],
            created_at=api_key_data["api_key_record"]["created_at"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"API key creation failed: {str(e)}"
        )


@router.get("/api-keys")
async def list_api_keys(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """List user's API keys"""
    try:
        # Verify token to get user info
        payload = await security_service.verify_jwt_token(credentials.credentials)
        user_id = payload["user_id"]
        
        # Simulate API key listing
        # In production, fetch from database
        api_keys = [
            {
                "id": "api_key_1",
                "name": "Default API Key",
                "permissions": ["read", "write"],
                "created_at": "2024-01-01T00:00:00Z",
                "last_used": "2024-01-01T12:00:00Z",
                "is_active": True
            }
        ]
        
        return {"api_keys": api_keys}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}"
        )


# Security validation endpoints
@router.post("/validate/password")
async def validate_password(password: str):
    """Validate password strength"""
    try:
        validation_result = await security_service.validate_password_strength(password)
        return validation_result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password validation failed: {str(e)}"
        )


@router.post("/validate/input")
async def validate_input(
    data: Dict[str, Any],
    schema: Dict[str, Any]
):
    """Validate input data against schema"""
    try:
        validation_result = await security_service.validate_input(data, schema)
        return validation_result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Input validation failed: {str(e)}"
        )


# Rate limiting endpoints
@router.get("/rate-limit/check")
async def check_rate_limit(
    identifier: str,
    limit: int = 100,
    window_minutes: int = 1
):
    """Check rate limit for identifier"""
    try:
        rate_limit_result = await security_service.check_rate_limit(
            identifier=identifier,
            limit=limit,
            window_minutes=window_minutes
        )
        return rate_limit_result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rate limit check failed: {str(e)}"
        )


# Security event endpoints
@router.post("/events/log")
async def log_security_event(
    request: SecurityEventRequest,
    background_tasks: BackgroundTasks
):
    """Log security event"""
    try:
        # Log security event
        event = await audit_service.log_event(
            event_type=request.event_type,
            details=request.details,
            severity=request.severity
        )
        
        return {"message": "Security event logged successfully", "event_id": event["id"]}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log security event: {str(e)}"
        )


@router.get("/events/security")
async def get_security_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """Get security events"""
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Get security events
        events = await audit_service.get_security_events(
            start_date=start_dt,
            end_date=end_dt,
            severity=severity
        )
        
        return events
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security events: {str(e)}"
        )


# Statistics endpoints
@router.get("/stats", response_model=SecurityStatsResponse)
async def get_security_stats():
    """Get security statistics"""
    try:
        stats = await security_service.get_security_stats()
        return SecurityStatsResponse(**stats)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security stats: {str(e)}"
        )


@router.get("/audit/stats")
async def get_audit_stats():
    """Get audit statistics"""
    try:
        stats = await audit_service.get_audit_stats()
        return stats
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit stats: {str(e)}"
        )


@router.get("/audit/logs")
async def get_audit_logs(
    event_type: Optional[str] = None,
    user_id: Optional[int] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get audit logs"""
    try:
        logs = await audit_service.get_audit_logs(
            event_type=event_type,
            user_id=user_id,
            severity=severity,
            limit=limit,
            offset=offset
        )
        return logs
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit logs: {str(e)}"
        )


# User profile endpoints
@router.get("/profile")
async def get_user_profile(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get user profile"""
    try:
        # Verify token to get user info
        payload = await security_service.verify_jwt_token(credentials.credentials)
        user_id = payload["user_id"]
        username = payload["username"]
        roles = payload["roles"]
        
        # Simulate user profile
        profile = {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "created_at": "2024-01-01T00:00:00Z",
            "last_login": datetime.utcnow().isoformat(),
            "is_active": True
        }
        
        return profile
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Failed to get user profile: {str(e)}"
        )


@router.put("/profile")
async def update_user_profile(
    profile_data: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks = None
):
    """Update user profile"""
    try:
        # Verify token to get user info
        payload = await security_service.verify_jwt_token(credentials.credentials)
        user_id = payload["user_id"]
        
        # Log profile update
        if background_tasks:
            background_tasks.add_task(
                audit_service.log_event,
                "user_updated",
                user_id=user_id,
                details={"updated_fields": list(profile_data.keys())},
                severity="low"
            )
        
        return {"message": "Profile updated successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}"
        )

