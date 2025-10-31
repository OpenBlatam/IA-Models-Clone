"""
Advanced Security API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from pydantic import BaseModel, Field

from ....services.advanced_security_service import AdvancedSecurityService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError, SecurityError

router = APIRouter()


class PasswordValidationRequest(BaseModel):
    """Request model for password validation."""
    password: str = Field(..., min_length=1, description="Password to validate")


class TokenVerificationRequest(BaseModel):
    """Request model for token verification."""
    token: str = Field(..., description="JWT token to verify")


class DataEncryptionRequest(BaseModel):
    """Request model for data encryption."""
    data: str = Field(..., description="Data to encrypt")


class DataDecryptionRequest(BaseModel):
    """Request model for data decryption."""
    encrypted_data: str = Field(..., description="Encrypted data to decrypt")


class SecurityPolicyUpdateRequest(BaseModel):
    """Request model for security policy update."""
    policy_name: str = Field(..., description="Name of the policy to update")
    policy_value: Any = Field(..., description="New policy value")


class TwoFactorSetupRequest(BaseModel):
    """Request model for 2FA setup."""
    user_id: str = Field(..., description="User ID to setup 2FA for")


class TwoFactorVerificationRequest(BaseModel):
    """Request model for 2FA verification."""
    user_id: str = Field(..., description="User ID")
    token: str = Field(..., description="2FA token")


async def get_security_service(session: DatabaseSessionDep) -> AdvancedSecurityService:
    """Get security service instance."""
    return AdvancedSecurityService(session)


@router.post("/password/validate", response_model=Dict[str, Any])
async def validate_password_strength(
    request: PasswordValidationRequest = Depends(),
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Validate password strength against security policies."""
    try:
        result = await security_service.validate_password_strength(request.password)
        
        return {
            "success": True,
            "data": result,
            "message": "Password validation completed"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate password"
        )


@router.post("/token/generate", response_model=Dict[str, Any])
async def generate_secure_token(
    token_type: str = Query(default="access", description="Type of token to generate"),
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Generate a secure JWT token."""
    try:
        token = await security_service.generate_secure_token(
            user_id=str(current_user.id),
            token_type=token_type
        )
        
        return {
            "success": True,
            "data": {
                "token": token,
                "token_type": token_type,
                "user_id": str(current_user.id)
            },
            "message": "Token generated successfully"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate token"
        )


@router.post("/token/verify", response_model=Dict[str, Any])
async def verify_token(
    request: TokenVerificationRequest = Depends(),
    security_service: AdvancedSecurityService = Depends(get_security_service)
):
    """Verify and decode a JWT token."""
    try:
        result = await security_service.verify_token(request.token)
        
        return {
            "success": True,
            "data": result,
            "message": "Token verification completed"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify token"
        )


@router.post("/encrypt", response_model=Dict[str, Any])
async def encrypt_sensitive_data(
    request: DataEncryptionRequest = Depends(),
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Encrypt sensitive data."""
    try:
        encrypted_data = await security_service.encrypt_sensitive_data(request.data)
        
        return {
            "success": True,
            "data": {
                "encrypted_data": encrypted_data,
                "original_length": len(request.data)
            },
            "message": "Data encrypted successfully"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encrypt data"
        )


@router.post("/decrypt", response_model=Dict[str, Any])
async def decrypt_sensitive_data(
    request: DataDecryptionRequest = Depends(),
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Decrypt sensitive data."""
    try:
        decrypted_data = await security_service.decrypt_sensitive_data(request.encrypted_data)
        
        return {
            "success": True,
            "data": {
                "decrypted_data": decrypted_data,
                "encrypted_length": len(request.encrypted_data)
            },
            "message": "Data decrypted successfully"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decrypt data"
        )


@router.get("/login-attempts/{user_id}", response_model=Dict[str, Any])
async def check_login_attempts(
    user_id: str,
    request: Request,
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Check if user has exceeded login attempts."""
    try:
        # Get client IP
        client_ip = request.client.host if request.client else "127.0.0.1"
        
        result = await security_service.check_login_attempts(user_id, client_ip)
        
        return {
            "success": True,
            "data": result,
            "message": "Login attempt check completed"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check login attempts"
        )


@router.post("/login-attempts/{user_id}/record", response_model=Dict[str, Any])
async def record_failed_login(
    user_id: str,
    reason: str = Query(default="invalid_password", description="Reason for failed login"),
    request: Request = None,
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Record a failed login attempt."""
    try:
        # Get client IP
        client_ip = request.client.host if request.client else "127.0.0.1"
        
        await security_service.record_failed_login(user_id, client_ip, reason)
        
        return {
            "success": True,
            "message": "Failed login attempt recorded"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record failed login"
        )


@router.delete("/login-attempts/{user_id}/clear", response_model=Dict[str, Any])
async def clear_failed_logins(
    user_id: str,
    request: Request,
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Clear failed login attempts for a user."""
    try:
        # Get client IP
        client_ip = request.client.host if request.client else "127.0.0.1"
        
        await security_service.clear_failed_logins(user_id, client_ip)
        
        return {
            "success": True,
            "message": "Failed login attempts cleared"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear failed logins"
        )


@router.get("/ip-address/{ip_address}/validate", response_model=Dict[str, Any])
async def validate_ip_address(
    ip_address: str,
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Validate and analyze IP address."""
    try:
        result = await security_service.validate_ip_address(ip_address)
        
        return {
            "success": True,
            "data": result,
            "message": "IP address validation completed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate IP address"
        )


@router.get("/events", response_model=Dict[str, Any])
async def get_security_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of events to return"),
    offset: int = Query(default=0, ge=0, description="Number of events to skip"),
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Get security events."""
    try:
        events = await security_service.get_security_events(
            event_type=event_type,
            user_id=user_id,
            severity=severity,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "data": events,
            "message": "Security events retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security events"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_security_stats(
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Get security statistics."""
    try:
        stats = await security_service.get_security_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Security statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security statistics"
        )


@router.put("/policies", response_model=Dict[str, Any])
async def update_security_policy(
    request: SecurityPolicyUpdateRequest = Depends(),
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Update a security policy."""
    try:
        result = await security_service.update_security_policy(
            policy_name=request.policy_name,
            policy_value=request.policy_value
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Security policy updated successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update security policy"
        )


@router.post("/2fa/setup", response_model=Dict[str, Any])
async def setup_two_factor_auth(
    request: TwoFactorSetupRequest = Depends(),
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Setup two-factor authentication for a user."""
    try:
        secret = await security_service.generate_2fa_secret(request.user_id)
        
        return {
            "success": True,
            "data": {
                "user_id": request.user_id,
                "secret": secret,
                "qr_code_url": f"otpauth://totp/BlogSystem:{request.user_id}?secret={secret}&issuer=BlogSystem"
            },
            "message": "2FA setup completed successfully"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to setup 2FA"
        )


@router.post("/2fa/verify", response_model=Dict[str, Any])
async def verify_two_factor_auth(
    request: TwoFactorVerificationRequest = Depends(),
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Verify two-factor authentication token."""
    try:
        is_valid = await security_service.verify_2fa_token(request.user_id, request.token)
        
        return {
            "success": True,
            "data": {
                "user_id": request.user_id,
                "valid": is_valid
            },
            "message": "2FA verification completed"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify 2FA token"
        )


@router.post("/audit", response_model=Dict[str, Any])
async def perform_security_audit(
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Perform a comprehensive security audit."""
    try:
        audit_result = await security_service.perform_security_audit()
        
        return {
            "success": True,
            "data": audit_result,
            "message": "Security audit completed successfully"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform security audit"
        )


@router.post("/anomaly-detection", response_model=Dict[str, Any])
async def detect_anomalous_activity(
    user_id: str = Query(..., description="User ID to check"),
    request: Request = None,
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Detect anomalous user activity."""
    try:
        # Get client IP
        client_ip = request.client.host if request.client else "127.0.0.1"
        
        result = await security_service.detect_anomalous_activity(user_id, client_ip)
        
        return {
            "success": True,
            "data": result,
            "message": "Anomaly detection completed"
        }
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect anomalous activity"
        )


@router.get("/policies", response_model=Dict[str, Any])
async def get_security_policies(
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Get current security policies."""
    try:
        return {
            "success": True,
            "data": {
                "policies": security_service.security_policies,
                "total_policies": len(security_service.security_policies)
            },
            "message": "Security policies retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security policies"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_security_health(
    security_service: AdvancedSecurityService = Depends(get_security_service),
    current_user: CurrentUserDep = Depends()
):
    """Get security system health status."""
    try:
        # Get security stats
        stats = await security_service.get_security_stats()
        
        # Calculate health score
        total_events = stats.get("total_security_events", 0)
        failed_logins = stats.get("total_failed_logins", 0)
        
        health_score = 100
        if total_events > 1000:
            health_score -= 20
        if failed_logins > 100:
            health_score -= 30
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_security_events": total_events,
                "total_failed_logins": failed_logins,
                "events_by_severity": stats.get("events_by_severity", {}),
                "timestamp": "2024-01-15T10:00:00Z"
            },
            "message": "Security health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security health status"
        )

























