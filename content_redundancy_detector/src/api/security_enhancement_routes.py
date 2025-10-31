"""
Security Enhancement API Routes - Advanced security and threat protection endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.security_enhancement_engine import (
    get_security_enhancement_engine, SecurityConfig, 
    SecurityEvent, SecurityMetrics, ThreatIntelligence
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/security-enhancement", tags=["Security Enhancement"])


# Request/Response Models
class SecurityAnalysisRequest(BaseModel):
    """Security analysis request model"""
    content: str = Field(..., description="Content to analyze for security threats", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for analysis")
    include_recommendations: bool = Field(default=True, description="Include security recommendations")
    threat_types: Optional[List[str]] = Field(default=None, description="Specific threat types to check")


class EncryptionRequest(BaseModel):
    """Encryption request model"""
    content: str = Field(..., description="Content to encrypt", min_length=1)
    encryption_type: str = Field(default="fernet", description="Encryption type (fernet, rsa)")
    include_metadata: bool = Field(default=True, description="Include encryption metadata")


class DecryptionRequest(BaseModel):
    """Decryption request model"""
    encrypted_content: str = Field(..., description="Encrypted content to decrypt", min_length=1)
    encryption_type: str = Field(default="fernet", description="Encryption type (fernet, rsa)")


class AuthenticationRequest(BaseModel):
    """Authentication request model"""
    username: str = Field(..., description="Username", min_length=1)
    password: str = Field(..., description="Password", min_length=1)
    remember_me: bool = Field(default=False, description="Remember user session")


class TokenVerificationRequest(BaseModel):
    """Token verification request model"""
    token: str = Field(..., description="JWT token to verify", min_length=1)


class IPBlockingRequest(BaseModel):
    """IP blocking request model"""
    ip_address: str = Field(..., description="IP address to block")
    duration_seconds: int = Field(default=3600, description="Block duration in seconds")
    reason: str = Field(default="security_violation", description="Reason for blocking")


class SecurityConfigRequest(BaseModel):
    """Security configuration request model"""
    enable_threat_detection: bool = Field(default=True, description="Enable threat detection")
    enable_encryption: bool = Field(default=True, description="Enable encryption")
    enable_authentication: bool = Field(default=True, description="Enable authentication")
    enable_authorization: bool = Field(default=True, description="Enable authorization")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    enable_ip_blocking: bool = Field(default=True, description="Enable IP blocking")
    enable_content_filtering: bool = Field(default=True, description="Enable content filtering")
    enable_malware_detection: bool = Field(default=True, description="Enable malware detection")
    enable_anomaly_detection: bool = Field(default=True, description="Enable anomaly detection")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=3600, description="Rate limit window in seconds")
    max_login_attempts: int = Field(default=5, description="Maximum login attempts")
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    enable_2fa: bool = Field(default=True, description="Enable two-factor authentication")
    enable_ssl_verification: bool = Field(default=True, description="Enable SSL verification")
    enable_content_scanning: bool = Field(default=True, description="Enable content scanning")
    threat_intelligence_enabled: bool = Field(default=True, description="Enable threat intelligence")


# Dependency to get security enhancement engine
async def get_security_engine():
    """Get security enhancement engine dependency"""
    engine = await get_security_enhancement_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Security Enhancement Engine not available")
    return engine


# Security Enhancement Routes
@router.post("/analyze-security", response_model=Dict[str, Any])
async def analyze_security(
    request: SecurityAnalysisRequest,
    http_request: Request,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Perform comprehensive security analysis"""
    try:
        start_time = time.time()
        
        # Extract metadata from HTTP request
        metadata = {
            "source_ip": http_request.client.host if http_request.client else "unknown",
            "user_agent": http_request.headers.get("user-agent", "unknown"),
            "content_length": len(request.content),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add custom metadata if provided
        if request.metadata:
            metadata.update(request.metadata)
        
        # Perform security analysis
        security_events = await engine.analyze_security(request.content, metadata)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Format security events
        formatted_events = []
        threat_count = 0
        high_severity_count = 0
        
        for event in security_events:
            formatted_events.append({
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "severity": event.severity,
                "source_ip": event.source_ip,
                "user_agent": event.user_agent,
                "description": event.description,
                "details": event.details,
                "action_taken": event.action_taken,
                "resolved": event.resolved
            })
            
            threat_count += 1
            if event.severity in ["high", "critical"]:
                high_severity_count += 1
        
        # Generate security recommendations
        recommendations = []
        if request.include_recommendations:
            if high_severity_count > 0:
                recommendations.append("High severity threats detected - immediate action required")
            if threat_count > 0:
                recommendations.append("Security threats detected - review and implement security measures")
            if not threat_count:
                recommendations.append("No security threats detected - content appears safe")
        
        return {
            "success": True,
            "security_analysis": {
                "content_length": len(request.content),
                "threat_count": threat_count,
                "high_severity_count": high_severity_count,
                "security_events": formatted_events,
                "metadata": metadata
            },
            "recommendations": recommendations,
            "processing_time_ms": processing_time,
            "message": f"Security analysis completed - {threat_count} threats detected"
        }
        
    except Exception as e:
        logger.error(f"Error in security analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Security analysis failed: {str(e)}")


@router.post("/encrypt", response_model=Dict[str, Any])
async def encrypt_content(
    request: EncryptionRequest,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Encrypt content"""
    try:
        start_time = time.time()
        
        # Encrypt content
        result = await engine.encrypt_content(request.content, request.encryption_type)
        
        processing_time = (time.time() - start_time) * 1000
        
        response_data = {
            "success": True,
            "encryption_result": {
                "encrypted_data": result["encrypted_data"],
                "encryption_type": result["encryption_type"],
                "timestamp": result["timestamp"]
            },
            "processing_time_ms": processing_time,
            "message": "Content encrypted successfully"
        }
        
        if request.include_metadata:
            response_data["encryption_result"]["metadata"] = {
                "original_length": len(request.content),
                "encrypted_length": len(result["encrypted_data"]),
                "compression_ratio": len(result["encrypted_data"]) / len(request.content)
            }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error encrypting content: {e}")
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")


@router.post("/decrypt", response_model=Dict[str, Any])
async def decrypt_content(
    request: DecryptionRequest,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Decrypt content"""
    try:
        start_time = time.time()
        
        # Decrypt content
        decrypted_content = await engine.decrypt_content(request.encrypted_content, request.encryption_type)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "decryption_result": {
                "decrypted_content": decrypted_content,
                "encryption_type": request.encryption_type,
                "timestamp": datetime.now().isoformat()
            },
            "processing_time_ms": processing_time,
            "message": "Content decrypted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error decrypting content: {e}")
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(e)}")


@router.post("/authenticate", response_model=Dict[str, Any])
async def authenticate_user(
    request: AuthenticationRequest,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Authenticate user"""
    try:
        start_time = time.time()
        
        # Authenticate user
        auth_result = await engine.authenticate_user(request.username, request.password)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "authentication_result": {
                "authenticated": auth_result["authenticated"],
                "token": auth_result["token"],
                "expires_in": auth_result["expires_in"],
                "permissions": auth_result["permissions"],
                "username": request.username,
                "remember_me": request.remember_me
            },
            "processing_time_ms": processing_time,
            "message": "User authenticated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


@router.post("/verify-token", response_model=Dict[str, Any])
async def verify_token(
    request: TokenVerificationRequest,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Verify JWT token"""
    try:
        start_time = time.time()
        
        # Verify token
        token_payload = await engine.verify_token(request.token)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "token_verification": {
                "valid": True,
                "payload": token_payload,
                "timestamp": datetime.now().isoformat()
            },
            "processing_time_ms": processing_time,
            "message": "Token verified successfully"
        }
        
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")


@router.post("/block-ip", response_model=Dict[str, Any])
async def block_ip_address(
    request: IPBlockingRequest,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Block IP address"""
    try:
        start_time = time.time()
        
        # Block IP address
        await engine.block_ip(request.ip_address, request.duration_seconds, request.reason)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "ip_blocking_result": {
                "ip_address": request.ip_address,
                "duration_seconds": request.duration_seconds,
                "reason": request.reason,
                "blocked_at": datetime.now().isoformat(),
                "unblocked_at": (datetime.now() + timedelta(seconds=request.duration_seconds)).isoformat()
            },
            "processing_time_ms": processing_time,
            "message": f"IP address {request.ip_address} blocked successfully"
        }
        
    except Exception as e:
        logger.error(f"Error blocking IP address: {e}")
        raise HTTPException(status_code=500, detail=f"IP blocking failed: {str(e)}")


@router.get("/security-events", response_model=Dict[str, Any])
async def get_security_events(
    limit: int = 100,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Get security events"""
    try:
        # Get security events
        events = await engine.get_security_events(limit, event_type)
        
        # Filter by severity if specified
        if severity:
            events = [e for e in events if e.severity == severity]
        
        # Format events
        formatted_events = []
        for event in events:
            formatted_events.append({
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "severity": event.severity,
                "source_ip": event.source_ip,
                "user_agent": event.user_agent,
                "description": event.description,
                "details": event.details,
                "action_taken": event.action_taken,
                "resolved": event.resolved
            })
        
        return {
            "success": True,
            "security_events": formatted_events,
            "total_count": len(formatted_events),
            "filters": {
                "limit": limit,
                "event_type": event_type,
                "severity": severity
            },
            "message": "Security events retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting security events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get security events: {str(e)}")


@router.get("/security-metrics", response_model=Dict[str, Any])
async def get_security_metrics(
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Get security metrics"""
    try:
        # Get security metrics
        metrics = await engine.get_security_metrics()
        
        return {
            "success": True,
            "security_metrics": {
                "timestamp": metrics.timestamp.isoformat(),
                "total_events": metrics.total_events,
                "high_severity_events": metrics.high_severity_events,
                "medium_severity_events": metrics.medium_severity_events,
                "low_severity_events": metrics.low_severity_events,
                "blocked_ips": metrics.blocked_ips,
                "rate_limited_requests": metrics.rate_limited_requests,
                "failed_logins": metrics.failed_logins,
                "successful_logins": metrics.successful_logins,
                "encryption_operations": metrics.encryption_operations,
                "decryption_operations": metrics.decryption_operations,
                "threat_detections": metrics.threat_detections,
                "false_positives": metrics.false_positives
            },
            "message": "Security metrics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting security metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get security metrics: {str(e)}")


@router.get("/audit-logs", response_model=Dict[str, Any])
async def get_audit_logs(
    limit: int = 100,
    operation_type: Optional[str] = None,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Get audit logs"""
    try:
        # Get audit logs
        logs = await engine.get_audit_logs(limit)
        
        # Filter by operation type if specified
        if operation_type:
            logs = [log for log in logs if log.get("operation") == operation_type]
        
        return {
            "success": True,
            "audit_logs": logs,
            "total_count": len(logs),
            "filters": {
                "limit": limit,
                "operation_type": operation_type
            },
            "message": "Audit logs retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audit logs: {str(e)}")


@router.post("/configure", response_model=Dict[str, Any])
async def configure_security(
    request: SecurityConfigRequest,
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Configure security enhancement settings"""
    try:
        # Update configuration
        config = SecurityConfig(
            enable_threat_detection=request.enable_threat_detection,
            enable_encryption=request.enable_encryption,
            enable_authentication=request.enable_authentication,
            enable_authorization=request.enable_authorization,
            enable_audit_logging=request.enable_audit_logging,
            enable_rate_limiting=request.enable_rate_limiting,
            enable_ip_blocking=request.enable_ip_blocking,
            enable_content_filtering=request.enable_content_filtering,
            enable_malware_detection=request.enable_malware_detection,
            enable_anomaly_detection=request.enable_anomaly_detection,
            rate_limit_requests=request.rate_limit_requests,
            rate_limit_window=request.rate_limit_window,
            max_login_attempts=request.max_login_attempts,
            session_timeout=request.session_timeout,
            enable_2fa=request.enable_2fa,
            enable_ssl_verification=request.enable_ssl_verification,
            enable_content_scanning=request.enable_content_scanning,
            threat_intelligence_enabled=request.threat_intelligence_enabled
        )
        
        # Update engine configuration
        engine.config = config
        
        return {
            "success": True,
            "configuration": {
                "enable_threat_detection": config.enable_threat_detection,
                "enable_encryption": config.enable_encryption,
                "enable_authentication": config.enable_authentication,
                "enable_authorization": config.enable_authorization,
                "enable_audit_logging": config.enable_audit_logging,
                "enable_rate_limiting": config.enable_rate_limiting,
                "enable_ip_blocking": config.enable_ip_blocking,
                "enable_content_filtering": config.enable_content_filtering,
                "enable_malware_detection": config.enable_malware_detection,
                "enable_anomaly_detection": config.enable_anomaly_detection,
                "rate_limit_requests": config.rate_limit_requests,
                "rate_limit_window": config.rate_limit_window,
                "max_login_attempts": config.max_login_attempts,
                "session_timeout": config.session_timeout,
                "enable_2fa": config.enable_2fa,
                "enable_ssl_verification": config.enable_ssl_verification,
                "enable_content_scanning": config.enable_content_scanning,
                "threat_intelligence_enabled": config.threat_intelligence_enabled
            },
            "message": "Security configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error configuring security: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_security_capabilities(
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Get security enhancement capabilities"""
    try:
        capabilities = {
            "threat_detection": {
                "sql_injection": "SQL injection attack detection and prevention",
                "xss": "Cross-site scripting (XSS) attack detection",
                "path_traversal": "Path traversal attack detection",
                "command_injection": "Command injection attack detection",
                "csrf": "Cross-site request forgery detection",
                "malware_detection": "Malware signature detection",
                "anomaly_detection": "Behavioral anomaly detection"
            },
            "encryption": {
                "fernet_encryption": "Symmetric encryption using Fernet",
                "rsa_encryption": "Asymmetric encryption using RSA",
                "password_hashing": "Secure password hashing with bcrypt",
                "jwt_tokens": "JSON Web Token generation and verification"
            },
            "authentication": {
                "user_authentication": "User authentication and authorization",
                "jwt_verification": "JWT token verification and validation",
                "session_management": "Secure session management",
                "two_factor_auth": "Two-factor authentication support"
            },
            "rate_limiting": {
                "request_rate_limiting": "Request rate limiting and throttling",
                "ip_based_limiting": "IP-based rate limiting",
                "user_based_limiting": "User-based rate limiting",
                "adaptive_limiting": "Adaptive rate limiting based on behavior"
            },
            "monitoring": {
                "security_events": "Comprehensive security event logging",
                "audit_logging": "Detailed audit trail logging",
                "threat_intelligence": "Threat intelligence integration",
                "real_time_monitoring": "Real-time security monitoring"
            },
            "advanced_features": {
                "ip_blocking": "Automatic and manual IP blocking",
                "content_filtering": "Content-based security filtering",
                "ssl_verification": "SSL certificate verification",
                "compliance_monitoring": "Security compliance monitoring"
            }
        }
        
        return {
            "success": True,
            "capabilities": capabilities,
            "message": "Security capabilities retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting security capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get security capabilities: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    engine: SecurityEnhancementEngine = Depends(get_security_engine)
):
    """Security Enhancement Engine health check"""
    try:
        # Check engine components
        components_status = {
            "threat_detector": engine.threat_detector is not None,
            "encryption_manager": engine.encryption_manager is not None,
            "rate_limiter": engine.rate_limiter is not None
        }
        
        # Get current security metrics
        current_metrics = await engine.get_security_metrics()
        
        # Check system security status
        security_status = {
            "total_events": current_metrics.total_events,
            "high_severity_events": current_metrics.high_severity_events,
            "blocked_ips": current_metrics.blocked_ips,
            "threat_detections": current_metrics.threat_detections,
            "failed_logins": current_metrics.failed_logins
        }
        
        # Determine overall health
        all_healthy = all(components_status.values())
        security_healthy = current_metrics.high_severity_events < 10  # Threshold for security health
        
        overall_health = "healthy" if all_healthy and security_healthy else "degraded"
        
        return {
            "status": overall_health,
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "security_status": security_status,
            "configuration": {
                "threat_detection_enabled": engine.config.enable_threat_detection,
                "encryption_enabled": engine.config.enable_encryption,
                "authentication_enabled": engine.config.enable_authentication,
                "rate_limiting_enabled": engine.config.enable_rate_limiting,
                "ip_blocking_enabled": engine.config.enable_ip_blocking
            },
            "message": "Security Enhancement Engine is operational" if overall_health == "healthy" else "Security attention may be required"
        }
        
    except Exception as e:
        logger.error(f"Error in Security Enhancement health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Security Enhancement Engine health check failed"
        }
