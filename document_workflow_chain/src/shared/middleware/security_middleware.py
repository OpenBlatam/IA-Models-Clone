"""
Security Middleware
===================

Advanced security middleware for authentication, authorization, and security monitoring.
"""

from __future__ import annotations
import logging
import time
from typing import Optional, Dict, Any, List
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..services.security_service import (
    verify_token,
    validate_session,
    record_failed_login,
    is_account_locked,
    get_security_events,
    get_security_stats
)
from ..services.audit_service import (
    log_authentication_event,
    log_security_event,
    AuditEventType,
    AuditStatus
)


logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Advanced security middleware"""
    
    def __init__(
        self,
        app: ASGIApp,
        enable_authentication: bool = True,
        enable_authorization: bool = True,
        enable_security_monitoring: bool = True,
        enable_rate_limiting: bool = True,
        enable_ip_filtering: bool = True,
        blocked_ips: Optional[List[str]] = None,
        allowed_ips: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.enable_authentication = enable_authentication
        self.enable_authorization = enable_authorization
        self.enable_security_monitoring = enable_security_monitoring
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_ip_filtering = enable_ip_filtering
        self.blocked_ips = blocked_ips or []
        self.allowed_ips = allowed_ips or []
        self.security_events: List[Dict[str, Any]] = []
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with security checks"""
        start_time = time.time()
        
        # Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        method = request.method
        url_path = request.url.path
        
        # IP filtering
        if self.enable_ip_filtering:
            if not self._is_ip_allowed(client_ip):
                await self._log_security_violation(
                    request, "IP_BLOCKED", f"IP {client_ip} is blocked", AuditStatus.FAILURE
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        # Security monitoring
        if self.enable_security_monitoring:
            await self._monitor_security(request, client_ip, user_agent)
        
        # Authentication
        user_id = None
        session_id = None
        if self.enable_authentication:
            user_id, session_id = await self._authenticate_request(request, client_ip, user_agent)
        
        # Authorization
        if self.enable_authorization and user_id:
            if not await self._authorize_request(request, user_id, client_ip, user_agent):
                await self._log_security_violation(
                    request, "AUTHORIZATION_FAILED", f"User {user_id} not authorized", AuditStatus.FAILURE
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        # Rate limiting
        if self.enable_rate_limiting:
            if not await self._check_rate_limit(request, client_ip, user_agent):
                await self._log_security_violation(
                    request, "RATE_LIMIT_EXCEEDED", f"Rate limit exceeded for {client_ip}", AuditStatus.FAILURE
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
        
        # Add security context to request
        request.state.security_context = {
            "user_id": user_id,
            "session_id": session_id,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "authenticated": user_id is not None,
            "authorized": user_id is not None
        }
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log successful request
            if self.enable_security_monitoring:
                await self._log_successful_request(request, response, user_id, client_ip, user_agent)
            
            return response
        
        except Exception as e:
            # Log failed request
            if self.enable_security_monitoring:
                await self._log_failed_request(request, e, user_id, client_ip, user_agent)
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if IP is allowed"""
        # Check blocked IPs
        if client_ip in self.blocked_ips:
            return False
        
        # Check allowed IPs (if specified)
        if self.allowed_ips and client_ip not in self.allowed_ips:
            return False
        
        return True
    
    async def _authenticate_request(self, request: Request, client_ip: str, user_agent: str) -> tuple[Optional[str], Optional[str]]:
        """Authenticate request"""
        # Check for JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = verify_token(token)
            if payload:
                user_id = payload.get("user_id")
                if user_id:
                    # Log successful authentication
                    log_authentication_event(
                        AuditEventType.LOGIN_SUCCESS,
                        user_id,
                        client_ip,
                        user_agent,
                        AuditStatus.SUCCESS,
                        {"method": "jwt_token"}
                    )
                    return user_id, None
        
        # Check for session
        session_id = request.cookies.get("session_id")
        if session_id:
            if validate_session(session_id, client_ip, user_agent):
                # Get user ID from session (simplified)
                user_id = "session_user"  # In real implementation, get from session
                return user_id, session_id
        
        # Check for API key
        api_key = request.headers.get("x-api-key")
        if api_key:
            # Verify API key (simplified)
            if len(api_key) >= 32:
                user_id = "api_user"  # In real implementation, get from API key
                return user_id, None
        
        return None, None
    
    async def _authorize_request(self, request: Request, user_id: str, client_ip: str, user_agent: str) -> bool:
        """Authorize request"""
        # Check if user is locked
        if is_account_locked(user_id):
            log_security_event(
                AuditEventType.UNAUTHORIZED_ACCESS,
                user_id,
                client_ip,
                user_agent,
                AuditStatus.FAILURE,
                {"reason": "account_locked"}
            )
            return False
        
        # Check user permissions (simplified)
        # In a real implementation, you would check user roles and permissions
        method = request.method
        url_path = request.url.path
        
        # Admin endpoints require admin role
        if url_path.startswith("/admin"):
            # Check if user has admin role
            if user_id != "admin":
                return False
        
        # API endpoints require API access
        if url_path.startswith("/api"):
            # Check if user has API access
            if user_id not in ["api_user", "admin"]:
                return False
        
        return True
    
    async def _check_rate_limit(self, request: Request, client_ip: str, user_agent: str) -> bool:
        """Check rate limit"""
        # Simplified rate limiting
        # In a real implementation, you would use the rate limiter service
        
        # Check for rate limit headers
        rate_limit_header = request.headers.get("x-rate-limit")
        if rate_limit_header:
            try:
                rate_limit = int(rate_limit_header)
                if rate_limit <= 0:
                    return False
            except ValueError:
                pass
        
        return True
    
    async def _monitor_security(self, request: Request, client_ip: str, user_agent: str):
        """Monitor security aspects of the request"""
        # Check for suspicious patterns
        suspicious_patterns = [
            "sql injection",
            "xss",
            "script",
            "javascript:",
            "eval(",
            "exec(",
            "system(",
            "cmd",
            "powershell"
        ]
        
        # Check URL path
        url_path = request.url.path.lower()
        for pattern in suspicious_patterns:
            if pattern in url_path:
                await self._log_security_violation(
                    request, "SUSPICIOUS_PATTERN", f"Suspicious pattern detected: {pattern}", AuditStatus.FAILURE
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Suspicious request detected"
                )
        
        # Check query parameters
        query_params = str(request.query_params).lower()
        for pattern in suspicious_patterns:
            if pattern in query_params:
                await self._log_security_violation(
                    request, "SUSPICIOUS_QUERY", f"Suspicious query parameter: {pattern}", AuditStatus.FAILURE
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Suspicious query detected"
                )
        
        # Check user agent
        if user_agent.lower() in ["curl", "wget", "python-requests"]:
            # Log but don't block
            await self._log_security_violation(
                request, "SUSPICIOUS_USER_AGENT", f"Suspicious user agent: {user_agent}", AuditStatus.SUCCESS
            )
    
    async def _log_security_violation(
        self, request: Request, violation_type: str, message: str, audit_status: AuditStatus
    ):
        """Log security violation"""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log to security events
        self.security_events.append({
            "timestamp": time.time(),
            "violation_type": violation_type,
            "message": message,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "url_path": request.url.path,
            "method": request.method,
            "status": audit_status.value
        })
        
        # Log to audit service
        log_security_event(
            AuditEventType.SECURITY_VIOLATION,
            None,
            client_ip,
            user_agent,
            audit_status,
            {
                "violation_type": violation_type,
                "message": message,
                "url_path": request.url.path,
                "method": request.method
            }
        )
        
        logger.warning(f"Security violation: {violation_type} - {message} - IP: {client_ip}")
    
    async def _log_successful_request(
        self, request: Request, response: Response, user_id: Optional[str], client_ip: str, user_agent: str
    ):
        """Log successful request"""
        # Log to audit service
        log_authentication_event(
            AuditEventType.LOGIN_SUCCESS,
            user_id,
            client_ip,
            user_agent,
            AuditStatus.SUCCESS,
            {
                "url_path": request.url.path,
                "method": request.method,
                "status_code": response.status_code
            }
        )
    
    async def _log_failed_request(
        self, request: Request, error: Exception, user_id: Optional[str], client_ip: str, user_agent: str
    ):
        """Log failed request"""
        # Log to audit service
        log_authentication_event(
            AuditEventType.LOGIN_FAILED,
            user_id,
            client_ip,
            user_agent,
            AuditStatus.FAILURE,
            {
                "url_path": request.url.path,
                "method": request.method,
                "error": str(error)
            }
        )
    
    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get security events"""
        return self.security_events[-limit:]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        total_events = len(self.security_events)
        violations = len([e for e in self.security_events if e["status"] == "failure"])
        
        return {
            "total_events": total_events,
            "violations": violations,
            "success_rate": (total_events - violations) / total_events if total_events > 0 else 0,
            "timestamp": time.time()
        }


class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication"""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        credentials = await super().__call__(request)
        if credentials:
            if not self.verify_jwt(credentials.credentials):
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid authentication credentials"
                    )
                return None
        return credentials
    
    def verify_jwt(self, token: str) -> bool:
        """Verify JWT token"""
        payload = verify_token(token)
        return payload is not None


class APIKeyBearer(HTTPBearer):
    """API Key Bearer authentication"""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        # Check for API key in header
        api_key = request.headers.get("x-api-key")
        if api_key:
            if self.verify_api_key(api_key):
                return HTTPAuthorizationCredentials(scheme="ApiKey", credentials=api_key)
        
        # Fall back to Bearer token
        credentials = await super().__call__(request)
        if credentials:
            if not self.verify_api_key(credentials.credentials):
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid API key"
                    )
                return None
        return credentials
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""
        # Simplified verification
        return len(api_key) >= 32


# Dependency functions
async def get_current_user(request: Request) -> Optional[str]:
    """Get current user from request"""
    if hasattr(request.state, 'security_context'):
        return request.state.security_context.get("user_id")
    return None


async def get_current_user_optional(request: Request) -> Optional[str]:
    """Get current user from request (optional)"""
    return await get_current_user(request)


async def require_authentication(request: Request) -> str:
    """Require authentication"""
    user_id = await get_current_user(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user_id


async def require_admin(request: Request) -> str:
    """Require admin authentication"""
    user_id = await require_authentication(request)
    if user_id != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user_id


async def require_api_access(request: Request) -> str:
    """Require API access"""
    user_id = await require_authentication(request)
    if user_id not in ["api_user", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API access required"
        )
    return user_id




