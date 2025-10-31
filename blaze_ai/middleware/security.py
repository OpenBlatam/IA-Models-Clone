"""
Advanced security middleware for Blaze AI.

This module provides comprehensive security features including authentication,
authorization, input validation, threat detection, and security monitoring.
"""

import asyncio
import hashlib
import hmac
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from urllib.parse import urlparse
import jwt
from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from cryptography.fernet import Fernet
import ipaddress
from collections import defaultdict

# =============================================================================
# Types
# =============================================================================

SecurityLevel = Union[str, int]
UserID = Union[str, int]
ResourceID = Union[str, int]

# =============================================================================
# Enums
# =============================================================================

class SecurityThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuthenticationMethod(Enum):
    """Authentication methods."""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH = "oauth"
    BASIC = "basic"
    CUSTOM = "custom"

class AuthorizationLevel(Enum):
    """Authorization levels."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_input_validation: bool = True
    enable_threat_detection: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    jwt_secret_key: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # seconds
    api_key_header: str = "X-API-Key"
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_encryption: bool = False
    encryption_key: Optional[str] = None

@dataclass
class ThreatDetectionConfig:
    """Threat detection configuration."""
    enable_sql_injection_detection: bool = True
    enable_xss_detection: bool = True
    enable_path_traversal_detection: bool = True
    enable_command_injection_detection: bool = True
    enable_rate_limit_bypass_detection: bool = True
    suspicious_patterns: List[str] = field(default_factory=list)
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # seconds
    enable_ip_blacklisting: bool = True
    enable_behavioral_analysis: bool = True

# =============================================================================
# Security Exceptions
# =============================================================================

class SecurityException(Exception):
    """Base security exception."""
    
    def __init__(self, message: str, threat_level: SecurityThreatLevel = SecurityThreatLevel.MEDIUM):
        super().__init__(message)
        self.message = message
        self.threat_level = threat_level
        self.timestamp = time.time()

class AuthenticationError(SecurityException):
    """Authentication failed."""
    pass

class AuthorizationError(SecurityException):
    """Authorization failed."""
    pass

class InputValidationError(SecurityException):
    """Input validation failed."""
    pass

class ThreatDetectedError(SecurityException):
    """Security threat detected."""
    pass

# =============================================================================
# Authentication System
# =============================================================================

class Authenticator(ABC):
    """Abstract base class for authentication."""
    
    @abstractmethod
    async def authenticate(self, credentials: Any) -> Optional[Dict[str, Any]]:
        """Authenticate credentials and return user info."""
        pass
    
    @abstractmethod
    async def is_valid(self, token: str) -> bool:
        """Check if token is valid."""
        pass

class JWTAuthenticator(Authenticator):
    """JWT-based authentication."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.secret_key = config.jwt_secret_key
        self.algorithm = config.jwt_algorithm
        self.expiration = config.jwt_expiration
    
    async def authenticate(self, credentials: str) -> Optional[Dict[str, Any]]:
        """Authenticate JWT token."""
        try:
            payload = jwt.decode(credentials, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            if 'exp' in payload and payload['exp'] < time.time():
                return None
            
            return payload
        except jwt.InvalidTokenError:
            return None
    
    async def is_valid(self, token: str) -> bool:
        """Check if JWT token is valid."""
        try:
            jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return True
        except jwt.InvalidTokenError:
            return False
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token."""
        payload = {
            **user_data,
            'exp': time.time() + self.expiration,
            'iat': time.time()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

class APIKeyAuthenticator(Authenticator):
    """API key-based authentication."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from configuration or database."""
        # This would typically load from a secure storage
        self.api_keys = {
            "test-key-123": {
                "user_id": "test-user",
                "permissions": ["read", "write"],
                "expires_at": None
            }
        }
    
    async def authenticate(self, credentials: str) -> Optional[Dict[str, Any]]:
        """Authenticate API key."""
        if credentials in self.api_keys:
            key_info = self.api_keys[credentials]
            
            # Check expiration
            if key_info.get('expires_at') and key_info['expires_at'] < time.time():
                return None
            
            return key_info
        return None
    
    async def is_valid(self, token: str) -> bool:
        """Check if API key is valid."""
        return token in self.api_keys

class AuthenticationManager:
    """Manages multiple authentication methods."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.authenticators: Dict[AuthenticationMethod, Authenticator] = {}
        self._setup_authenticators()
    
    def _setup_authenticators(self):
        """Setup authentication methods."""
        if self.config.enable_authentication:
            self.authenticators[AuthenticationMethod.JWT] = JWTAuthenticator(self.config)
            self.authenticators[AuthenticationMethod.API_KEY] = APIKeyAuthenticator(self.config)
    
    async def authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate incoming request."""
        # Try JWT authentication first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            jwt_auth = self.authenticators.get(AuthenticationMethod.JWT)
            if jwt_auth:
                user_info = await jwt_auth.authenticate(token)
                if user_info:
                    return user_info
        
        # Try API key authentication
        api_key = request.headers.get(self.config.api_key_header)
        if api_key:
            api_auth = self.authenticators.get(AuthenticationMethod.API_KEY)
            if api_auth:
                user_info = await api_auth.authenticate(api_key)
                if user_info:
                    return user_info
        
        return None

# =============================================================================
# Authorization System
# =============================================================================

class AuthorizationManager:
    """Manages user authorization and permissions."""
    
    def __init__(self):
        self.permissions: Dict[UserID, Set[str]] = {}
        self.resource_permissions: Dict[ResourceID, Dict[UserID, Set[str]]] = {}
        self.role_permissions: Dict[str, Set[str]] = {}
        self._setup_default_permissions()
    
    def _setup_default_permissions(self):
        """Setup default permissions."""
        # Default role permissions
        self.role_permissions = {
            "user": {"read"},
            "moderator": {"read", "write"},
            "admin": {"read", "write", "delete"},
            "super_admin": {"read", "write", "delete", "admin"}
        }
    
    def add_user_permission(self, user_id: UserID, permission: str):
        """Add permission for a user."""
        if user_id not in self.permissions:
            self.permissions[user_id] = set()
        self.permissions[user_id].add(permission)
    
    def add_role_permission(self, role: str, permission: str):
        """Add permission for a role."""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        self.role_permissions[role].add(permission)
    
    def check_permission(self, user_id: UserID, permission: str, resource_id: Optional[ResourceID] = None) -> bool:
        """Check if user has specific permission."""
        # Check user-specific permissions
        if user_id in self.permissions and permission in self.permissions[user_id]:
            return True
        
        # Check resource-specific permissions
        if resource_id and resource_id in self.resource_permissions:
            if user_id in self.resource_permissions[resource_id] and permission in self.resource_permissions[resource_id][user_id]:
                return True
        
        # Check role-based permissions
        user_roles = self._get_user_roles(user_id)
        for role in user_roles:
            if role in self.role_permissions and permission in self.role_permissions[role]:
                return True
        
        return False
    
    def _get_user_roles(self, user_id: UserID) -> List[str]:
        """Get roles for a user."""
        # This would typically query a database
        # For now, return default role
        return ["user"]
    
    def require_permission(self, permission: str, resource_id: Optional[ResourceID] = None):
        """Decorator to require specific permission."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract user_id from request context
                request = kwargs.get('request')
                if not request:
                    raise AuthorizationError("Request context not found")
                
                user_info = getattr(request.state, 'user_info', None)
                if not user_info:
                    raise AuthorizationError("User not authenticated")
                
                user_id = user_info.get('user_id')
                if not user_id:
                    raise AuthorizationError("User ID not found")
                
                if not self.check_permission(user_id, permission, resource_id):
                    raise AuthorizationError(f"Insufficient permissions: {permission}")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# =============================================================================
# Input Validation
# =============================================================================

class InputValidator:
    """Validates and sanitizes input data."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.suspicious_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",  # XSS
            r"on\w+\s*=",  # XSS
            r"union\s+select",  # SQL injection
            r"drop\s+table",  # SQL injection
            r"exec\s*\(",  # Command injection
            r"system\s*\(",  # Command injection
            r"\.\./",  # Path traversal
            r"\.\.\\",  # Path traversal
        ]
        
        if config.enable_threat_detection:
            self.suspicious_patterns.extend(config.suspicious_patterns)
    
    def validate_string(self, value: str, max_length: int = 1000) -> str:
        """Validate and sanitize string input."""
        if not isinstance(value, str):
            raise InputValidationError("Value must be a string")
        
        if len(value) > max_length:
            raise InputValidationError(f"String too long: {len(value)} > {max_length}")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ThreatDetectedError(f"Suspicious pattern detected: {pattern}")
        
        # Basic sanitization
        sanitized = value.strip()
        sanitized = re.sub(r'<[^>]*>', '', sanitized)  # Remove HTML tags
        
        return sanitized
    
    def validate_json(self, value: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Validate and parse JSON input."""
        if len(value) > max_size:
            raise InputValidationError(f"JSON too large: {len(value)} > {max_size}")
        
        try:
            data = json.loads(value)
            if not isinstance(data, dict):
                raise InputValidationError("JSON must be an object")
            return data
        except json.JSONDecodeError as e:
            raise InputValidationError(f"Invalid JSON: {e}")
    
    def validate_url(self, value: str) -> str:
        """Validate URL input."""
        try:
            parsed = urlparse(value)
            if not parsed.scheme or not parsed.netloc:
                raise InputValidationError("Invalid URL format")
            
            # Check for suspicious schemes
            if parsed.scheme not in ['http', 'https', 'ftp']:
                raise InputValidationError(f"Unsupported URL scheme: {parsed.scheme}")
            
            return value
        except Exception as e:
            raise InputValidationError(f"URL validation failed: {e}")
    
    def validate_file_upload(self, file_size: int, file_type: str) -> bool:
        """Validate file upload."""
        if file_size > self.config.max_request_size:
            raise InputValidationError(f"File too large: {file_size} > {self.config.max_request_size}")
        
        # Check file type
        allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'text/plain', 'application/json']
        if file_type not in allowed_types:
            raise InputValidationError(f"File type not allowed: {file_type}")
        
        return True

# =============================================================================
# Threat Detection
# =============================================================================

class ThreatDetector:
    """Detects security threats and suspicious behavior."""
    
    def __init__(self, config: ThreatDetectionConfig):
        self.config = config
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.suspicious_activities: List[Dict[str, Any]] = []
        self.behavioral_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def detect_threats(self, request: Request, user_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect potential security threats."""
        threats = []
        
        # Check IP address
        client_ip = self._get_client_ip(request)
        if self._is_ip_blocked(client_ip):
            threats.append({
                "type": "blocked_ip",
                "level": SecurityThreatLevel.HIGH,
                "details": f"IP {client_ip} is blocked",
                "timestamp": time.time()
            })
        
        # Check for suspicious patterns in request
        if self.config.enable_sql_injection_detection:
            sql_threats = self._detect_sql_injection(request)
            threats.extend(sql_threats)
        
        if self.config.enable_xss_detection:
            xss_threats = self._detect_xss(request)
            threats.extend(xss_threats)
        
        if self.config.enable_path_traversal_detection:
            path_threats = self._detect_path_traversal(request)
            threats.extend(path_threats)
        
        # Behavioral analysis
        if self.config.enable_behavioral_analysis:
            behavior_threats = self._analyze_behavior(request, user_info)
            threats.extend(behavior_threats)
        
        # Record suspicious activities
        if threats:
            self._record_suspicious_activity(request, threats)
        
        return threats
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips
    
    def _detect_sql_injection(self, request: Request) -> List[Dict[str, Any]]:
        """Detect SQL injection attempts."""
        threats = []
        sql_patterns = [
            r"union\s+select", r"drop\s+table", r"delete\s+from",
            r"insert\s+into", r"update\s+set", r"alter\s+table"
        ]
        
        # Check query parameters
        for param_name, param_value in request.query_params.items():
            for pattern in sql_patterns:
                if re.search(pattern, str(param_value), re.IGNORECASE):
                    threats.append({
                        "type": "sql_injection",
                        "level": SecurityThreatLevel.HIGH,
                        "details": f"SQL injection pattern detected in {param_name}: {pattern}",
                        "timestamp": time.time()
                    })
        
        return threats
    
    def _detect_xss(self, request: Request) -> List[Dict[str, Any]]:
        """Detect XSS attempts."""
        threats = []
        xss_patterns = [
            r"<script[^>]*>", r"javascript:", r"on\w+\s*=",
            r"vbscript:", r"expression\s*\(", r"url\s*\("
        ]
        
        # Check request body
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = request.body()
                if body:
                    body_str = body.decode('utf-8')
                    for pattern in xss_patterns:
                        if re.search(pattern, body_str, re.IGNORECASE):
                            threats.append({
                                "type": "xss",
                                "level": SecurityThreatLevel.MEDIUM,
                                "details": f"XSS pattern detected: {pattern}",
                                "timestamp": time.time()
                            })
            except Exception:
                pass
        
        return threats
    
    def _detect_path_traversal(self, request: Request) -> List[Dict[str, Any]]:
        """Detect path traversal attempts."""
        threats = []
        path_patterns = [r"\.\./", r"\.\.\\", r"%2e%2e%2f", r"%2e%2e%5c"]
        
        path = request.url.path
        for pattern in path_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                threats.append({
                    "type": "path_traversal",
                    "level": SecurityThreatLevel.HIGH,
                    "details": f"Path traversal pattern detected: {pattern}",
                    "timestamp": time.time()
                })
        
        return threats
    
    def _analyze_behavior(self, request: Request, user_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Analyze request behavior for suspicious patterns."""
        threats = []
        client_ip = self._get_client_ip(request)
        
        # Track request frequency
        current_time = time.time()
        self.behavioral_patterns[client_ip].append({
            "timestamp": current_time,
            "method": request.method,
            "path": request.url.path,
            "user_agent": request.headers.get("User-Agent", "")
        })
        
        # Keep only recent requests
        cutoff_time = current_time - 300  # 5 minutes
        self.behavioral_patterns[client_ip] = [
            req for req in self.behavioral_patterns[client_ip]
            if req["timestamp"] > cutoff_time
        ]
        
        # Check for suspicious patterns
        recent_requests = self.behavioral_patterns[client_ip]
        
        # Too many requests
        if len(recent_requests) > 100:
            threats.append({
                "type": "rate_limit_bypass",
                "level": SecurityThreatLevel.MEDIUM,
                "details": f"High request frequency: {len(recent_requests)} requests in 5 minutes",
                "timestamp": current_time
            })
        
        # Suspicious user agent
        user_agents = [req["user_agent"] for req in recent_requests]
        suspicious_ua_patterns = ["bot", "crawler", "scraper", "python", "curl"]
        for pattern in suspicious_ua_patterns:
            if any(pattern.lower() in ua.lower() for ua in user_agents):
                threats.append({
                    "type": "suspicious_user_agent",
                    "level": SecurityThreatLevel.LOW,
                    "details": f"Suspicious user agent pattern: {pattern}",
                    "timestamp": current_time
                })
        
        return threats
    
    def _record_suspicious_activity(self, request: Request, threats: List[Dict[str, Any]]):
        """Record suspicious activity for analysis."""
        client_ip = self._get_client_ip(request)
        
        for threat in threats:
            self.suspicious_activities.append({
                "timestamp": threat["timestamp"],
                "ip": client_ip,
                "threat": threat,
                "request": {
                    "method": request.method,
                    "path": request.url.path,
                    "headers": dict(request.headers)
                }
            })
        
        # Keep only recent activities
        cutoff_time = time.time() - 3600  # 1 hour
        self.suspicious_activities = [
            activity for activity in self.suspicious_activities
            if activity["timestamp"] > cutoff_time
        ]
    
    def block_ip(self, ip: str, duration: int = 3600):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        
        # Auto-unblock after duration
        asyncio.create_task(self._unblock_ip_after(ip, duration))
    
    async def _unblock_ip_after(self, ip: str, duration: int):
        """Unblock IP after specified duration."""
        await asyncio.sleep(duration)
        self.blocked_ips.discard(ip)
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""
        return {
            "blocked_ips": list(self.blocked_ips),
            "suspicious_activities_count": len(self.suspicious_activities),
            "recent_threats": self.suspicious_activities[-10:] if self.suspicious_activities else [],
            "behavioral_patterns": {
                ip: len(patterns) for ip, patterns in self.behavioral_patterns.items()
            }
        }

# =============================================================================
# Security Middleware
# =============================================================================

class SecurityMiddleware:
    """Main security middleware for FastAPI."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.auth_manager = AuthenticationManager(config)
        self.auth_manager_instance = AuthorizationManager()
        self.input_validator = InputValidator(config)
        self.threat_detector = ThreatDetector(ThreatDetectionConfig())
        self.security_events: List[Dict[str, Any]] = []
    
    async def __call__(self, request: Request, call_next):
        """Process request through security middleware."""
        start_time = time.time()
        
        try:
            # Input validation
            if self.config.enable_input_validation:
                await self._validate_input(request)
            
            # Threat detection
            if self.config.enable_threat_detection:
                threats = self.threat_detector.detect_threats(request)
                if threats:
                    await self._handle_threats(request, threats)
            
            # Authentication
            user_info = None
            if self.config.enable_authentication:
                user_info = await self.auth_manager.authenticate_request(request)
                if user_info:
                    request.state.user_info = user_info
                else:
                    # Allow unauthenticated requests to public endpoints
                    if not self._is_public_endpoint(request.url.path):
                        raise AuthenticationError("Authentication required")
            
            # Authorization (if user is authenticated)
            if user_info and self.config.enable_authorization:
                # Basic authorization check - can be extended
                pass
            
            # Process request
            response = await call_next(request)
            
            # Security headers
            response = self._add_security_headers(response)
            
            # Log security event
            await self._log_security_event(request, response, "success", time.time() - start_time)
            
            return response
            
        except SecurityException as e:
            # Log security event
            await self._log_security_event(request, None, "security_error", time.time() - start_time, error=str(e))
            
            # Return appropriate error response
            if isinstance(e, AuthenticationError):
                raise HTTPException(status_code=401, detail=str(e))
            elif isinstance(e, AuthorizationError):
                raise HTTPException(status_code=403, detail=str(e))
            elif isinstance(e, InputValidationError):
                raise HTTPException(status_code=400, detail=str(e))
            elif isinstance(e, ThreatDetectedError):
                raise HTTPException(status_code=429, detail="Too many requests")
            else:
                raise HTTPException(status_code=400, detail=str(e))
        
        except Exception as e:
            # Log unexpected error
            await self._log_security_event(request, None, "unexpected_error", time.time() - start_time, error=str(e))
            raise
    
    async def _validate_input(self, request: Request):
        """Validate request input."""
        # Validate query parameters
        for param_name, param_value in request.query_params.items():
            self.input_validator.validate_string(str(param_value))
        
        # Validate request body for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = await request.body()
                    if body:
                        body_str = body.decode('utf-8')
                        self.input_validator.validate_json(body_str)
                except Exception:
                    pass
    
    async def _handle_threats(self, request: Request, threats: List[Dict[str, Any]]):
        """Handle detected security threats."""
        client_ip = self.threat_detector._get_client_ip(request)
        
        for threat in threats:
            if threat["level"] in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
                # Block IP for high-level threats
                self.threat_detector.block_ip(client_ip, 3600)  # 1 hour
                raise ThreatDetectedError(f"Security threat detected: {threat['type']}")
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no auth required)."""
        public_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/metrics"]
        return any(path.startswith(public_path) for public_path in public_paths)
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
    
    async def _log_security_event(self, request: Request, response: Optional[Response], 
                                 event_type: str, duration: float, error: Optional[str] = None):
        """Log security event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "ip": self.threat_detector._get_client_ip(request),
            "method": request.method,
            "path": request.url.path,
            "duration": duration,
            "user_agent": request.headers.get("User-Agent", ""),
            "error": error
        }
        
        if response:
            event["status_code"] = response.status_code
        
        self.security_events.append(event)
        
        # Keep only recent events
        cutoff_time = time.time() - 86400  # 24 hours
        self.security_events = [
            event for event in self.security_events
            if event["timestamp"] > cutoff_time
        ]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary."""
        return {
            "total_events": len(self.security_events),
            "recent_events": self.security_events[-10:] if self.security_events else [],
            "threat_summary": self.threat_detector.get_threat_summary(),
            "blocked_ips_count": len(self.threat_detector.blocked_ips)
        }

# =============================================================================
# Security Dependencies
# =============================================================================

def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current authenticated user."""
    user_info = getattr(request.state, 'user_info', None)
    if not user_info:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_info

def require_auth():
    """Dependency to require authentication."""
    return Depends(get_current_user)

def require_permission(permission: str):
    """Dependency to require specific permission."""
    def permission_checker(user: Dict[str, Any] = Depends(get_current_user)):
        # This would check permissions using AuthorizationManager
        return user
    return permission_checker
