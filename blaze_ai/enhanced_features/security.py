"""
Enhanced Security Module for Blaze AI.

This module provides comprehensive security features including authentication,
authorization, threat detection, and security monitoring.
"""

import asyncio
import hashlib
import hmac
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from urllib.parse import urlparse

import jwt
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

from core.config import SecurityConfig
from core.exceptions import SecurityError, AuthenticationError, AuthorizationError
from core.logging import get_logger


# ============================================================================
# SECURITY MODELS AND CONFIGURATION
# ============================================================================

class SecurityThreat(BaseModel):
    """Represents a detected security threat."""
    threat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: str = Field(..., description="Type of security threat")
    severity: str = Field(..., description="Threat severity level")
    description: str = Field(..., description="Threat description")
    ip_address: Optional[str] = Field(None, description="Source IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    blocked: bool = Field(default=False, description="Whether the threat was blocked")
    action_taken: Optional[str] = Field(None, description="Action taken against the threat")


class SecurityEvent(BaseModel):
    """Represents a security event for logging and monitoring."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., description="Type of security event")
    severity: str = Field(..., description="Event severity level")
    description: str = Field(..., description="Event description")
    user_id: Optional[str] = Field(None, description="User identifier")
    ip_address: Optional[str] = Field(None, description="Source IP address")
    endpoint: Optional[str] = Field(None, description="API endpoint")
    method: Optional[str] = Field(None, description="HTTP method")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = Field(default=True, description="Whether the security check passed")


class UserSession(BaseModel):
    """Represents a user session for authentication."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Session expiration time")
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = Field(None, description="IP address where session was created")
    user_agent: Optional[str] = Field(None, description="User agent string")
    is_active: bool = Field(default=True, description="Whether the session is active")


class SecurityMetrics(BaseModel):
    """Security metrics for monitoring and alerting."""
    total_requests: int = Field(default=0, description="Total requests processed")
    blocked_requests: int = Field(default=0, description="Requests blocked by security")
    authentication_failures: int = Field(default=0, description="Authentication failures")
    authorization_failures: int = Field(default=0, description="Authorization failures")
    threats_detected: int = Field(default=0, description="Security threats detected")
    suspicious_activities: int = Field(default=0, description="Suspicious activities detected")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# SECURITY INTERFACES AND BASE CLASSES
# ============================================================================

class SecurityProvider(ABC):
    """Abstract base class for security providers."""
    
    @abstractmethod
    async def authenticate(self, credentials: Any) -> Optional[UserSession]:
        """Authenticate user credentials."""
        pass
    
    @abstractmethod
    async def authorize(self, user: UserSession, resource: str, action: str) -> bool:
        """Authorize user access to a resource."""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[UserSession]:
        """Validate authentication token."""
        pass


class ThreatDetector(ABC):
    """Abstract base class for threat detection."""
    
    @abstractmethod
    async def detect_threats(self, request: Request) -> List[SecurityThreat]:
        """Detect security threats in the request."""
        pass
    
    @abstractmethod
    async def is_threat(self, request: Request) -> bool:
        """Check if the request contains threats."""
        pass


class SecurityValidator(ABC):
    """Abstract base class for input validation."""
    
    @abstractmethod
    async def validate_input(self, data: Any, input_type: str) -> Tuple[bool, List[str]]:
        """Validate input data for security threats."""
        pass
    
    @abstractmethod
    async def sanitize_input(self, data: Any, input_type: str) -> Any:
        """Sanitize input data."""
        pass


# ============================================================================
# IMPLEMENTATIONS
# ============================================================================

class JWTProvider(SecurityProvider):
    """JWT-based authentication provider."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.secret_key = config.jwt_secret_key
        self.algorithm = config.jwt_algorithm
        self.expiration_time = config.jwt_expiration
        
        # In-memory session storage (replace with Redis in production)
        self._sessions: Dict[str, UserSession] = {}
    
    async def authenticate(self, credentials: HTTPAuthorizationCredentials) -> Optional[UserSession]:
        """Authenticate JWT token."""
        try:
            token = credentials.credentials
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # Check if session exists and is valid
            session = self._sessions.get(user_id)
            if session and session.is_active and session.expires_at > datetime.utcnow():
                # Update last activity
                session.last_activity = datetime.utcnow()
                return session
            
            # Create new session
            session = UserSession(
                user_id=user_id,
                username=payload.get("username", "unknown"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                expires_at=datetime.utcnow() + timedelta(seconds=self.expiration_time)
            )
            
            self._sessions[user_id] = session
            self.logger.info(f"User authenticated: {user_id}")
            
            return session
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None
    
    async def authorize(self, user: UserSession, resource: str, action: str) -> bool:
        """Authorize user access based on roles and permissions."""
        try:
            # Check if user has required permissions
            required_permission = f"{resource}:{action}"
            if required_permission in user.permissions:
                return True
            
            # Check if user has admin role
            if "admin" in user.roles:
                return True
            
            # Check role-based permissions
            role_permissions = self._get_role_permissions(user.roles)
            if required_permission in role_permissions:
                return True
            
            self.logger.warning(f"Authorization failed for user {user.user_id}: {required_permission}")
            return False
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False
    
    async def validate_token(self, token: str) -> Optional[UserSession]:
        """Validate JWT token and return user session."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            
            if user_id and user_id in self._sessions:
                session = self._sessions[user_id]
                if session.is_active and session.expires_at > datetime.utcnow():
                    return session
            
            return None
            
        except jwt.InvalidTokenError:
            return None
    
    def _get_role_permissions(self, roles: List[str]) -> Set[str]:
        """Get permissions for user roles."""
        # Role-based permission mapping (replace with database lookup in production)
        role_permissions = {
            "user": {"read:own", "write:own"},
            "moderator": {"read:all", "write:own", "moderate:content"},
            "admin": {"read:all", "write:all", "delete:all", "admin:system"}
        }
        
        permissions = set()
        for role in roles:
            if role in role_permissions:
                permissions.update(role_permissions[role])
        
        return permissions


class AdvancedThreatDetector(ThreatDetector):
    """Advanced threat detection with multiple detection methods."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Threat patterns
        self.sql_injection_patterns = [
            r"(\b(select|insert|update|delete|drop|create|alter|exec|execute|union|script)\b)",
            r"(--|#|\/\*|\*\/)",
            r"(\b(and|or)\b\s+\d+\s*[=<>])",
            r"(\b(union|select)\b\s+.*\bfrom\b)",
        ]
        
        self.xss_patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(javascript:)",
            r"(on\w+\s*=)",
            r"(<iframe[^>]*>)",
            r"(<object[^>]*>)",
            r"(<embed[^>]*>)",
        ]
        
        self.path_traversal_patterns = [
            r"(\.\.\/|\.\.\\)",
            r"(\/etc\/|\/var\/|\/proc\/|\/sys\/)",
            r"(c:\\|d:\\)",
        ]
        
        self.command_injection_patterns = [
            r"(\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig|ipconfig)\b)",
            r"(\b(rm|del|mkdir|rmdir|cp|mv|chmod|chown)\b)",
            r"(\b(&&|\|\||;|`|\\$\(|\\$\\{)\b)",
        ]
        
        # IP blacklist (replace with database/Redis in production)
        self._blacklisted_ips: Set[str] = set()
        self._suspicious_ips: Dict[str, int] = {}
        
        # Rate limiting for threat detection
        self._request_counts: Dict[str, int] = {}
        self._last_reset = time.time()
    
    async def detect_threats(self, request: Request) -> List[SecurityThreat]:
        """Detect multiple types of security threats."""
        threats = []
        
        try:
            # Get request information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            path = str(request.url.path)
            method = request.method
            
            # Check IP blacklist
            if client_ip in self._blacklisted_ips:
                threats.append(SecurityThreat(
                    threat_type="blacklisted_ip",
                    severity="high",
                    description=f"Request from blacklisted IP: {client_ip}",
                    ip_address=client_ip,
                    user_agent=user_agent,
                    blocked=True,
                    action_taken="blocked"
                ))
            
            # Check for suspicious patterns
            threats.extend(await self._check_sql_injection(request, client_ip, user_agent))
            threats.extend(await self._check_xss(request, client_ip, user_agent))
            threats.extend(await self._check_path_traversal(path, client_ip, user_agent))
            threats.extend(await self._check_command_injection(request, client_ip, user_agent))
            
            # Check for suspicious behavior
            threats.extend(await self._check_suspicious_behavior(request, client_ip, user_agent))
            
            # Update metrics
            if threats:
                self._suspicious_ips[client_ip] = self._suspicious_ips.get(client_ip, 0) + 1
                
                # Blacklist IP if too many threats
                if self._suspicious_ips[client_ip] >= 10:
                    self._blacklisted_ips.add(client_ip)
                    self.logger.warning(f"IP {client_ip} added to blacklist due to multiple threats")
            
        except Exception as e:
            self.logger.error(f"Error in threat detection: {e}")
        
        return threats
    
    async def is_threat(self, request: Request) -> bool:
        """Check if the request contains any threats."""
        threats = await self.detect_threats(request)
        return len(threats) > 0
    
    async def _check_sql_injection(self, request: Request, ip: str, user_agent: str) -> List[SecurityThreat]:
        """Check for SQL injection attempts."""
        threats = []
        
        try:
            # Check query parameters
            query_params = dict(request.query_params)
            for param_name, param_value in query_params.items():
                if await self._matches_patterns(param_value, self.sql_injection_patterns):
                    threats.append(SecurityThreat(
                        threat_type="sql_injection",
                        severity="high",
                        description=f"SQL injection attempt in query parameter: {param_name}",
                        ip_address=ip,
                        user_agent=user_agent,
                        metadata={"parameter": param_name, "value": param_value}
                    ))
            
            # Check request body for JSON
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.json()
                    if await self._check_dict_for_patterns(body, self.sql_injection_patterns):
                        threats.append(SecurityThreat(
                            threat_type="sql_injection",
                            severity="high",
                            description="SQL injection attempt in request body",
                            ip_address=ip,
                            user_agent=user_agent,
                            metadata={"body": str(body)}
                        ))
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error checking SQL injection: {e}")
        
        return threats
    
    async def _check_xss(self, request: Request, ip: str, user_agent: str) -> List[SecurityThreat]:
        """Check for XSS attempts."""
        threats = []
        
        try:
            # Check query parameters
            query_params = dict(request.query_params)
            for param_name, param_value in query_params.items():
                if await self._matches_patterns(param_value, self.xss_patterns):
                    threats.append(SecurityThreat(
                        threat_type="xss",
                        severity="high",
                        description=f"XSS attempt in query parameter: {param_name}",
                        ip_address=ip,
                        user_agent=user_agent,
                        metadata={"parameter": param_name, "value": param_value}
                    ))
            
            # Check request body
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.json()
                    if await self._check_dict_for_patterns(body, self.xss_patterns):
                        threats.append(SecurityThreat(
                            threat_type="xss",
                            severity="high",
                            description="XSS attempt in request body",
                            ip_address=ip,
                            user_agent=user_agent,
                            metadata={"body": str(body)}
                        ))
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error checking XSS: {e}")
        
        return threats
    
    async def _check_path_traversal(self, path: str, ip: str, user_agent: str) -> List[SecurityThreat]:
        """Check for path traversal attempts."""
        threats = []
        
        try:
            if await self._matches_patterns(path, self.path_traversal_patterns):
                threats.append(SecurityThreat(
                    threat_type="path_traversal",
                    severity="high",
                    description="Path traversal attempt detected",
                    ip_address=ip,
                    user_agent=user_agent,
                    metadata={"path": path}
                ))
                    
        except Exception as e:
            self.logger.error(f"Error checking path traversal: {e}")
        
        return threats
    
    async def _check_command_injection(self, request: Request, ip: str, user_agent: str) -> List[SecurityThreat]:
        """Check for command injection attempts."""
        threats = []
        
        try:
            # Check query parameters
            query_params = dict(request.query_params)
            for param_name, param_value in query_params.items():
                if await self._matches_patterns(param_value, self.command_injection_patterns):
                    threats.append(SecurityThreat(
                        threat_type="command_injection",
                        severity="critical",
                        description=f"Command injection attempt in query parameter: {param_name}",
                        ip_address=ip,
                        user_agent=user_agent,
                        metadata={"parameter": param_name, "value": param_value}
                    ))
            
            # Check request body
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.json()
                    if await self._check_dict_for_patterns(body, self.command_injection_patterns):
                        threats.append(SecurityThreat(
                            threat_type="command_injection",
                            severity="critical",
                            description="Command injection attempt in request body",
                            ip_address=ip,
                            user_agent=user_agent,
                            metadata={"body": str(body)}
                        ))
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error checking command injection: {e}")
        
        return threats
    
    async def _check_suspicious_behavior(self, request: Request, ip: str, user_agent: str) -> List[SecurityThreat]:
        """Check for suspicious behavior patterns."""
        threats = []
        
        try:
            # Rate limiting check
            current_time = time.time()
            if current_time - self._last_reset > 3600:  # Reset every hour
                self._request_counts.clear()
                self._last_reset = current_time
            
            self._request_counts[ip] = self._request_counts.get(ip, 0) + 1
            
            # Check for excessive requests
            if self._request_counts[ip] > 1000:  # More than 1000 requests per hour
                threats.append(SecurityThreat(
                    threat_type="rate_limit_exceeded",
                    severity="medium",
                    description="Excessive requests from IP address",
                    ip_address=ip,
                    user_agent=user_agent,
                    metadata={"request_count": self._request_counts[ip]}
                ))
            
            # Check for suspicious user agents
            suspicious_user_agents = [
                "sqlmap", "nikto", "nmap", "w3af", "burp", "zap",
                "curl", "wget", "python-requests", "scanner"
            ]
            
            user_agent_lower = user_agent.lower()
            for suspicious_ua in suspicious_user_agents:
                if suspicious_ua in user_agent_lower:
                    threats.append(SecurityThreat(
                        threat_type="suspicious_user_agent",
                        severity="medium",
                        description=f"Suspicious user agent detected: {suspicious_ua}",
                        ip_address=ip,
                        user_agent=user_agent,
                        metadata={"suspicious_ua": suspicious_ua}
                    ))
                    break
                    
        except Exception as e:
            self.logger.error(f"Error checking suspicious behavior: {e}")
        
        return threats
    
    async def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns."""
        if not text:
            return False
        
        text_str = str(text).lower()
        for pattern in patterns:
            if re.search(pattern, text_str, re.IGNORECASE):
                return True
        
        return False
    
    async def _check_dict_for_patterns(self, data: Any, patterns: List[str]) -> bool:
        """Recursively check dictionary for patterns."""
        if isinstance(data, dict):
            for key, value in data.items():
                if await self._check_dict_for_patterns(value, patterns):
                    return True
        elif isinstance(data, list):
            for item in data:
                if await self._check_dict_for_patterns(item, patterns):
                    return True
        elif isinstance(data, str):
            return await self._matches_patterns(data, patterns)
        
        return False
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class InputSecurityValidator(SecurityValidator):
    """Input validation and sanitization for security."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Maximum sizes for different input types
        self.max_string_length = 10000
        self.max_json_depth = 10
        self.max_array_length = 1000
        
        # Allowed file extensions
        self.allowed_extensions = {".txt", ".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png", ".gif"}
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    async def validate_input(self, data: Any, input_type: str) -> Tuple[bool, List[str]]:
        """Validate input data for security threats."""
        errors = []
        
        try:
            if input_type == "string":
                errors.extend(await self._validate_string(data))
            elif input_type == "json":
                errors.extend(await self._validate_json(data))
            elif input_type == "url":
                errors.extend(await self._validate_url(data))
            elif input_type == "file":
                errors.extend(await self._validate_file(data))
            else:
                errors.append(f"Unknown input type: {input_type}")
                
        except Exception as e:
            self.logger.error(f"Error validating {input_type} input: {e}")
            errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    async def sanitize_input(self, data: Any, input_type: str) -> Any:
        """Sanitize input data."""
        try:
            if input_type == "string":
                return await self._sanitize_string(data)
            elif input_type == "json":
                return await self._sanitize_json(data)
            elif input_type == "url":
                return await self._sanitize_url(data)
            else:
                return data
                
        except Exception as e:
            self.logger.error(f"Error sanitizing {input_type} input: {e}")
            return data
    
    async def _validate_string(self, data: str) -> List[str]:
        """Validate string input."""
        errors = []
        
        if not isinstance(data, str):
            errors.append("Input must be a string")
            return errors
        
        if len(data) > self.max_string_length:
            errors.append(f"String too long (max: {self.max_string_length})")
        
        # Check for null bytes
        if "\x00" in data:
            errors.append("String contains null bytes")
        
        # Check for control characters
        if any(ord(char) < 32 for char in data):
            errors.append("String contains control characters")
        
        return errors
    
    async def _validate_json(self, data: Any) -> List[str]:
        """Validate JSON input."""
        errors = []
        
        if not isinstance(data, (dict, list)):
            errors.append("Input must be JSON object or array")
            return errors
        
        # Check depth
        depth = await self._get_json_depth(data)
        if depth > self.max_json_depth:
            errors.append(f"JSON too deep (max: {self.max_json_depth})")
        
        # Check array length
        if isinstance(data, list) and len(data) > self.max_array_length:
            errors.append(f"Array too long (max: {self.max_array_length})")
        
        return errors
    
    async def _validate_url(self, data: str) -> List[str]:
        """Validate URL input."""
        errors = []
        
        if not isinstance(data, str):
            errors.append("Input must be a string")
            return errors
        
        try:
            parsed = urlparse(data)
            if not parsed.scheme or not parsed.netloc:
                errors.append("Invalid URL format")
            
            # Check for dangerous schemes
            dangerous_schemes = {"javascript", "data", "vbscript"}
            if parsed.scheme.lower() in dangerous_schemes:
                errors.append(f"Dangerous URL scheme: {parsed.scheme}")
                
        except Exception:
            errors.append("Invalid URL format")
        
        return errors
    
    async def _validate_file(self, data: Any) -> List[str]:
        """Validate file input."""
        errors = []
        
        # This is a simplified validation - in practice, you'd validate actual file objects
        if not hasattr(data, 'filename'):
            errors.append("Input must be a file object")
            return errors
        
        filename = getattr(data, 'filename', '')
        if not filename:
            errors.append("File must have a filename")
            return errors
        
        # Check file extension
        file_ext = filename.lower()
        if not any(file_ext.endswith(ext) for ext in self.allowed_extensions):
            errors.append(f"File type not allowed. Allowed: {', '.join(self.allowed_extensions)}")
        
        # Check file size
        if hasattr(data, 'size') and data.size > self.max_file_size:
            errors.append(f"File too large (max: {self.max_file_size / (1024*1024):.1f}MB)")
        
        return errors
    
    async def _get_json_depth(self, data: Any, current_depth: int = 0) -> int:
        """Get the maximum depth of JSON data."""
        if current_depth > self.max_json_depth:
            return current_depth
        
        if isinstance(data, dict):
            max_depth = current_depth
            for value in data.values():
                depth = await self._get_json_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
            return max_depth
        elif isinstance(data, list):
            max_depth = current_depth
            for item in data:
                depth = await self._get_json_depth(item, current_depth + 1)
                max_depth = max(max_depth, depth)
            return max_depth
        else:
            return current_depth
    
    async def _sanitize_string(self, data: str) -> str:
        """Sanitize string input."""
        if not isinstance(data, str):
            return str(data)
        
        # Remove null bytes
        data = data.replace("\x00", "")
        
        # Remove control characters
        data = "".join(char for char in data if ord(char) >= 32)
        
        # Limit length
        if len(data) > self.max_string_length:
            data = data[:self.max_string_length]
        
        return data
    
    async def _sanitize_json(self, data: Any) -> Any:
        """Sanitize JSON input."""
        if isinstance(data, dict):
            return {key: await self._sanitize_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [await self._sanitize_json(item) for item in data]
        elif isinstance(data, str):
            return await self._sanitize_string(data)
        else:
            return data
    
    async def _sanitize_url(self, data: str) -> str:
        """Sanitize URL input."""
        if not isinstance(data, str):
            return str(data)
        
        # Basic URL sanitization
        data = data.strip()
        
        # Ensure URL has a scheme
        if not data.startswith(('http://', 'https://')):
            data = 'https://' + data
        
        return data


# ============================================================================
# MAIN SECURITY MIDDLEWARE
# ============================================================================

class SecurityMiddleware:
    """Main security middleware that orchestrates all security features."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize security components
        self.auth_provider = JWTProvider(config)
        self.threat_detector = AdvancedThreatDetector(config)
        self.input_validator = InputSecurityValidator(config)
        
        # Security state
        self.metrics = SecurityMetrics()
        self.threats: List[SecurityThreat] = []
        self.events: List[SecurityEvent] = []
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        self.logger.info("Security middleware initialized")
    
    async def process_request(self, request: Request) -> Tuple[bool, Optional[str], List[SecurityThreat]]:
        """
        Process incoming request for security checks.
        
        Returns:
            Tuple of (is_safe, error_message, detected_threats)
        """
        try:
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.last_updated = datetime.utcnow()
            
            # Get client information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            
            # Check for threats
            threats = await self.threat_detector.detect_threats(request)
            
            if threats:
                self.metrics.threats_detected += len(threats)
                self.threats.extend(threats)
                
                # Log security event
                event = SecurityEvent(
                    event_type="threat_detected",
                    severity="high",
                    description=f"Security threats detected: {len(threats)} threats",
                    ip_address=client_ip,
                    user_agent=user_agent,
                    endpoint=str(request.url.path),
                    method=request.method,
                    success=False,
                    metadata={"threats": [t.dict() for t in threats]}
                )
                self.events.append(event)
                
                # Block request if threats detected
                return False, "Security threats detected", threats
            
            # Check authentication if required
            if self.config.enable_authentication:
                auth_result = await self._check_authentication(request)
                if not auth_result[0]:
                    self.metrics.authentication_failures += 1
                    return False, auth_result[1], []
                
                # Check authorization if required
                if self.config.enable_authorization:
                    authz_result = await self._check_authorization(request, auth_result[2])
                    if not authz_result[0]:
                        self.metrics.authorization_failures += 1
                        return False, authz_result[1], []
            
            # Add security headers
            self._add_security_headers(request)
            
            # Log successful security check
            event = SecurityEvent(
                event_type="security_check_passed",
                severity="low",
                description="Security check passed",
                ip_address=client_ip,
                user_agent=user_agent,
                endpoint=str(request.url.path),
                method=request.method,
                success=True
            )
            self.events.append(event)
            
            return True, None, []
            
        except Exception as e:
            self.logger.error(f"Error in security processing: {e}")
            return False, f"Security processing error: {str(e)}", []
    
    async def _check_authentication(self, request: Request) -> Tuple[bool, Optional[str], Optional[UserSession]]:
        """Check request authentication."""
        try:
            auth_header = request.headers.get("authorization")
            if not auth_header:
                return False, "Authentication header required", None
            
            if not auth_header.startswith("Bearer "):
                return False, "Invalid authentication format", None
            
            token = auth_header[7:]  # Remove "Bearer " prefix
            user_session = await self.auth_provider.validate_token(token)
            
            if not user_session:
                return False, "Invalid or expired token", None
            
            return True, None, user_session
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False, f"Authentication error: {str(e)}", None
    
    async def _check_authorization(self, request: Request, user: UserSession) -> Tuple[bool, Optional[str]]:
        """Check request authorization."""
        try:
            resource = str(request.url.path)
            action = request.method.lower()
            
            is_authorized = await self.auth_provider.authorize(user, resource, action)
            
            if not is_authorized:
                return False, "Access denied"
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False, f"Authorization error: {str(e)}"
    
    def _add_security_headers(self, request: Request) -> None:
        """Add security headers to the request."""
        # This would be implemented in the actual middleware
        # For now, we just log the headers that would be added
        self.logger.debug(f"Adding security headers: {self.security_headers}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get security status and statistics."""
        return {
            "status": "active",
            "metrics": self.metrics.dict(),
            "recent_threats": len([t for t in self.threats if t.timestamp > datetime.utcnow() - timedelta(hours=1)]),
            "total_threats": len(self.threats),
            "recent_events": len([e for e in self.events if e.timestamp > datetime.utcnow() - timedelta(hours=1)]),
            "total_events": len(self.events),
            "blacklisted_ips": len(self.threat_detector._blacklisted_ips),
            "suspicious_ips": len(self.threat_detector._suspicious_ips),
            "last_updated": self.metrics.last_updated.isoformat()
        }
    
    async def cleanup(self) -> None:
        """Cleanup security resources."""
        try:
            # Clear in-memory storage
            self.threats.clear()
            self.events.clear()
            
            self.logger.info("Security middleware cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during security cleanup: {e}")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_security_middleware(config: SecurityConfig) -> SecurityMiddleware:
    """Create and configure security middleware."""
    return SecurityMiddleware(config)


def create_auth_provider(config: SecurityConfig) -> SecurityProvider:
    """Create and configure authentication provider."""
    return JWTProvider(config)


def create_threat_detector(config: SecurityConfig) -> ThreatDetector:
    """Create and configure threat detector."""
    return AdvancedThreatDetector(config)


def create_input_validator(config: SecurityConfig) -> SecurityValidator:
    """Create and configure input validator."""
    return InputSecurityValidator(config)
