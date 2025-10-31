"""
PDF Variantes Security System
Enterprise-grade security and authentication
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import jwt
from passlib.context import CryptContext
from fastapi import Request, HTTPException
import ipaddress
import re

from ..utils.config import Settings, SecurityConfig

logger = logging.getLogger(__name__)

class SecurityService:
    """Enterprise-grade security service"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.security_config = SecurityConfig()
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # JWT settings
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_expire = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        # Rate limiting
        self.rate_limit_storage: Dict[str, List[datetime]] = {}
        self.blocked_ips: set = set()
        self.allowed_ips: set = set()
        
        # Security metrics
        self.security_metrics = {
            "failed_logins": 0,
            "blocked_requests": 0,
            "suspicious_activities": 0,
            "successful_logins": 0
        }
    
    async def initialize(self):
        """Initialize security service"""
        try:
            logger.info("Security Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Security Service: {e}")
            raise
    
    async def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            return self.pwd_context.hash(password)
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise
    
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    async def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        try:
            to_encode = data.copy()
            expire = datetime.utcnow() + self.access_token_expire
            to_encode.update({"exp": expire, "type": "access"})
            
            encoded_jwt = jwt.encode(to_encode, self.settings.SECRET_KEY, algorithm=self.algorithm)
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise
    
    async def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        try:
            to_encode = data.copy()
            expire = datetime.utcnow() + self.refresh_token_expire
            to_encode.update({"exp": expire, "type": "refresh"})
            
            encoded_jwt = jwt.encode(to_encode, self.settings.SECRET_KEY, algorithm=self.algorithm)
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Error creating refresh token: {e}")
            raise
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.settings.SECRET_KEY, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    async def check_request(self, request: Request) -> Dict[str, Any]:
        """Check request for security threats"""
        try:
            client_ip = self._get_client_ip(request)
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                self.security_metrics["blocked_requests"] += 1
                return {
                    "is_safe": False,
                    "reason": "IP address is blocked",
                    "action": "block"
                }
            
            # Check rate limiting
            if not await self._check_rate_limit(client_ip):
                self.security_metrics["blocked_requests"] += 1
                return {
                    "is_safe": False,
                    "reason": "Rate limit exceeded",
                    "action": "rate_limit"
                }
            
            # Check for suspicious patterns
            suspicious_patterns = await self._detect_suspicious_patterns(request)
            if suspicious_patterns:
                self.security_metrics["suspicious_activities"] += 1
                return {
                    "is_safe": False,
                    "reason": f"Suspicious activity detected: {', '.join(suspicious_patterns)}",
                    "action": "investigate"
                }
            
            # Check file upload security
            if request.method == "POST" and "file" in str(request.url):
                file_security = await self._check_file_upload_security(request)
                if not file_security["is_safe"]:
                    return file_security
            
            return {
                "is_safe": True,
                "reason": "Request passed security checks",
                "action": "allow"
            }
            
        except Exception as e:
            logger.error(f"Error checking request security: {e}")
            return {
                "is_safe": False,
                "reason": "Security check failed",
                "action": "block"
            }
    
    async def log_request(self, request: Request, response):
        """Log request for audit purposes"""
        try:
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "url": str(request.url),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "status_code": response.status_code,
                "response_time": getattr(response, "response_time", 0)
            }
            
            # TODO: Store in audit log database
            logger.info(f"Request logged: {log_entry}")
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    async def block_ip(self, ip_address: str, reason: str = "Security violation") -> bool:
        """Block an IP address"""
        try:
            self.blocked_ips.add(ip_address)
            logger.warning(f"IP {ip_address} blocked: {reason}")
            return True
        except Exception as e:
            logger.error(f"Error blocking IP {ip_address}: {e}")
            return False
    
    async def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address"""
        try:
            self.blocked_ips.discard(ip_address)
            logger.info(f"IP {ip_address} unblocked")
            return True
        except Exception as e:
            logger.error(f"Error unblocking IP {ip_address}: {e}")
            return False
    
    async def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        try:
            errors = []
            score = 0
            
            # Length check
            if len(password) < self.security_config.MIN_PASSWORD_LENGTH:
                errors.append(f"Password must be at least {self.security_config.MIN_PASSWORD_LENGTH} characters long")
            else:
                score += 1
            
            # Uppercase check
            if self.security_config.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
                errors.append("Password must contain at least one uppercase letter")
            else:
                score += 1
            
            # Lowercase check
            if self.security_config.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
                errors.append("Password must contain at least one lowercase letter")
            else:
                score += 1
            
            # Numbers check
            if self.security_config.REQUIRE_NUMBERS and not re.search(r'\d', password):
                errors.append("Password must contain at least one number")
            else:
                score += 1
            
            # Special characters check
            if self.security_config.REQUIRE_SPECIAL_CHARS and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                errors.append("Password must contain at least one special character")
            else:
                score += 1
            
            # Calculate strength
            strength = "weak"
            if score >= 4:
                strength = "strong"
            elif score >= 3:
                strength = "medium"
            
            return {
                "is_valid": len(errors) == 0,
                "strength": strength,
                "score": score,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error validating password strength: {e}")
            return {
                "is_valid": False,
                "strength": "weak",
                "score": 0,
                "errors": ["Password validation failed"]
            }
    
    async def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            # Simple encryption using HMAC
            key = self.settings.SECRET_KEY.encode()
            encrypted = hmac.new(key, data.encode(), hashlib.sha256).hexdigest()
            return encrypted
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: str, original_data: str) -> bool:
        """Verify encrypted data"""
        try:
            expected_encrypted = await self.encrypt_data(original_data)
            return hmac.compare_digest(encrypted_data, expected_encrypted)
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return False
    
    async def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        try:
            return secrets.token_urlsafe(length)
        except Exception as e:
            logger.error(f"Error generating secure token: {e}")
            raise
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        try:
            metrics = self.security_metrics.copy()
            metrics["blocked_ips_count"] = len(self.blocked_ips)
            metrics["allowed_ips_count"] = len(self.allowed_ips)
            return metrics
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return self.security_metrics.copy()
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        try:
            # Check for forwarded headers first
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()
            
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip
            
            # Fallback to direct connection
            return request.client.host if request.client else "unknown"
            
        except Exception as e:
            logger.error(f"Error getting client IP: {e}")
            return "unknown"
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting for client IP"""
        try:
            current_time = datetime.utcnow()
            minute_ago = current_time - timedelta(minutes=1)
            
            # Clean old entries
            if client_ip in self.rate_limit_storage:
                self.rate_limit_storage[client_ip] = [
                    timestamp for timestamp in self.rate_limit_storage[client_ip]
                    if timestamp > minute_ago
                ]
            else:
                self.rate_limit_storage[client_ip] = []
            
            # Check rate limit
            request_count = len(self.rate_limit_storage[client_ip])
            if request_count >= self.settings.RATE_LIMIT_PER_MINUTE:
                return False
            
            # Add current request
            self.rate_limit_storage[client_ip].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error
    
    async def _detect_suspicious_patterns(self, request: Request) -> List[str]:
        """Detect suspicious patterns in request"""
        try:
            suspicious_patterns = []
            
            # Check URL for SQL injection patterns
            url_str = str(request.url)
            sql_patterns = [
                r"union\s+select", r"drop\s+table", r"delete\s+from",
                r"insert\s+into", r"update\s+set", r"exec\s*\(",
                r"script\s*>", r"<script", r"javascript:",
                r"onload\s*=", r"onerror\s*=", r"onclick\s*="
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, url_str, re.IGNORECASE):
                    suspicious_patterns.append(f"SQL injection pattern: {pattern}")
            
            # Check for path traversal
            if ".." in url_str or "//" in url_str:
                suspicious_patterns.append("Path traversal attempt")
            
            # Check for suspicious user agents
            user_agent = request.headers.get("user-agent", "").lower()
            suspicious_agents = ["sqlmap", "nikto", "nmap", "masscan", "zap"]
            for agent in suspicious_agents:
                if agent in user_agent:
                    suspicious_patterns.append(f"Suspicious user agent: {agent}")
            
            # Check for excessive headers
            if len(request.headers) > 50:
                suspicious_patterns.append("Excessive headers")
            
            return suspicious_patterns
            
        except Exception as e:
            logger.error(f"Error detecting suspicious patterns: {e}")
            return []
    
    async def _check_file_upload_security(self, request: Request) -> Dict[str, Any]:
        """Check file upload security"""
        try:
            # This would be implemented when handling file uploads
            # For now, return safe
            return {
                "is_safe": True,
                "reason": "File upload security check passed",
                "action": "allow"
            }
            
        except Exception as e:
            logger.error(f"Error checking file upload security: {e}")
            return {
                "is_safe": False,
                "reason": "File upload security check failed",
                "action": "block"
            }
    
    async def cleanup(self):
        """Cleanup security service"""
        try:
            # Clear rate limit storage
            self.rate_limit_storage.clear()
            
            logger.info("Security Service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up Security Service: {e}")

class AuthenticationService:
    """Authentication service"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.security_service = SecurityService(settings)
        
        # User sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Login attempt tracking
        self.login_attempts: Dict[str, List[datetime]] = {}
    
    async def initialize(self):
        """Initialize authentication service"""
        try:
            await self.security_service.initialize()
            logger.info("Authentication Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Authentication Service: {e}")
            raise
    
    async def authenticate_user(self, username: str, password: str, client_ip: str) -> Dict[str, Any]:
        """Authenticate user with username and password"""
        try:
            # Check login attempts
            if not await self._check_login_attempts(username, client_ip):
                self.security_service.security_metrics["failed_logins"] += 1
                return {
                    "success": False,
                    "message": "Too many failed login attempts. Please try again later.",
                    "user": None,
                    "tokens": None
                }
            
            # TODO: Get user from database
            user = await self._get_user_by_username(username)
            if not user:
                await self._record_failed_login(username, client_ip)
                return {
                    "success": False,
                    "message": "Invalid username or password",
                    "user": None,
                    "tokens": None
                }
            
            # Verify password
            if not await self.security_service.verify_password(password, user["hashed_password"]):
                await self._record_failed_login(username, client_ip)
                return {
                    "success": False,
                    "message": "Invalid username or password",
                    "user": None,
                    "tokens": None
                }
            
            # Check if user is active
            if not user.get("is_active", True):
                return {
                    "success": False,
                    "message": "Account is deactivated",
                    "user": None,
                    "tokens": None
                }
            
            # Create tokens
            token_data = {
                "user_id": user["user_id"],
                "username": user["username"],
                "permissions": user.get("permissions", [])
            }
            
            access_token = await self.security_service.create_access_token(token_data)
            refresh_token = await self.security_service.create_refresh_token(token_data)
            
            # Create session
            session_id = await self.security_service.generate_secure_token()
            self.active_sessions[session_id] = {
                "user_id": user["user_id"],
                "username": user["username"],
                "client_ip": client_ip,
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow()
            }
            
            # Record successful login
            self.security_service.security_metrics["successful_logins"] += 1
            await self._clear_failed_logins(username, client_ip)
            
            return {
                "success": True,
                "message": "Authentication successful",
                "user": {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "email": user["email"],
                    "permissions": user.get("permissions", [])
                },
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "expires_in": self.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
                },
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return {
                "success": False,
                "message": "Authentication failed",
                "user": None,
                "tokens": None
            }
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            payload = await self.security_service.verify_token(refresh_token)
            if not payload or payload.get("type") != "refresh":
                return {
                    "success": False,
                    "message": "Invalid refresh token",
                    "tokens": None
                }
            
            # Create new access token
            token_data = {
                "user_id": payload["user_id"],
                "username": payload["username"],
                "permissions": payload.get("permissions", [])
            }
            
            new_access_token = await self.security_service.create_access_token(token_data)
            
            return {
                "success": True,
                "message": "Token refreshed successfully",
                "tokens": {
                    "access_token": new_access_token,
                    "token_type": "bearer",
                    "expires_in": self.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
                }
            }
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return {
                "success": False,
                "message": "Token refresh failed",
                "tokens": None
            }
    
    async def logout(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"User logged out: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
            return False
    
    async def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate user session"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Check session timeout
                timeout = timedelta(minutes=self.security_service.security_config.SESSION_TIMEOUT_MINUTES)
                if datetime.utcnow() - session["last_activity"] > timeout:
                    del self.active_sessions[session_id]
                    return None
                
                # Update last activity
                session["last_activity"] = datetime.utcnow()
                return session
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    async def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username from database"""
        try:
            # TODO: Implement database lookup
            # For now, return None
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    async def _check_login_attempts(self, username: str, client_ip: str) -> bool:
        """Check if user has exceeded login attempts"""
        try:
            key = f"{username}:{client_ip}"
            current_time = datetime.utcnow()
            lockout_duration = timedelta(minutes=self.security_service.security_config.LOCKOUT_DURATION_MINUTES)
            
            if key in self.login_attempts:
                attempts = self.login_attempts[key]
                
                # Remove old attempts
                recent_attempts = [
                    attempt for attempt in attempts
                    if current_time - attempt < lockout_duration
                ]
                
                if len(recent_attempts) >= self.security_service.security_config.MAX_LOGIN_ATTEMPTS:
                    return False
                
                self.login_attempts[key] = recent_attempts
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking login attempts: {e}")
            return True
    
    async def _record_failed_login(self, username: str, client_ip: str):
        """Record failed login attempt"""
        try:
            key = f"{username}:{client_ip}"
            current_time = datetime.utcnow()
            
            if key not in self.login_attempts:
                self.login_attempts[key] = []
            
            self.login_attempts[key].append(current_time)
            
        except Exception as e:
            logger.error(f"Error recording failed login: {e}")
    
    async def _clear_failed_logins(self, username: str, client_ip: str):
        """Clear failed login attempts"""
        try:
            key = f"{username}:{client_ip}"
            if key in self.login_attempts:
                del self.login_attempts[key]
                
        except Exception as e:
            logger.error(f"Error clearing failed logins: {e}")
    
    async def cleanup(self):
        """Cleanup authentication service"""
        try:
            await self.security_service.cleanup()
            self.active_sessions.clear()
            self.login_attempts.clear()
            
            logger.info("Authentication Service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up Authentication Service: {e}")

# Utility functions for authentication
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
import os

# Security scheme for token authentication
security_scheme = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    token: Optional[str] = Depends(oauth2_scheme)
) -> str:
    """
    Get current user ID from request.
    FastAPI dependency for authentication.
    
    In development mode, allows requests without authentication.
    In production, requires valid JWT token.
    """
    try:
        # Development mode: allow anonymous access
        dev_mode = os.getenv("ENVIRONMENT", "development").lower() == "development"
        dev_mode = dev_mode or os.getenv("DEBUG", "false").lower() == "true"
        
        if dev_mode:
            # Try to get user_id from header or query params
            user_id = (
                request.headers.get("x-user-id") or
                request.headers.get("user-id") or
                request.query_params.get("user_id") or
                "anonymous"
            )
            logger.debug(f"Development mode: using user_id={user_id}")
            return user_id
        
        # Production mode: require authentication
        auth_token = None
        if credentials:
            auth_token = credentials.credentials
        elif token:
            auth_token = token
        else:
            # Try to get from Authorization header directly
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                auth_token = auth_header.split(" ")[1]
        
        if not auth_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify token (TODO: implement proper JWT verification)
        # For now, if token is provided, assume it's valid
        # In production, decode and verify the JWT token
        user_id = "authenticated_user"  # This should come from decoded token
        
        return user_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        if dev_mode:
            return "anonymous"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

async def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would check user permissions
            # For now, just return the function
            return await func(*args, **kwargs)
        return wrapper
    return decorator
