"""
AI Integration System - Middleware
Authentication, security, and request processing middleware
"""

import time
import logging
import hashlib
import hmac
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import jwt
from jwt.exceptions import InvalidTokenError
import redis
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db_session
from .models import UserSession, IntegrationLog

logger = logging.getLogger(__name__)

# Security schemes
security = HTTPBearer()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 100, burst_size: int = 20):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis client for rate limiting"""
        try:
            import redis
            from urllib.parse import urlparse
            
            parsed = urlparse(settings.redis.url)
            self.redis_client = redis.Redis(
                host=parsed.hostname or 'localhost',
                port=parsed.port or 6379,
                db=int(parsed.path.lstrip('/')) if parsed.path else 0,
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting: {str(e)}")
            self.redis_client = None
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        if not self.redis_client:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        # Try to get user ID from token
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, settings.security.secret_key, algorithms=[settings.security.algorithm])
                return f"user:{payload.get('user_id', 'anonymous')}"
        except:
            pass
        
        # Fallback to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host
        
        return f"ip:{client_ip}"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        try:
            current_time = int(time.time())
            minute_key = f"rate_limit:{client_id}:{current_time // 60}"
            
            # Get current count
            current_count = self.redis_client.get(minute_key)
            if current_count is None:
                current_count = 0
            else:
                current_count = int(current_count)
            
            # Check if limit exceeded
            if current_count >= self.requests_per_minute:
                return False
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)  # Expire after 1 minute
            pipe.execute()
            
            return True
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            return True  # Allow request if rate limiting fails

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication"""
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Check for authentication token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication required", "message": "Bearer token missing"}
            )
        
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, settings.security.secret_key, algorithms=[settings.security.algorithm])
            
            # Validate session
            if not await self._validate_session(payload.get("session_id")):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid session", "message": "Session expired or invalid"}
                )
            
            # Add user info to request state
            request.state.user_id = payload.get("user_id")
            request.state.session_id = payload.get("session_id")
            
        except InvalidTokenError:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid token", "message": "Token is invalid or expired"}
            )
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication failed", "message": "Token validation failed"}
            )
        
        return await call_next(request)
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public"""
        public_paths = [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/ai-integration/health",
            "/metrics"
        ]
        return any(path.startswith(public_path) for public_path in public_paths)
    
    async def _validate_session(self, session_id: str) -> bool:
        """Validate user session"""
        if not session_id:
            return False
        
        try:
            with get_db_session() as session:
                user_session = session.query(UserSession).filter(
                    UserSession.session_token == session_id,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                ).first()
                
                if user_session:
                    # Update last accessed time
                    user_session.last_accessed = datetime.utcnow()
                    session.commit()
                    return True
                
                return False
        except Exception as e:
            logger.error(f"Session validation error: {str(e)}")
            return False

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response"""
        start_time = time.time()
        
        # Log request
        logger.info(f"📥 {request.method} {request.url.path} - {request.client.host}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"📤 {response.status_code} - {process_time:.3f}s")
        
        # Log to database if authenticated
        if hasattr(request.state, 'user_id'):
            await self._log_to_database(request, response, process_time)
        
        return response
    
    async def _log_to_database(self, request: Request, response: Response, process_time: float):
        """Log request to database"""
        try:
            with get_db_session() as session:
                log_entry = IntegrationLog(
                    platform="api",
                    action=f"{request.method.lower()}_{request.url.path.replace('/', '_')}",
                    status="success" if response.status_code < 400 else "error",
                    message=f"{request.method} {request.url.path} - {response.status_code}",
                    details={
                        "user_id": getattr(request.state, 'user_id', None),
                        "session_id": getattr(request.state, 'session_id', None),
                        "client_ip": request.client.host,
                        "user_agent": request.headers.get("User-Agent"),
                        "process_time": process_time,
                        "status_code": response.status_code,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                session.add(log_entry)
                session.commit()
        except Exception as e:
            logger.error(f"Database logging failed: {str(e)}")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response"""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class WebhookSignatureMiddleware(BaseHTTPMiddleware):
    """Webhook signature verification middleware"""
    
    def __init__(self, app, webhook_secrets: Dict[str, str]):
        super().__init__(app)
        self.webhook_secrets = webhook_secrets
    
    async def dispatch(self, request: Request, call_next):
        """Verify webhook signatures"""
        if not request.url.path.startswith("/ai-integration/webhooks/"):
            return await call_next(request)
        
        # Extract platform from path
        platform = request.url.path.split("/")[-1]
        
        if platform not in self.webhook_secrets:
            logger.warning(f"No webhook secret configured for platform: {platform}")
            return await call_next(request)
        
        # Verify signature
        signature = request.headers.get("X-Hub-Signature-256") or request.headers.get("X-Signature")
        if not signature:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing signature", "message": "Webhook signature required"}
            )
        
        if not self._verify_signature(request, signature, self.webhook_secrets[platform]):
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid signature", "message": "Webhook signature verification failed"}
            )
        
        return await call_next(request)
    
    def _verify_signature(self, request: Request, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        try:
            # Get request body
            body = request.body()
            
            # Calculate expected signature
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                body,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            if signature.startswith("sha256="):
                signature = signature[7:]
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Signature verification error: {str(e)}")
            return False

# Authentication utilities
class AuthManager:
    """Authentication manager"""
    
    @staticmethod
    def create_access_token(user_id: str, session_id: str) -> str:
        """Create JWT access token"""
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "exp": datetime.utcnow() + timedelta(minutes=settings.security.access_token_expire_minutes),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, settings.security.secret_key, algorithm=settings.security.algorithm)
    
    @staticmethod
    def create_user_session(user_id: str) -> str:
        """Create user session"""
        session_token = hashlib.sha256(f"{user_id}{datetime.utcnow()}{settings.security.secret_key}".encode()).hexdigest()
        
        with get_db_session() as session:
            user_session = UserSession(
                user_id=user_id,
                session_token=session_token,
                expires_at=datetime.utcnow() + timedelta(hours=24),
                is_active=True
            )
            session.add(user_session)
            session.commit()
            
            return session_token
    
    @staticmethod
    def revoke_user_session(session_token: str) -> bool:
        """Revoke user session"""
        try:
            with get_db_session() as session:
                user_session = session.query(UserSession).filter(
                    UserSession.session_token == session_token
                ).first()
                
                if user_session:
                    user_session.is_active = False
                    session.commit()
                    return True
                
                return False
        except Exception as e:
            logger.error(f"Session revocation error: {str(e)}")
            return False
    
    @staticmethod
    def cleanup_expired_sessions():
        """Clean up expired sessions"""
        try:
            with get_db_session() as session:
                expired_sessions = session.query(UserSession).filter(
                    UserSession.expires_at < datetime.utcnow()
                ).all()
                
                for expired_session in expired_sessions:
                    expired_session.is_active = False
                
                session.commit()
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        except Exception as e:
            logger.error(f"Session cleanup error: {str(e)}")

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.security.secret_key, algorithms=[settings.security.algorithm])
        
        return {
            "user_id": payload.get("user_id"),
            "session_id": payload.get("session_id")
        }
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def require_permissions(required_permissions: List[str]):
    """Require specific permissions"""
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        # Implement permission checking logic here
        # For now, just return the user
        return current_user
    
    return permission_checker

# Webhook signature verification
def verify_webhook_signature(platform: str, signature: str, body: bytes, secret: str) -> bool:
    """Verify webhook signature"""
    try:
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()
        
        if signature.startswith("sha256="):
            signature = signature[7:]
        
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error(f"Webhook signature verification error: {str(e)}")
        return False

# Export main components
__all__ = [
    "RateLimitMiddleware",
    "AuthenticationMiddleware", 
    "RequestLoggingMiddleware",
    "SecurityHeadersMiddleware",
    "WebhookSignatureMiddleware",
    "AuthManager",
    "get_current_user",
    "require_permissions",
    "verify_webhook_signature"
]



























