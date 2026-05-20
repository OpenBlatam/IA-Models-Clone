"""
Advanced Authentication Middleware for Business Agents
====================================================

Comprehensive authentication and authorization middleware for business agents system.
"""

import asyncio
import logging
import json
import jwt
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import redis
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import (
    BusinessAgent, AgentRequest, AgentResponse, AgentAnalytics,
    AgentWorkflow, AgentCollaboration, AgentSettings,
    ErrorResponse
)
from ..exceptions import (
    AgentAuthenticationError, AgentAuthorizationError, AgentValidationError,
    AgentNotFoundError, AgentPermissionDeniedError, AgentSystemError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Advanced authentication middleware for business agents"""
    
    def __init__(self, app, redis_client: redis.Redis, db_session: AsyncSession):
        super().__init__(app)
        self.redis = redis_client
        self.db = db_session
        self.settings = get_settings()
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authentication"""
        try:
            # Skip authentication for certain paths
            if self._should_skip_auth(request.url.path):
                return await call_next(request)
            
            # Extract token from request
            token = await self._extract_token(request)
            if not token:
                return self._create_auth_error("No authentication token provided")
            
            # Validate token
            user_data = await self._validate_token(token)
            if not user_data:
                return self._create_auth_error("Invalid authentication token")
            
            # Check user permissions
            if not await self._check_permissions(request, user_data):
                return self._create_auth_error("Insufficient permissions")
            
            # Add user data to request state
            request.state.user = user_data
            request.state.user_id = user_data.get("user_id")
            request.state.roles = user_data.get("roles", [])
            request.state.permissions = user_data.get("permissions", [])
            
            # Log authentication
            await self._log_authentication(request, user_data)
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response = self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            error = handle_agent_error(e, request=request)
            log_agent_error(error)
            return JSONResponse(
                status_code=500,
                content=get_error_response(error)
            )
    
    def _should_skip_auth(self, path: str) -> bool:
        """Check if authentication should be skipped for this path"""
        skip_paths = [
            "/health",
            "/docs",
            "/openapi.json",
            "/metrics",
            "/auth/login",
            "/auth/register",
            "/auth/refresh"
        ]
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    async def _extract_token(self, request: Request) -> Optional[str]:
        """Extract authentication token from request"""
        # Try Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]
        
        # Try query parameter
        token = request.query_params.get("token")
        if token:
            return token
        
        # Try cookie
        token = request.cookies.get("access_token")
        if token:
            return token
        
        return None
    
    async def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user data"""
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.settings.security.secret_key,
                algorithms=[self.settings.security.algorithm]
            )
            
            # Check token expiration
            if datetime.utcnow() > datetime.fromtimestamp(payload.get("exp", 0)):
                return None
            
            # Get user data from cache or database
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # Try cache first
            cached_user = await self._get_cached_user(user_id)
            if cached_user:
                return cached_user
            
            # Get from database
            user_data = await self._get_user_from_db(user_id)
            if user_data:
                # Cache user data
                await self._cache_user(user_id, user_data)
                return user_data
            
            return None
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    async def _check_permissions(self, request: Request, user_data: Dict[str, Any]) -> bool:
        """Check if user has required permissions for the request"""
        try:
            # Get required permissions for the endpoint
            required_permissions = self._get_required_permissions(request)
            if not required_permissions:
                return True  # No specific permissions required
            
            # Check user permissions
            user_permissions = user_data.get("permissions", [])
            user_roles = user_data.get("roles", [])
            
            # Check direct permissions
            if any(perm in user_permissions for perm in required_permissions):
                return True
            
            # Check role-based permissions
            role_permissions = await self._get_role_permissions(user_roles)
            if any(perm in role_permissions for perm in required_permissions):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False
    
    def _get_required_permissions(self, request: Request) -> List[str]:
        """Get required permissions for the endpoint"""
        # Define endpoint permissions mapping
        permissions_map = {
            "POST /agents": ["agents:create"],
            "GET /agents": ["agents:read"],
            "PUT /agents/{agent_id}": ["agents:update"],
            "DELETE /agents/{agent_id}": ["agents:delete"],
            "POST /agents/{agent_id}/execute": ["agents:execute"],
            "GET /agents/{agent_id}/analytics": ["agents:analytics"],
            "POST /workflows": ["workflows:create"],
            "GET /workflows": ["workflows:read"],
            "POST /workflows/{workflow_id}/execute": ["workflows:execute"],
            "GET /analytics": ["analytics:read"],
            "POST /collaboration": ["collaboration:create"],
            "GET /collaboration": ["collaboration:read"]
        }
        
        method = request.method
        path = request.url.path
        
        # Find matching permission
        for pattern, permissions in permissions_map.items():
            if self._match_path(pattern, method, path):
                return permissions
        
        return []
    
    def _match_path(self, pattern: str, method: str, path: str) -> bool:
        """Check if path matches pattern"""
        pattern_method, pattern_path = pattern.split(" ", 1)
        
        if pattern_method != method:
            return False
        
        # Simple pattern matching (could be enhanced with regex)
        if "{" in pattern_path and "}" in pattern_path:
            # Handle parameterized paths
            pattern_parts = pattern_path.split("/")
            path_parts = path.split("/")
            
            if len(pattern_parts) != len(path_parts):
                return False
            
            for pattern_part, path_part in zip(pattern_parts, path_parts):
                if pattern_part.startswith("{") and pattern_part.endswith("}"):
                    continue  # Parameter match
                if pattern_part != path_part:
                    return False
            
            return True
        else:
            return pattern_path == path
    
    async def _get_cached_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data from cache"""
        try:
            cache_key = f"user:{user_id}"
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Failed to get cached user {user_id}: {e}")
        return None
    
    async def _cache_user(self, user_id: str, user_data: Dict[str, Any]) -> None:
        """Cache user data"""
        try:
            cache_key = f"user:{user_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(user_data, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache user {user_id}: {e}")
    
    async def _get_user_from_db(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data from database"""
        try:
            # This would integrate with actual user database
            # For now, return mock user data
            return {
                "user_id": user_id,
                "username": f"user_{user_id}",
                "email": f"user_{user_id}@example.com",
                "roles": ["agent_user", "analytics_user"],
                "permissions": [
                    "agents:read",
                    "agents:create",
                    "agents:update",
                    "agents:execute",
                    "analytics:read",
                    "workflows:read",
                    "workflows:create",
                    "collaboration:read"
                ],
                "is_active": True,
                "last_login": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get user from database {user_id}: {e}")
            return None
    
    async def _get_role_permissions(self, roles: List[str]) -> List[str]:
        """Get permissions for user roles"""
        try:
            # Define role permissions mapping
            role_permissions = {
                "admin": [
                    "agents:create", "agents:read", "agents:update", "agents:delete", "agents:execute",
                    "workflows:create", "workflows:read", "workflows:update", "workflows:delete", "workflows:execute",
                    "analytics:read", "analytics:export",
                    "collaboration:create", "collaboration:read", "collaboration:update", "collaboration:delete",
                    "users:create", "users:read", "users:update", "users:delete"
                ],
                "agent_user": [
                    "agents:read", "agents:create", "agents:update", "agents:execute",
                    "workflows:read", "workflows:create", "workflows:execute",
                    "analytics:read",
                    "collaboration:read", "collaboration:create"
                ],
                "analytics_user": [
                    "agents:read",
                    "analytics:read", "analytics:export",
                    "collaboration:read"
                ],
                "viewer": [
                    "agents:read",
                    "analytics:read",
                    "collaboration:read"
                ]
            }
            
            permissions = []
            for role in roles:
                if role in role_permissions:
                    permissions.extend(role_permissions[role])
            
            return list(set(permissions))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to get role permissions: {e}")
            return []
    
    async def _log_authentication(self, request: Request, user_data: Dict[str, Any]) -> None:
        """Log authentication event"""
        try:
            log_data = {
                "user_id": user_data.get("user_id"),
                "username": user_data.get("username"),
                "method": request.method,
                "path": request.url.path,
                "ip_address": request.client.host,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Authentication successful: {json.dumps(log_data)}")
            
            # Store in audit log
            await self._store_audit_log("authentication", log_data)
            
        except Exception as e:
            logger.error(f"Failed to log authentication: {e}")
    
    async def _store_audit_log(self, event_type: str, log_data: Dict[str, Any]) -> None:
        """Store audit log entry"""
        try:
            audit_entry = {
                "event_id": str(uuid4()),
                "event_type": event_type,
                "data": log_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis for real-time access
            await self.redis.lpush(
                "audit_logs",
                json.dumps(audit_entry, default=str)
            )
            
            # Keep only last 1000 entries
            await self.redis.ltrim("audit_logs", 0, 999)
            
        except Exception as e:
            logger.error(f"Failed to store audit log: {e}")
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        try:
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Add CSP header
            csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
            response.headers["Content-Security-Policy"] = csp
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to add security headers: {e}")
            return response
    
    def _create_auth_error(self, message: str) -> JSONResponse:
        """Create authentication error response"""
        error = AgentAuthenticationError(
            "authentication_failed",
            message,
            {"timestamp": datetime.utcnow().isoformat()}
        )
        
        return JSONResponse(
            status_code=401,
            content=get_error_response(error),
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationMiddleware(BaseHTTPMiddleware):
    """Advanced authorization middleware for business agents"""
    
    def __init__(self, app, redis_client: redis.Redis):
        super().__init__(app)
        self.redis = redis_client
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authorization"""
        try:
            # Skip authorization for certain paths
            if self._should_skip_auth(request.url.path):
                return await call_next(request)
            
            # Get user from request state
            user_data = getattr(request.state, 'user', None)
            if not user_data:
                return self._create_auth_error("User not authenticated")
            
            # Check resource-specific permissions
            if not await self._check_resource_permissions(request, user_data):
                return self._create_auth_error("Insufficient permissions for this resource")
            
            # Check rate limiting
            if not await self._check_rate_limit(request, user_data):
                return self._create_rate_limit_error("Rate limit exceeded")
            
            # Process request
            response = await call_next(request)
            
            # Log authorization
            await self._log_authorization(request, user_data)
            
            return response
            
        except Exception as e:
            error = handle_agent_error(e, request=request)
            log_agent_error(error)
            return JSONResponse(
                status_code=500,
                content=get_error_response(error)
            )
    
    def _should_skip_auth(self, path: str) -> bool:
        """Check if authorization should be skipped for this path"""
        skip_paths = [
            "/health",
            "/docs",
            "/openapi.json",
            "/metrics"
        ]
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    async def _check_resource_permissions(self, request: Request, user_data: Dict[str, Any]) -> bool:
        """Check resource-specific permissions"""
        try:
            # Extract resource ID from path
            resource_id = self._extract_resource_id(request.url.path)
            if not resource_id:
                return True  # No specific resource
            
            # Check if user owns the resource or has admin access
            user_id = user_data.get("user_id")
            user_roles = user_data.get("roles", [])
            
            # Admin users have access to all resources
            if "admin" in user_roles:
                return True
            
            # Check resource ownership
            if await self._check_resource_ownership(resource_id, user_id):
                return True
            
            # Check shared access
            if await self._check_shared_access(resource_id, user_id):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Resource permission check error: {e}")
            return False
    
    def _extract_resource_id(self, path: str) -> Optional[str]:
        """Extract resource ID from path"""
        try:
            # Extract agent ID from paths like /agents/{agent_id}
            if "/agents/" in path:
                parts = path.split("/agents/")
                if len(parts) > 1:
                    return parts[1].split("/")[0]
            
            # Extract workflow ID from paths like /workflows/{workflow_id}
            if "/workflows/" in path:
                parts = path.split("/workflows/")
                if len(parts) > 1:
                    return parts[1].split("/")[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract resource ID: {e}")
            return None
    
    async def _check_resource_ownership(self, resource_id: str, user_id: str) -> bool:
        """Check if user owns the resource"""
        try:
            # This would integrate with actual database
            # For now, return mock ownership check
            return True  # Simplified for demo
            
        except Exception as e:
            logger.error(f"Failed to check resource ownership: {e}")
            return False
    
    async def _check_shared_access(self, resource_id: str, user_id: str) -> bool:
        """Check if user has shared access to the resource"""
        try:
            # This would integrate with actual database
            # For now, return mock shared access check
            return False  # Simplified for demo
            
        except Exception as e:
            logger.error(f"Failed to check shared access: {e}")
            return False
    
    async def _check_rate_limit(self, request: Request, user_data: Dict[str, Any]) -> bool:
        """Check rate limiting"""
        try:
            user_id = user_data.get("user_id")
            endpoint = f"{request.method}:{request.url.path}"
            
            # Create rate limit key
            rate_limit_key = f"rate_limit:{user_id}:{endpoint}"
            
            # Get current count
            current_count = await self.redis.get(rate_limit_key)
            if current_count is None:
                current_count = 0
            else:
                current_count = int(current_count)
            
            # Check if limit exceeded
            rate_limit = self._get_rate_limit(endpoint)
            if current_count >= rate_limit:
                return False
            
            # Increment counter
            await self.redis.incr(rate_limit_key)
            await self.redis.expire(rate_limit_key, 60)  # 1 minute window
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error
    
    def _get_rate_limit(self, endpoint: str) -> int:
        """Get rate limit for endpoint"""
        # Define rate limits per endpoint
        rate_limits = {
            "POST:agents": 10,  # 10 requests per minute
            "GET:agents": 100,  # 100 requests per minute
            "PUT:agents": 20,   # 20 requests per minute
            "DELETE:agents": 5, # 5 requests per minute
            "POST:workflows": 10,
            "GET:workflows": 100,
            "POST:analytics": 50,
            "GET:analytics": 200
        }
        
        return rate_limits.get(endpoint, 60)  # Default 60 requests per minute
    
    async def _log_authorization(self, request: Request, user_data: Dict[str, Any]) -> None:
        """Log authorization event"""
        try:
            log_data = {
                "user_id": user_data.get("user_id"),
                "username": user_data.get("username"),
                "method": request.method,
                "path": request.url.path,
                "ip_address": request.client.host,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Authorization successful: {json.dumps(log_data)}")
            
            # Store in audit log
            await self._store_audit_log("authorization", log_data)
            
        except Exception as e:
            logger.error(f"Failed to log authorization: {e}")
    
    async def _store_audit_log(self, event_type: str, log_data: Dict[str, Any]) -> None:
        """Store audit log entry"""
        try:
            audit_entry = {
                "event_id": str(uuid4()),
                "event_type": event_type,
                "data": log_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis for real-time access
            await self.redis.lpush(
                "audit_logs",
                json.dumps(audit_entry, default=str)
            )
            
            # Keep only last 1000 entries
            await self.redis.ltrim("audit_logs", 0, 999)
            
        except Exception as e:
            logger.error(f"Failed to store audit log: {e}")
    
    def _create_auth_error(self, message: str) -> JSONResponse:
        """Create authorization error response"""
        error = AgentAuthorizationError(
            "authorization_failed",
            message,
            {"timestamp": datetime.utcnow().isoformat()}
        )
        
        return JSONResponse(
            status_code=403,
            content=get_error_response(error)
        )
    
    def _create_rate_limit_error(self, message: str) -> JSONResponse:
        """Create rate limit error response"""
        error = AgentSystemError(
            "rate_limit_exceeded",
            message,
            {"timestamp": datetime.utcnow().isoformat()}
        )
        
        return JSONResponse(
            status_code=429,
            content=get_error_response(error),
            headers={"Retry-After": "60"}
        )


# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
    """Get current authenticated user"""
    # This would be used in FastAPI endpoints
    # The middleware handles the actual authentication
    pass


async def require_permissions(required_permissions: List[str]):
    """Dependency to require specific permissions"""
    def permission_checker(user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = user.get("permissions", [])
        if not any(perm in user_permissions for perm in required_permissions):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions"
            )
        return user
    
    return permission_checker


async def require_roles(required_roles: List[str]):
    """Dependency to require specific roles"""
    def role_checker(user: Dict[str, Any] = Depends(get_current_user)):
        user_roles = user.get("roles", [])
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                status_code=403,
                detail="Insufficient roles"
            )
        return user
    
    return role_checker