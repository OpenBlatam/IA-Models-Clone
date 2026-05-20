"""
OAuth2 Middleware for Secure API Access
Implements OAuth2 with JWT tokens and role-based access control
"""

import logging
from typing import Optional, List
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import jwt
from jwt import PyJWTError

from config.aws_settings import get_aws_settings
from aws.aws_services import SecretsManagerService, ParameterStoreService

logger = logging.getLogger(__name__)
aws_settings = get_aws_settings()
security = HTTPBearer(auto_error=False)


class OAuth2Middleware(BaseHTTPMiddleware):
    """
    OAuth2 middleware for authentication and authorization
    
    Features:
    - JWT token validation
    - Role-based access control (RBAC)
    - Token refresh support
    - Integration with AWS Secrets Manager
    """
    
    def __init__(self, app: ASGIApp, public_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.public_paths = public_paths or [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/recovery/health",
            "/recovery/auth/login",
            "/recovery/auth/register"
        ]
        self._secret_key: Optional[str] = None
        self._algorithm = "HS256"
    
    def _get_secret_key(self) -> str:
        """Get JWT secret key from Secrets Manager or Parameter Store"""
        if self._secret_key:
            return self._secret_key
        
        try:
            # Try Secrets Manager first
            if aws_settings.secrets_manager_secret_name:
                secrets = SecretsManagerService()
                secret_data = secrets.get_secret()
                self._secret_key = secret_data.get("JWT_SECRET_KEY") or secret_data.get("SECRET_KEY")
            
            # Fallback to Parameter Store
            if not self._secret_key:
                params = ParameterStoreService()
                self._secret_key = params.get_parameter("jwt/secret_key")
            
            # Fallback to environment variable
            if not self._secret_key:
                import os
                self._secret_key = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY")
            
            if not self._secret_key:
                raise ValueError("JWT secret key not found")
            
            return self._secret_key
            
        except Exception as e:
            logger.error(f"Error getting JWT secret key: {str(e)}")
            raise
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no authentication required)"""
        return any(path.startswith(public_path) for public_path in self.public_paths)
    
    async def dispatch(self, request: Request, call_next):
        """Process request with OAuth2 authentication"""
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # Get authorization header
        credentials: Optional[HTTPAuthorizationCredentials] = await security(request)
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            # Verify token
            token = credentials.credentials
            payload = self._verify_token(token)
            
            # Add user info to request state
            request.state.user = payload
            request.state.user_id = payload.get("sub") or payload.get("user_id")
            request.state.roles = payload.get("roles", [])
            
            # Check permissions (optional)
            if not self._check_permissions(request, payload):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await call_next(request)
            
        except PyJWTError as e:
            logger.warning(f"JWT validation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
    
    def _verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        secret_key = self._get_secret_key()
        
        try:
            payload = jwt.decode(
                token,
                secret_key,
                algorithms=[self._algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                }
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    def _check_permissions(self, request: Request, payload: dict) -> bool:
        """Check if user has permissions for the requested resource"""
        # Extract required role from path (optional)
        # Example: /admin/* requires admin role
        if request.url.path.startswith("/admin"):
            required_role = "admin"
            user_roles = payload.get("roles", [])
            return required_role in user_roles
        
        # Default: allow if authenticated
        return True


def get_current_user(request: Request) -> dict:
    """Dependency to get current user from request state"""
    if not hasattr(request.state, "user"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return request.state.user


def require_role(required_role: str):
    """Decorator to require specific role"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user = get_current_user(request)
            user_roles = user.get("roles", [])
            
            if required_role not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires role: {required_role}"
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator















