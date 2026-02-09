"""
Authentication Middleware
=========================

Advanced authentication middleware.
"""

import logging
from typing import Optional, List
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import jwt

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware."""
    
    def __init__(
        self,
        app,
        secret_key: str,
        algorithm: str = "HS256",
        excluded_paths: Optional[List[str]] = None,
        required_roles: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.excluded_paths = excluded_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.required_roles = required_roles or []
    
    async def dispatch(self, request: Request, call_next):
        """Validate authentication."""
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Get token
        token = self._extract_token(request)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Validate token
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            request.state.user = payload
            request.state.user_id = payload.get("sub")
            request.state.roles = payload.get("roles", [])
            
            # Check roles if required
            if self.required_roles:
                if not any(role in request.state.roles for role in self.required_roles):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return await call_next(request)
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract token from request."""
        # Try Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
        
        # Try cookie
        return request.cookies.get("access_token")















