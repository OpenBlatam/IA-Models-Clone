"""
Middleware de autenticación y autorización JWT
"""

import logging
import jwt
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware para autenticación JWT"""
    
    def __init__(self, app, secret_key: str = None, algorithm: str = "HS256"):
        super().__init__(app)
        self.secret_key = secret_key or getattr(settings, 'jwt_secret_key', 'your-secret-key-change-in-production')
        self.algorithm = algorithm
        self.public_paths = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/suno/health"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Verificar si la ruta es pública
        if request.url.path in self.public_paths or request.url.path.startswith("/docs"):
            return await call_next(request)
        
        # Obtener token
        token = self._extract_token(request)
        
        if not token:
            # Permitir requests sin token pero marcar como anónimo
            request.state.user = None
            request.state.is_authenticated = False
            return await call_next(request)
        
        # Validar token
        try:
            payload = self._verify_token(token)
            request.state.user = payload
            request.state.is_authenticated = True
            request.state.user_id = payload.get("user_id") or payload.get("sub")
        except jwt.ExpiredSignatureError:
            logger.warning(f"Expired token for {request.url.path}")
            request.state.user = None
            request.state.is_authenticated = False
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            request.state.user = None
            request.state.is_authenticated = False
        
        return await call_next(request)
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """Extrae el token JWT del header Authorization"""
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
        return None
    
    def _verify_token(self, token: str) -> Dict[str, Any]:
        """Verifica y decodifica el token JWT"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise
        except jwt.InvalidTokenError as e:
            raise


def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Dependency para obtener el usuario actual"""
    if not hasattr(request.state, 'is_authenticated') or not request.state.is_authenticated:
        return None
    return getattr(request.state, 'user', None)


def require_auth(request: Request) -> Dict[str, Any]:
    """Dependency que requiere autenticación"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(required_role: str):
    """Dependency factory para requerir un rol específico"""
    def role_checker(user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        user_roles = user.get("roles", [])
        if required_role not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return user
    return role_checker

