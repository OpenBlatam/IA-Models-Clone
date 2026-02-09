"""
Security Middleware - Middleware de Seguridad
=============================================

Middleware para FastAPI con funcionalidades de seguridad.
"""

import logging
from typing import Optional, Callable
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Intentar importar módulos de autenticación
try:
    from .auth import AuthManager, Permission
    from .security_audit import SecurityAuditor, AuditEventType
    _has_auth = True
except ImportError:
    _has_auth = False
    logger.warning("Auth modules not available. Security middleware will be limited.")


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware de seguridad para FastAPI.
    
    Incluye:
    - Autenticación por token
    - Rate limiting por IP
    - Validación de headers de seguridad
    - Logging de requests
    """
    
    def __init__(
        self,
        app,
        auth_manager: Optional[AuthManager] = None,
        auditor: Optional[SecurityAuditor] = None,
        require_auth: bool = False,
        allowed_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.auth_manager = auth_manager
        self.auditor = auditor
        self.require_auth = require_auth
        self.allowed_paths = allowed_paths or ["/health", "/docs", "/openapi.json", "/redoc"]
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesar request con seguridad.
        
        Args:
            request: Request HTTP
            call_next: Siguiente middleware/handler
            
        Returns:
            Response HTTP
        """
        # Obtener información del request
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        path = request.url.path
        
        # Verificar si la ruta requiere autenticación
        needs_auth = self.require_auth and path not in self.allowed_paths
        
        # Autenticación
        user = None
        if needs_auth and self.auth_manager:
            try:
                # Intentar obtener token desde header
                authorization = request.headers.get("authorization", "")
                if authorization.startswith("Bearer "):
                    token = authorization[7:]  # Remover "Bearer "
                    user = self.auth_manager.validate_token(token)
                    
                    if not user:
                        # Log intento fallido
                        if self.auditor:
                            self.auditor.log_event(
                                event_type=AuditEventType.AUTH_FAILURE,
                                ip_address=client_ip,
                                user_agent=user_agent,
                                resource=path,
                                success=False,
                                severity="warning",
                                details={"reason": "invalid_token"}
                            )
                        
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token"
                        )
                    
                    # Log autenticación exitosa
                    if self.auditor:
                        self.auditor.log_event(
                            event_type=AuditEventType.AUTH_SUCCESS,
                            user_id=user.id,
                            username=user.username,
                            ip_address=client_ip,
                            user_agent=user_agent,
                            resource=path,
                            success=True,
                            severity="info"
                        )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in authentication: {e}")
                if self.auditor:
                    self.auditor.log_event(
                        event_type=AuditEventType.SECURITY_VIOLATION,
                        ip_address=client_ip,
                        user_agent=user_agent,
                        resource=path,
                        success=False,
                        severity="error",
                        details={"error": str(e)}
                    )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication failed"
                )
        
        # Agregar usuario al request state
        request.state.user = user
        
        # Headers de seguridad
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        }
        
        # Procesar request
        try:
            response = await call_next(request)
            
            # Agregar headers de seguridad
            for header, value in security_headers.items():
                response.headers[header] = value
            
            # Log request exitoso
            if self.auditor and user:
                self.auditor.log_event(
                    event_type=AuditEventType.DATA_ACCESSED,
                    user_id=user.id,
                    username=user.username,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    resource=path,
                    action=request.method,
                    success=True,
                    severity="info",
                    details={
                        "status_code": response.status_code,
                        "method": request.method
                    }
                )
            
            return response
            
        except HTTPException as e:
            # Log error HTTP
            if self.auditor:
                self.auditor.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    user_id=user.id if user else None,
                    username=user.username if user else None,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    resource=path,
                    action=request.method,
                    success=False,
                    severity="warning",
                    details={
                        "status_code": e.status_code,
                        "detail": str(e.detail)
                    }
                )
            raise
        except Exception as e:
            # Log error inesperado
            logger.error(f"Unexpected error in security middleware: {e}", exc_info=True)
            if self.auditor:
                self.auditor.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    user_id=user.id if user else None,
                    username=user.username if user else None,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    resource=path,
                    action=request.method,
                    success=False,
                    severity="error",
                    details={"error": str(e)}
                )
            raise


def require_permission(permission: Permission):
    """
    Decorador para requerir permiso en endpoint.
    
    Args:
        permission: Permiso requerido
        
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            user = getattr(request.state, 'user', None)
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not user.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission {permission.value} required"
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator




