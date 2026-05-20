"""
Security Headers Middleware
===========================

Middleware para agregar headers de seguridad a las respuestas.
"""

import logging
from typing import Optional, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware para agregar headers de seguridad.
    
    Agrega headers de seguridad estándar a todas las respuestas:
    - X-Content-Type-Options: Prevenir MIME type sniffing
    - X-Frame-Options: Prevenir clickjacking
    - X-XSS-Protection: Protección XSS
    - Strict-Transport-Security: Forzar HTTPS
    - Referrer-Policy: Control de referrer
    - Content-Security-Policy: Política de seguridad de contenido (opcional)
    - Permissions-Policy: Política de permisos (opcional)
    """
    
    def __init__(
        self,
        app,
        enable_csp: bool = False,
        csp_policy: Optional[str] = None,
        enable_permissions_policy: bool = False,
        permissions_policy: Optional[str] = None,
        hsts_max_age: int = 31536000,
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = False
    ):
        """
        Inicializar middleware de seguridad.
        
        Args:
            app: Aplicación FastAPI
            enable_csp: Habilitar Content-Security-Policy
            csp_policy: Política CSP personalizada (opcional)
            enable_permissions_policy: Habilitar Permissions-Policy
            permissions_policy: Política de permisos personalizada (opcional)
            hsts_max_age: Max age para HSTS en segundos
            hsts_include_subdomains: Incluir subdominios en HSTS
            hsts_preload: Habilitar preload en HSTS
        """
        super().__init__(app)
        self.enable_csp = enable_csp
        self.csp_policy = csp_policy or "default-src 'self'"
        self.enable_permissions_policy = enable_permissions_policy
        self.permissions_policy = permissions_policy or "geolocation=(), microphone=(), camera=()"
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Procesar request y agregar headers de seguridad.
        
        Args:
            request: Request object
            call_next: Next middleware/handler
        
        Returns:
            Response con headers de seguridad
        """
        response = await call_next(request)
        
        # Headers de seguridad básicos
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HSTS solo para HTTPS
        if request.url.scheme == "https":
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.hsts_preload:
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value
        
        # Content-Security-Policy (opcional)
        if self.enable_csp:
            response.headers["Content-Security-Policy"] = self.csp_policy
        
        # Permissions-Policy (opcional)
        if self.enable_permissions_policy:
            response.headers["Permissions-Policy"] = self.permissions_policy
        
        return response


def create_security_middleware(
    app,
    enable_csp: bool = False,
    csp_policy: Optional[str] = None,
    enable_permissions_policy: bool = False,
    **kwargs
):
    """
    Crear middleware de seguridad configurado.
    
    Args:
        app: FastAPI application
        enable_csp: Habilitar CSP
        csp_policy: Política CSP
        enable_permissions_policy: Habilitar Permissions-Policy
        **kwargs: Argumentos adicionales para SecurityHeadersMiddleware
    
    Returns:
        FastAPI app con middleware de seguridad
    """
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_csp=enable_csp,
        csp_policy=csp_policy,
        enable_permissions_policy=enable_permissions_policy,
        **kwargs
    )
    
    logger.info("Security headers middleware configured")
    return app

