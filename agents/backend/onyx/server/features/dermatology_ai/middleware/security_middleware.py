"""
Middleware de seguridad avanzado
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List
import time
import hashlib


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware de seguridad"""
    
    def __init__(self, app, allowed_ips: List[str] = None,
                 require_https: bool = False):
        """
        Inicializa el middleware
        
        Args:
            app: Aplicación FastAPI
            allowed_ips: Lista de IPs permitidas (None = todas)
            require_https: Requerir HTTPS
        """
        super().__init__(app)
        self.allowed_ips = allowed_ips or []
        self.require_https = require_https
        self.blocked_ips: set = set()
    
    async def dispatch(self, request: Request, call_next):
        """Procesa el request con seguridad"""
        client_ip = request.client.host if request.client else "unknown"
        
        # Verificar IP bloqueada
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=403,
                content={"error": "IP bloqueada"}
            )
        
        # Verificar IP permitida
        if self.allowed_ips and client_ip not in self.allowed_ips:
            return JSONResponse(
                status_code=403,
                content={"error": "IP no permitida"}
            )
        
        # Verificar HTTPS
        if self.require_https and request.url.scheme != "https":
            return JSONResponse(
                status_code=403,
                content={"error": "HTTPS requerido"}
            )
        
        # Verificar headers de seguridad
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
        
        response = await call_next(request)
        
        # Agregar headers de seguridad
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    def block_ip(self, ip: str):
        """Bloquea una IP"""
        self.blocked_ips.add(ip)
    
    def unblock_ip(self, ip: str):
        """Desbloquea una IP"""
        self.blocked_ips.discard(ip)






