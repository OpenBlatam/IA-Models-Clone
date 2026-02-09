"""
Security Middleware - Middleware de seguridad avanzado
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import hashlib
import hmac
import secrets

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Middleware de seguridad avanzado."""
    
    def __init__(self):
        self.rate_limit_storage = {}
        self.blocked_ips = set()
        self.suspicious_ips = set()
        self.request_counts = {}
        
        # Configuración de seguridad
        self.rate_limit_requests = 100  # requests per minute
        self.rate_limit_window = 60  # seconds
        self.block_duration = 300  # 5 minutes
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        
        # Headers de seguridad
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'"
        }
    
    async def __call__(self, request: Request, call_next):
        """Ejecutar middleware de seguridad."""
        start_time = time.time()
        
        try:
            # Verificar IP bloqueada
            client_ip = self._get_client_ip(request)
            if client_ip in self.blocked_ips:
                return JSONResponse(
                    status_code=403,
                    content={"error": "IP bloqueada", "code": "IP_BLOCKED"}
                )
            
            # Verificar rate limiting
            if not self._check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit excedido", "code": "RATE_LIMIT_EXCEEDED"}
                )
            
            # Verificar tamaño de request
            if not self._check_request_size(request):
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request demasiado grande", "code": "REQUEST_TOO_LARGE"}
                )
            
            # Verificar headers sospechosos
            if self._check_suspicious_headers(request):
                self.suspicious_ips.add(client_ip)
                logger.warning(f"Headers sospechosos detectados desde IP: {client_ip}")
            
            # Procesar request
            response = await call_next(request)
            
            # Agregar headers de seguridad
            self._add_security_headers(response)
            
            # Registrar request
            self._log_request(request, response, time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error en middleware de seguridad: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Error interno del servidor", "code": "INTERNAL_ERROR"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Obtener IP del cliente."""
        # Verificar headers de proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Verificar rate limiting."""
        now = time.time()
        window_start = now - self.rate_limit_window
        
        # Limpiar entradas antiguas
        if client_ip in self.rate_limit_storage:
            self.rate_limit_storage[client_ip] = [
                timestamp for timestamp in self.rate_limit_storage[client_ip]
                if timestamp > window_start
            ]
        else:
            self.rate_limit_storage[client_ip] = []
        
        # Verificar límite
        if len(self.rate_limit_storage[client_ip]) >= self.rate_limit_requests:
            # IP sospechosa
            self.suspicious_ips.add(client_ip)
            return False
        
        # Agregar timestamp actual
        self.rate_limit_storage[client_ip].append(now)
        return True
    
    def _check_request_size(self, request: Request) -> bool:
        """Verificar tamaño del request."""
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                return size <= self.max_request_size
            except ValueError:
                return False
        return True
    
    def _check_suspicious_headers(self, request: Request) -> bool:
        """Verificar headers sospechosos."""
        suspicious_patterns = [
            "script", "javascript", "vbscript", "onload", "onerror",
            "eval", "expression", "url(", "import", "require"
        ]
        
        # Verificar User-Agent
        user_agent = request.headers.get("User-Agent", "").lower()
        for pattern in suspicious_patterns:
            if pattern in user_agent:
                return True
        
        # Verificar otros headers
        for header_name, header_value in request.headers.items():
            if any(pattern in header_value.lower() for pattern in suspicious_patterns):
                return True
        
        return False
    
    def _add_security_headers(self, response: Response):
        """Agregar headers de seguridad."""
        for header, value in self.security_headers.items():
            response.headers[header] = value
    
    def _log_request(self, request: Request, response: Response, duration: float):
        """Registrar request para análisis."""
        client_ip = self._get_client_ip(request)
        
        # Incrementar contador de requests
        self.request_counts[client_ip] = self.request_counts.get(client_ip, 0) + 1
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} - Status: {response.status_code} "
            f"Duration: {duration:.3f}s"
        )
    
    def block_ip(self, ip: str, duration: int = None):
        """Bloquear IP temporalmente."""
        self.blocked_ips.add(ip)
        
        if duration:
            # Programar desbloqueo
            import asyncio
            asyncio.create_task(self._unblock_ip_after(ip, duration))
        
        logger.warning(f"IP bloqueada: {ip}")
    
    async def _unblock_ip_after(self, ip: str, duration: int):
        """Desbloquear IP después de un tiempo."""
        await asyncio.sleep(duration)
        self.blocked_ips.discard(ip)
        logger.info(f"IP desbloqueada: {ip}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de seguridad."""
        return {
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "total_requests": sum(self.request_counts.values()),
            "unique_ips": len(self.request_counts),
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
            "last_updated": datetime.now().isoformat()
        }


class APIKeyMiddleware:
    """Middleware para autenticación con API Key."""
    
    def __init__(self, valid_api_keys: set = None):
        self.valid_api_keys = valid_api_keys or set()
        self.api_key_usage = {}
    
    async def __call__(self, request: Request, call_next):
        """Verificar API Key."""
        # Rutas que no requieren autenticación
        public_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
        
        if request.url.path in public_paths:
            return await call_next(request)
        
        # Obtener API Key
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "API Key requerida", "code": "API_KEY_REQUIRED"}
            )
        
        # Verificar API Key
        if api_key not in self.valid_api_keys:
            return JSONResponse(
                status_code=401,
                content={"error": "API Key inválida", "code": "API_KEY_INVALID"}
            )
        
        # Registrar uso
        self.api_key_usage[api_key] = self.api_key_usage.get(api_key, 0) + 1
        
        return await call_next(request)
    
    def add_api_key(self, api_key: str):
        """Agregar API Key válida."""
        self.valid_api_keys.add(api_key)
    
    def remove_api_key(self, api_key: str):
        """Remover API Key."""
        self.valid_api_keys.discard(api_key)
    
    def get_api_key_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de API Keys."""
        return {
            "total_api_keys": len(self.valid_api_keys),
            "api_key_usage": self.api_key_usage,
            "last_updated": datetime.now().isoformat()
        }


class RequestLoggingMiddleware:
    """Middleware para logging detallado de requests."""
    
    def __init__(self):
        self.request_log = []
        self.max_log_entries = 1000
    
    async def __call__(self, request: Request, call_next):
        """Loggear request detallado."""
        start_time = time.time()
        
        # Información del request
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", ""),
        }
        
        try:
            # Procesar request
            response = await call_next(request)
            
            # Información de la respuesta
            request_info.update({
                "status_code": response.status_code,
                "response_headers": dict(response.headers),
                "duration": time.time() - start_time,
                "success": 200 <= response.status_code < 400
            })
            
            # Agregar al log
            self.request_log.append(request_info)
            
            # Mantener solo las últimas entradas
            if len(self.request_log) > self.max_log_entries:
                self.request_log = self.request_log[-self.max_log_entries:]
            
            return response
            
        except Exception as e:
            request_info.update({
                "error": str(e),
                "duration": time.time() - start_time,
                "success": False
            })
            
            self.request_log.append(request_info)
            raise
    
    def get_request_logs(self, limit: int = 100) -> list:
        """Obtener logs de requests."""
        return self.request_log[-limit:]
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de requests."""
        if not self.request_log:
            return {"total_requests": 0}
        
        total_requests = len(self.request_log)
        successful_requests = sum(1 for req in self.request_log if req.get("success", False))
        failed_requests = total_requests - successful_requests
        
        # Promedio de duración
        durations = [req.get("duration", 0) for req in self.request_log if "duration" in req]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Status codes más comunes
        status_codes = {}
        for req in self.request_log:
            status = req.get("status_code", 0)
            status_codes[status] = status_codes.get(status, 0) + 1
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "average_duration": avg_duration,
            "status_codes": status_codes,
            "last_updated": datetime.now().isoformat()
        }




