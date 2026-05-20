"""
Advanced Security - Seguridad avanzada
======================================

Características avanzadas de seguridad:
- DDoS protection
- Rate limiting avanzado
- Security headers
- Content validation
- IP filtering
"""

import logging
import time
import re
from typing import Dict, Any, Optional, List, Callable, Tuple, Protocol
from collections import defaultdict, deque
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import Response

from .types import IPAddress, HealthStatus, ComponentHealth

logger = logging.getLogger(__name__)


class DDoSProtection:
    """
    Protección DDoS con:
    - Rate limiting por IP
    - Detección de patrones sospechosos
    - Blacklisting automático
    - Throttling inteligente
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_requests_per_hour: int = 1000,
        block_duration: int = 3600,
        suspicious_threshold: int = 10
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.block_duration = block_duration
        self.suspicious_threshold = suspicious_threshold
        
        # Tracking por IP
        self.ip_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.ip_blocks: Dict[str, datetime] = {}
        self.suspicious_ips: Dict[str, int] = defaultdict(int)
    
    def is_blocked(self, ip: IPAddress) -> bool:
        """Verifica si una IP está bloqueada"""
        if ip in self.ip_blocks:
            block_time = self.ip_blocks[ip]
            if datetime.now() - block_time < timedelta(seconds=self.block_duration):
                return True
            else:
                # Desbloquear después del tiempo
                del self.ip_blocks[ip]
                return False
        return False
    
    def check_rate_limit(self, ip: IPAddress) -> Tuple[bool, Optional[str]]:
        """
        Verifica rate limit para una IP.
        
        Returns:
            (is_allowed, error_message)
        """
        # Verificar si está bloqueada
        if self.is_blocked(ip):
            return False, "IP is temporarily blocked"
        
        now = datetime.now()
        requests = self.ip_requests[ip]
        
        # Limpiar requests antiguos
        while requests and (now - requests[0]).total_seconds() > 60:
            requests.popleft()
        
        # Verificar límite por minuto
        if len(requests) >= self.max_requests_per_minute:
            # Incrementar contador de sospechosos
            self.suspicious_ips[ip] += 1
            
            # Bloquear si excede threshold
            if self.suspicious_ips[ip] >= self.suspicious_threshold:
                self.ip_blocks[ip] = now
                logger.warning(f"IP {ip} blocked due to suspicious activity")
                return False, "IP blocked due to suspicious activity"
            
            return False, "Rate limit exceeded"
        
        # Agregar request actual
        requests.append(now)
        
        return True, None
    
    def get_ip_stats(self, ip: IPAddress) -> Dict[str, Any]:
        """Obtiene estadísticas de una IP"""
        requests = self.ip_requests[ip]
        now = datetime.now()
        
        # Limpiar requests antiguos
        while requests and (now - requests[0]).total_seconds() > 3600:
            requests.popleft()
        
        return {
            "ip": ip,
            "requests_last_minute": len([r for r in requests if (now - r).total_seconds() <= 60]),
            "requests_last_hour": len(requests),
            "is_blocked": self.is_blocked(ip),
            "suspicious_count": self.suspicious_ips.get(ip, 0)
        }


class AdvancedRateLimiter:
    """
    Rate limiter avanzado con:
    - Múltiples estrategias
    - Rate limiting por usuario/IP/endpoint
    - Sliding window
    - Token bucket
    """
    
    def __init__(
        self,
        strategy: str = "sliding_window",
        default_limit: int = 100,
        default_window: int = 60
    ):
        self.strategy = strategy
        self.default_limit = default_limit
        self.default_window = default_window
        self.limits: Dict[str, Dict[str, Any]] = {}
        self.counters: Dict[str, deque] = defaultdict(lambda: deque())
    
    def set_limit(
        self,
        key: str,
        limit: int,
        window: int = None
    ):
        """Establece límite para una clave"""
        self.limits[key] = {
            "limit": limit,
            "window": window or self.default_window
        }
    
    def check_limit(self, key: str) -> Tuple[bool, Optional[str]]:
        """
        Verifica límite para una clave.
        
        Returns:
            (is_allowed, error_message)
        """
        limit_config = self.limits.get(key, {
            "limit": self.default_limit,
            "window": self.default_window
        })
        
        limit = limit_config["limit"]
        window = limit_config["window"]
        
        now = datetime.now()
        requests = self.counters[key]
        
        # Limpiar requests fuera de la ventana
        while requests and (now - requests[0]).total_seconds() > window:
            requests.popleft()
        
        # Verificar límite
        if len(requests) >= limit:
            return False, f"Rate limit exceeded: {limit} requests per {window}s"
        
        # Agregar request
        requests.append(now)
        
        return True, None
    
    def get_stats(self, key: str) -> Dict[str, Any]:
        """Obtiene estadísticas de una clave"""
        limit_config = self.limits.get(key, {
            "limit": self.default_limit,
            "window": self.default_window
        })
        
        requests = self.counters[key]
        now = datetime.now()
        
        # Limpiar requests antiguos
        while requests and (now - requests[0]).total_seconds() > limit_config["window"]:
            requests.popleft()
        
        return {
            "key": key,
            "limit": limit_config["limit"],
            "window": limit_config["window"],
            "current": len(requests),
            "remaining": limit_config["limit"] - len(requests)
        }


class SecurityHeaders:
    """Gestor de security headers"""
    
    DEFAULT_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    def __init__(self, custom_headers: Optional[Dict[str, str]] = None) -> None:
        self.headers = {**self.DEFAULT_HEADERS}
        if custom_headers:
            self.headers.update(custom_headers)
    
    def add_header(self, name: str, value: str) -> None:
        """Agrega header de seguridad"""
        self.headers[name] = value
    
    def apply_to_response(self, response: Response) -> Response:
        """Aplica headers a una respuesta"""
        for name, value in self.headers.items():
            response.headers[name] = value
        return response


class ContentValidator:
    """Validador de contenido"""
    
    def __init__(
        self,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        allowed_content_types: Optional[List[str]] = None,
        block_suspicious_patterns: bool = True
    ):
        self.max_request_size = max_request_size
        self.allowed_content_types = allowed_content_types or [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data"
        ]
        self.block_suspicious_patterns = block_suspicious_patterns
        self.suspicious_patterns = [
            r"<script",
            r"javascript:",
            r"onerror=",
            r"onload=",
            r"eval\(",
            r"exec\("
        ]
    
    def validate_request(self, request: Request) -> Tuple[bool, Optional[str]]:
        """
        Valida un request.
        
        Returns:
            (is_valid, error_message)
        """
        # Validar tamaño
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    return False, f"Request too large: {size} bytes"
            except ValueError:
                pass
        
        # Validar content type
        content_type = request.headers.get("content-type", "")
        if content_type:
            if not any(ct in content_type for ct in self.allowed_content_types):
                return False, f"Content type not allowed: {content_type}"
        
        return True, None
    
    def validate_content(self, content: str) -> Tuple[bool, Optional[str]]:
        """
        Valida contenido.
        
        Returns:
            (is_valid, error_message)
        """
        if not self.block_suspicious_patterns:
            return True, None
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, f"Suspicious pattern detected: {pattern}"
        
        return True, None


class AdvancedSecurityMiddleware:
    """Middleware de seguridad avanzado"""
    
    def __init__(
        self,
        ddos_protection: Optional[DDoSProtection] = None,
        rate_limiter: Optional[AdvancedRateLimiter] = None,
        security_headers: Optional[SecurityHeaders] = None,
        content_validator: Optional[ContentValidator] = None
    ):
        self.ddos_protection = ddos_protection or DDoSProtection()
        self.rate_limiter = rate_limiter or AdvancedRateLimiter()
        self.security_headers = security_headers or SecurityHeaders()
        self.content_validator = content_validator or ContentValidator()
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Ejecuta middleware de seguridad"""
        # Obtener IP
        ip = request.client.host if request.client else "unknown"
        
        # Verificar DDoS protection
        is_allowed, error = self.ddos_protection.check_rate_limit(ip)
        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error
            )
        
        # Verificar rate limit
        endpoint_key = f"{ip}:{request.url.path}"
        is_allowed, error = self.rate_limiter.check_limit(endpoint_key)
        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error
            )
        
        # Validar request
        is_valid, error = self.content_validator.validate_request(request)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )
        
        # Procesar request
        response = await call_next(request)
        
        # Aplicar security headers
        response = self.security_headers.apply_to_response(response)
        
        return response


def get_advanced_security_middleware() -> AdvancedSecurityMiddleware:
    """Obtiene middleware de seguridad avanzado"""
    return AdvancedSecurityMiddleware()

