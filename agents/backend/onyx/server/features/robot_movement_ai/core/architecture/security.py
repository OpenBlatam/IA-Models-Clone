"""
Módulo de seguridad para Robot Movement AI v2.0
Incluye rate limiting, validación de entrada, y protección CSRF
"""

import time
import hashlib
import hmac
from typing import Optional, Dict, List, Callable
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps

try:
    from fastapi import Request, HTTPException, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@dataclass
class RateLimitConfig:
    """Configuración de rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10


class RateLimiter:
    """Rate limiter simple basado en memoria"""
    
    def __init__(self, config: RateLimitConfig):
        """
        Inicializar rate limiter
        
        Args:
            config: Configuración de rate limiting
        """
        self.config = config
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked: Dict[str, datetime] = {}
    
    def _get_client_id(self, request: Optional[Request] = None, client_ip: Optional[str] = None) -> str:
        """Obtener identificador único del cliente"""
        if request and FASTAPI_AVAILABLE:
            # Intentar obtener de header X-Forwarded-For primero
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
            return request.client.host if request.client else "unknown"
        return client_ip or "unknown"
    
    def is_allowed(self, client_id: Optional[str] = None, request: Optional[Request] = None) -> tuple[bool, Optional[str]]:
        """
        Verificar si el cliente puede hacer una petición
        
        Returns:
            Tuple de (allowed, reason)
        """
        client_id = client_id or self._get_client_id(request)
        now = time.time()
        
        # Verificar si está bloqueado
        if client_id in self.blocked:
            if datetime.now() < self.blocked[client_id]:
                return False, "Rate limit exceeded. Please try again later."
            del self.blocked[client_id]
        
        # Limpiar requests antiguos
        client_requests = self.requests[client_id]
        client_requests[:] = [req_time for req_time in client_requests if now - req_time < 60]
        
        # Verificar límites
        recent_requests = len([r for r in client_requests if now - r < 60])
        if recent_requests >= self.config.requests_per_minute:
            # Bloquear por 1 minuto
            self.blocked[client_id] = datetime.now() + timedelta(minutes=1)
            return False, "Rate limit exceeded. Too many requests per minute."
        
        # Agregar request actual
        client_requests.append(now)
        
        return True, None
    
    def reset(self, client_id: Optional[str] = None):
        """Resetear contador para un cliente"""
        if client_id:
            self.requests.pop(client_id, None)
            self.blocked.pop(client_id, None)
        else:
            self.requests.clear()
            self.blocked.clear()


class InputValidator:
    """Validador de entrada para prevenir inyecciones y datos maliciosos"""
    
    MAX_STRING_LENGTH = 10000
    MAX_ARRAY_SIZE = 1000
    ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-@ ")
    
    @staticmethod
    def validate_string(value: str, max_length: Optional[int] = None, allow_special: bool = True) -> bool:
        """
        Validar string
        
        Args:
            value: String a validar
            max_length: Longitud máxima
            allow_special: Permitir caracteres especiales
            
        Returns:
            True si es válido
        """
        if not isinstance(value, str):
            return False
        
        max_len = max_length or InputValidator.MAX_STRING_LENGTH
        if len(value) > max_len:
            return False
        
        if not allow_special:
            if not all(c in InputValidator.ALLOWED_CHARS for c in value):
                return False
        
        # Verificar patrones sospechosos
        suspicious_patterns = [
            "<script",
            "javascript:",
            "onerror=",
            "onload=",
            "eval(",
            "exec(",
        ]
        
        value_lower = value.lower()
        for pattern in suspicious_patterns:
            if pattern in value_lower:
                return False
        
        return True
    
    @staticmethod
    def validate_number(value: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
        """Validar número"""
        if not isinstance(value, (int, float)):
            return False
        
        if min_value is not None and value < min_value:
            return False
        
        if max_value is not None and value > max_value:
            return False
        
        return True
    
    @staticmethod
    def validate_array(value: list, max_size: Optional[int] = None) -> bool:
        """Validar array"""
        if not isinstance(value, list):
            return False
        
        max_sz = max_size or InputValidator.MAX_ARRAY_SIZE
        if len(value) > max_sz:
            return False
        
        return True


class CSRFProtection:
    """Protección CSRF simple"""
    
    def __init__(self, secret_key: str):
        """
        Inicializar protección CSRF
        
        Args:
            secret_key: Clave secreta para generar tokens
        """
        self.secret_key = secret_key.encode()
        self.tokens: Dict[str, datetime] = {}
    
    def generate_token(self, session_id: str) -> str:
        """Generar token CSRF"""
        message = f"{session_id}{time.time()}".encode()
        token = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        self.tokens[token] = datetime.now() + timedelta(hours=1)
        return token
    
    def validate_token(self, token: str, session_id: str) -> bool:
        """Validar token CSRF"""
        if token not in self.tokens:
            return False
        
        if datetime.now() > self.tokens[token]:
            del self.tokens[token]
            return False
        
        return True
    
    def cleanup_expired(self):
        """Limpiar tokens expirados"""
        now = datetime.now()
        expired = [token for token, expiry in self.tokens.items() if now > expiry]
        for token in expired:
            del self.tokens[token]


class SecurityMiddleware:
    """Middleware de seguridad para FastAPI"""
    
    def __init__(
        self,
        rate_limit_config: Optional[RateLimitConfig] = None,
        enable_csrf: bool = False,
        secret_key: Optional[str] = None
    ):
        """
        Inicializar middleware de seguridad
        
        Args:
            rate_limit_config: Configuración de rate limiting
            enable_csrf: Habilitar protección CSRF
            secret_key: Clave secreta para CSRF
        """
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.enable_csrf = enable_csrf
        self.csrf_protection = CSRFProtection(secret_key or "default-secret-key") if enable_csrf else None
    
    async def __call__(self, request: Request, call_next: Callable):
        """Ejecutar middleware"""
        if not FASTAPI_AVAILABLE:
            return await call_next(request)
        
        # Rate limiting
        allowed, reason = self.rate_limiter.is_allowed(request=request)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=reason or "Rate limit exceeded"
            )
        
        # CSRF protection (solo para métodos que modifican datos)
        if self.enable_csrf and request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            token = request.headers.get("X-CSRF-Token")
            session_id = request.headers.get("X-Session-ID", "default")
            
            if not token or not self.csrf_protection.validate_token(token, session_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or missing CSRF token"
                )
        
        # Continuar con la petición
        response = await call_next(request)
        
        # Agregar headers de seguridad
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


# Instancias globales
_rate_limiter: Optional[RateLimiter] = None
_input_validator = InputValidator()


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Obtener instancia global de rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config or RateLimitConfig())
    return _rate_limiter


def get_input_validator() -> InputValidator:
    """Obtener instancia global de validador"""
    return _input_validator




