"""
Rate Limiting y Throttling
===========================

Sistema para limitar la tasa de peticiones y prevenir abuso.
"""

import time
import logging
from typing import Dict, Optional, Callable
from collections import defaultdict, deque
from threading import Lock
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimiter:
    """Limitador de tasa de peticiones"""
    
    def __init__(
        self,
        max_requests: int = 100,
        time_window: int = 60,
        max_burst: Optional[int] = None
    ):
        """
        Inicializar rate limiter
        
        Args:
            max_requests: Número máximo de peticiones
            time_window: Ventana de tiempo en segundos
            max_burst: Tamaño máximo de burst (opcional)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.max_burst = max_burst or max_requests
        
        # Almacenar timestamps de peticiones por identificador
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = Lock()
    
    def is_allowed(self, identifier: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verificar si una petición está permitida
        
        Args:
            identifier: Identificador único (IP, user_id, etc.)
        
        Returns:
            Tupla (allowed, info_dict)
        """
        now = time.time()
        
        with self.lock:
            # Limpiar peticiones antiguas
            requests = self.requests[identifier]
            while requests and requests[0] < now - self.time_window:
                requests.popleft()
            
            # Verificar límite
            if len(requests) >= self.max_requests:
                oldest_request = requests[0]
                retry_after = int(oldest_request + self.time_window - now)
                
                return False, {
                    "limit": self.max_requests,
                    "remaining": 0,
                    "reset_at": datetime.fromtimestamp(now + retry_after).isoformat(),
                    "retry_after": retry_after
                }
            
            # Permitir petición
            requests.append(now)
            
            return True, {
                "limit": self.max_requests,
                "remaining": self.max_requests - len(requests),
                "reset_at": datetime.fromtimestamp(now + self.time_window).isoformat()
            }
    
    def get_status(self, identifier: str) -> Dict[str, Any]:
        """Obtener estado actual del rate limiter"""
        now = time.time()
        
        with self.lock:
            requests = self.requests[identifier]
            # Limpiar peticiones antiguas
            while requests and requests[0] < now - self.time_window:
                requests.popleft()
            
            return {
                "limit": self.max_requests,
                "remaining": max(0, self.max_requests - len(requests)),
                "used": len(requests),
                "reset_at": datetime.fromtimestamp(now + self.time_window).isoformat()
            }
    
    def reset(self, identifier: Optional[str] = None):
        """Resetear contador para un identificador o todos"""
        with self.lock:
            if identifier:
                if identifier in self.requests:
                    del self.requests[identifier]
            else:
                self.requests.clear()


class TokenBucket:
    """Implementación de Token Bucket para rate limiting"""
    
    def __init__(
        self,
        capacity: int = 100,
        refill_rate: float = 1.0
    ):
        """
        Inicializar Token Bucket
        
        Args:
            capacity: Capacidad máxima del bucket
            refill_rate: Tokens por segundo
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens: Dict[str, float] = defaultdict(lambda: capacity)
        self.last_refill: Dict[str, float] = defaultdict(time.time)
        self.lock = Lock()
    
    def consume(self, identifier: str, tokens: int = 1) -> bool:
        """
        Consumir tokens
        
        Args:
            identifier: Identificador único
            tokens: Número de tokens a consumir
        
        Returns:
            True si hay suficientes tokens, False si no
        """
        now = time.time()
        
        with self.lock:
            # Refill tokens
            last_refill = self.last_refill[identifier]
            elapsed = now - last_refill
            current_tokens = self.tokens[identifier]
            
            # Agregar tokens basado en refill rate
            new_tokens = min(
                self.capacity,
                current_tokens + elapsed * self.refill_rate
            )
            
            self.tokens[identifier] = new_tokens
            self.last_refill[identifier] = now
            
            # Verificar si hay suficientes tokens
            if new_tokens >= tokens:
                self.tokens[identifier] -= tokens
                return True
            else:
                return False
    
    def get_tokens(self, identifier: str) -> float:
        """Obtener número de tokens disponibles"""
        now = time.time()
        
        with self.lock:
            last_refill = self.last_refill[identifier]
            elapsed = now - last_refill
            current_tokens = self.tokens[identifier]
            
            new_tokens = min(
                self.capacity,
                current_tokens + elapsed * self.refill_rate
            )
            
            self.tokens[identifier] = new_tokens
            self.last_refill[identifier] = now
            
            return new_tokens


def rate_limit(
    max_requests: int = 100,
    time_window: int = 60,
    get_identifier: Optional[Callable] = None
):
    """
    Decorator para rate limiting
    
    Args:
        max_requests: Número máximo de peticiones
        time_window: Ventana de tiempo en segundos
        get_identifier: Función para obtener identificador (por defecto usa request.client.host)
    """
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Obtener identificador
            if get_identifier:
                identifier = get_identifier(*args, **kwargs)
            else:
                # Intentar obtener de request
                request = kwargs.get("request") or (args[0] if args else None)
                if hasattr(request, "client"):
                    identifier = request.client.host if request.client else "unknown"
                else:
                    identifier = "default"
            
            # Verificar rate limit
            allowed, info = limiter.is_allowed(identifier)
            
            if not allowed:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": info.get("retry_after"),
                        "limit": info.get("limit")
                    },
                    headers={"Retry-After": str(info.get("retry_after", 60))}
                )
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Agregar headers de rate limit
            if hasattr(result, "headers"):
                result.headers["X-RateLimit-Limit"] = str(info["limit"])
                result.headers["X-RateLimit-Remaining"] = str(info["remaining"])
                result.headers["X-RateLimit-Reset"] = info["reset_at"]
            
            return result
        
        return wrapper
    return decorator


# Instancias globales
_default_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(
    max_requests: int = 100,
    time_window: int = 60
) -> RateLimiter:
    """Obtener instancia global de RateLimiter"""
    global _default_rate_limiter
    if _default_rate_limiter is None:
        _default_rate_limiter = RateLimiter(max_requests, time_window)
    return _default_rate_limiter
















