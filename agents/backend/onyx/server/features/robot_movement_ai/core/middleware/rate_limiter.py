"""
Rate Limiter System
===================

Sistema de limitación de tasa.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Límite de tasa."""
    max_requests: int
    window_seconds: float
    current_requests: int = 0
    window_start: float = 0.0


class RateLimiter:
    """
    Limitador de tasa.
    
    Limita el número de solicitudes por período de tiempo.
    """
    
    def __init__(self):
        """Inicializar limitador de tasa."""
        self.limits: Dict[str, RateLimit] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
    
    def add_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: float
    ) -> None:
        """
        Agregar límite de tasa.
        
        Args:
            key: Clave del límite (ej: "user:123", "ip:192.168.1.1")
            max_requests: Máximo de solicitudes
            window_seconds: Ventana de tiempo en segundos
        """
        self.limits[key] = RateLimit(
            max_requests=max_requests,
            window_seconds=window_seconds,
            window_start=time.time()
        )
        logger.info(f"Added rate limit: {key} ({max_requests} req/{window_seconds}s)")
    
    def is_allowed(self, key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verificar si se permite la solicitud.
        
        Args:
            key: Clave del límite
            
        Returns:
            (allowed, info) - Si está permitido y información
        """
        if key not in self.limits:
            return True, None
        
        limit = self.limits[key]
        now = time.time()
        
        # Limpiar historial fuera de la ventana
        history = self.request_history[key]
        while history and history[0] < now - limit.window_seconds:
            history.popleft()
        
        # Verificar límite
        if len(history) >= limit.max_requests:
            reset_time = history[0] + limit.window_seconds
            return False, {
                "limit_exceeded": True,
                "max_requests": limit.max_requests,
                "current_requests": len(history),
                "window_seconds": limit.window_seconds,
                "reset_at": reset_time,
                "retry_after": max(0, reset_time - now)
            }
        
        # Registrar solicitud
        history.append(now)
        limit.current_requests = len(history)
        
        return True, {
            "limit_exceeded": False,
            "max_requests": limit.max_requests,
            "current_requests": len(history),
            "remaining": limit.max_requests - len(history),
            "window_seconds": limit.window_seconds
        }
    
    def reset_limit(self, key: str) -> None:
        """Resetear límite."""
        if key in self.request_history:
            self.request_history[key].clear()
        if key in self.limits:
            self.limits[key].current_requests = 0
            self.limits[key].window_start = time.time()
    
    def get_limit_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Obtener información de límite."""
        if key not in self.limits:
            return None
        
        limit = self.limits[key]
        history = self.request_history[key]
        now = time.time()
        
        # Limpiar historial
        while history and history[0] < now - limit.window_seconds:
            history.popleft()
        
        return {
            "max_requests": limit.max_requests,
            "current_requests": len(history),
            "remaining": limit.max_requests - len(history),
            "window_seconds": limit.window_seconds,
            "window_start": limit.window_start
        }


# Instancia global
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Obtener instancia global del limitador de tasa."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

