"""
Advanced Rate Limiting
======================
Utilidades avanzadas para rate limiting.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio


class TokenBucket:
    """Implementación de Token Bucket para rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Inicializar token bucket.
        
        Args:
            capacity: Capacidad máxima de tokens
            refill_rate: Tasa de relleno (tokens por segundo)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = datetime.now()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Consumir tokens.
        
        Args:
            tokens: Número de tokens a consumir
            
        Returns:
            True si se pueden consumir, False si no
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Rellenar tokens según el tiempo transcurrido."""
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    async def get_wait_time(self, tokens: int = 1) -> float:
        """
        Obtener tiempo de espera necesario.
        
        Args:
            tokens: Número de tokens necesarios
            
        Returns:
            Segundos a esperar
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0.0
            
            needed = tokens - self.tokens
            return needed / self.refill_rate


class SlidingWindowRateLimiter:
    """Rate limiter con ventana deslizante."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Inicializar rate limiter.
        
        Args:
            max_requests: Máximo de requests permitidas
            window_seconds: Ventana de tiempo en segundos
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> Tuple[bool, Optional[float]]:
        """
        Verificar si una request está permitida.
        
        Args:
            key: Clave única (ej: IP, user_id)
            
        Returns:
            Tupla (permitido, tiempo_restante)
        """
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)
            
            # Limpiar requests antiguas
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > cutoff
            ]
            
            if len(self.requests[key]) < self.max_requests:
                self.requests[key].append(now)
                return (True, None)
            
            # Calcular tiempo hasta que se libere un slot
            oldest_request = min(self.requests[key])
            wait_time = (oldest_request + timedelta(seconds=self.window_seconds) - now).total_seconds()
            
            return (False, max(0, wait_time))
    
    async def get_stats(self, key: str) -> Dict:
        """Obtener estadísticas para una clave."""
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)
            
            recent_requests = [
                req_time for req_time in self.requests.get(key, [])
                if req_time > cutoff
            ]
            
            return {
                "requests": len(recent_requests),
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
                "remaining": max(0, self.max_requests - len(recent_requests))
            }

