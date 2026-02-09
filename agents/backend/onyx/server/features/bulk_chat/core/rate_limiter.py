"""
Rate Limiter - Control de tasa de solicitudes
==============================================

Sistema de rate limiting para prevenir abuso y controlar el uso de recursos.
"""

import asyncio
import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuración de rate limiting."""
    max_requests: int = 60  # Máximo de requests
    time_window: float = 60.0  # Ventana de tiempo en segundos
    max_concurrent: int = 10  # Máximo de requests concurrentes


class RateLimiter:
    """Rate limiter con sliding window."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.requests: Dict[str, list] = defaultdict(list)
        self.concurrent: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(
        self,
        identifier: str,
        increment: bool = True,
    ) -> tuple[bool, Optional[str]]:
        """
        Verificar si una solicitud está dentro del rate limit.
        
        Returns:
            (allowed, message) - Si está permitido y mensaje de error si no
        """
        async with self._lock:
            now = time.time()
            
            # Limpiar requests antiguos
            self.requests[identifier] = [
                req_time
                for req_time in self.requests[identifier]
                if now - req_time < self.config.time_window
            ]
            
            # Verificar límite de requests por ventana de tiempo
            if len(self.requests[identifier]) >= self.config.max_requests:
                return False, f"Rate limit exceeded: {self.config.max_requests} requests per {self.config.time_window}s"
            
            # Verificar límite de requests concurrentes
            if self.concurrent[identifier] >= self.config.max_concurrent:
                return False, f"Concurrent limit exceeded: {self.config.max_concurrent} concurrent requests"
            
            # Incrementar contadores si se permite
            if increment:
                self.requests[identifier].append(now)
                self.concurrent[identifier] += 1
            
            return True, None
    
    async def release(self, identifier: str):
        """Liberar un slot concurrente."""
        async with self._lock:
            if self.concurrent[identifier] > 0:
                self.concurrent[identifier] -= 1
    
    def get_stats(self, identifier: str) -> Dict:
        """Obtener estadísticas de rate limiting."""
        now = time.time()
        recent_requests = [
            req_time
            for req_time in self.requests[identifier]
            if now - req_time < self.config.time_window
        ]
        
        return {
            "requests_in_window": len(recent_requests),
            "max_requests": self.config.max_requests,
            "time_window": self.config.time_window,
            "concurrent": self.concurrent[identifier],
            "max_concurrent": self.config.max_concurrent,
        }
































