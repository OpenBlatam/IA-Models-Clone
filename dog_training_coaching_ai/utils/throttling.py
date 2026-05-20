"""
Throttling Utilities
====================
Utilidades para throttling de requests.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio


class Throttler:
    """Throttler para limitar frecuencia de operaciones."""
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: float
    ):
        """
        Inicializar throttler.
        
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
            key: Clave única (ej: user_id, ip)
            
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
    
    async def wait_if_needed(self, key: str):
        """
        Esperar si es necesario para cumplir con throttling.
        
        Args:
            key: Clave única
        """
        allowed, wait_time = await self.is_allowed(key)
        
        if not allowed and wait_time:
            await asyncio.sleep(wait_time)
            # Intentar de nuevo
            await self.is_allowed(key)


class AdaptiveThrottler:
    """Throttler adaptativo que ajusta límites dinámicamente."""
    
    def __init__(
        self,
        initial_max_requests: int,
        window_seconds: float,
        min_requests: int = 1,
        max_requests: int = 1000
    ):
        """
        Inicializar throttler adaptativo.
        
        Args:
            initial_max_requests: Requests iniciales permitidas
            window_seconds: Ventana de tiempo
            min_requests: Mínimo de requests
            max_requests: Máximo de requests
        """
        self.current_max = initial_max_requests
        self.min_requests = min_requests
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.throttler = Throttler(initial_max_requests, window_seconds)
        self.success_rate = 1.0
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> Tuple[bool, Optional[float]]:
        """Verificar si está permitido."""
        return await self.throttler.is_allowed(key)
    
    async def record_success(self):
        """Registrar éxito y ajustar límites."""
        async with self._lock:
            # Aumentar límite si hay muchos éxitos
            if self.success_rate > 0.95 and self.current_max < self.max_requests:
                self.current_max = min(self.max_requests, int(self.current_max * 1.1))
                self.throttler = Throttler(self.current_max, self.window_seconds)
    
    async def record_failure(self):
        """Registrar fallo y ajustar límites."""
        async with self._lock:
            # Reducir límite si hay muchos fallos
            if self.success_rate < 0.8 and self.current_max > self.min_requests:
                self.current_max = max(self.min_requests, int(self.current_max * 0.9))
                self.throttler = Throttler(self.current_max, self.window_seconds)

