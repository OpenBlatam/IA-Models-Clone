"""
MCP Throttling - Throttling avanzado
=====================================
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class ThrottleConfig:
    """Configuración de throttling"""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        burst_size: int = 10,
    ):
        """
        Args:
            max_requests: Máximo de requests en la ventana
            window_seconds: Tamaño de la ventana en segundos
            burst_size: Tamaño de burst permitido
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_size = burst_size


class Throttler:
    """
    Throttler avanzado
    
    Implementa throttling con ventana deslizante y control de burst.
    """
    
    def __init__(self, config: ThrottleConfig):
        """
        Args:
            config: Configuración de throttling
        """
        self.config = config
        self._request_times: deque = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Intenta adquirir permiso para procesar request
        
        Returns:
            True si se permite, False si está throttled
        """
        async with self._lock:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.config.window_seconds)
            
            # Limpiar requests fuera de la ventana
            while self._request_times and self._request_times[0] < window_start:
                self._request_times.popleft()
            
            # Verificar límite
            if len(self._request_times) >= self.config.max_requests:
                return False
            
            # Verificar burst
            recent_requests = sum(
                1 for t in self._request_times
                if t > now - timedelta(seconds=1)
            )
            
            if recent_requests >= self.config.burst_size:
                return False
            
            # Permitir request
            self._request_times.append(now)
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del throttler
        
        Returns:
            Diccionario con estadísticas
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.config.window_seconds)
        
        requests_in_window = sum(
            1 for t in self._request_times
            if t >= window_start
        )
        
        return {
            "max_requests": self.config.max_requests,
            "window_seconds": self.config.window_seconds,
            "requests_in_window": requests_in_window,
            "remaining": max(0, self.config.max_requests - requests_in_window),
            "utilization": requests_in_window / self.config.max_requests,
        }


class AdaptiveThrottler(Throttler):
    """
    Throttler adaptativo
    
    Ajusta límites automáticamente según carga del sistema.
    """
    
    def __init__(self, config: ThrottleConfig):
        super().__init__(config)
        self._base_config = config
        self._adjustment_factor = 1.0
    
    async def adjust_limits(self, system_load: float):
        """
        Ajusta límites según carga del sistema
        
        Args:
            system_load: Carga del sistema (0.0 - 1.0)
        """
        if system_load > 0.8:
            # Alta carga, reducir límites
            self._adjustment_factor = 0.7
        elif system_load < 0.3:
            # Baja carga, aumentar límites
            self._adjustment_factor = 1.3
        else:
            self._adjustment_factor = 1.0
        
        self.config.max_requests = int(
            self._base_config.max_requests * self._adjustment_factor
        )
        
        logger.info(f"Adjusted throttling limits: {self.config.max_requests} requests")

