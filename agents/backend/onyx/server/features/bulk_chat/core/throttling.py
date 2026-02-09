"""
Throttling - Sistema de Throttling
==================================

Sistema de throttling para controlar el ritmo de procesamiento.
"""

import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ThrottleConfig:
    """Configuración de throttling."""
    max_requests: int
    window_seconds: float
    identifier: Optional[str] = None


class Throttler:
    """Gestor de throttling."""
    
    def __init__(self):
        self.requests: Dict[str, deque] = {}
        self.configs: Dict[str, ThrottleConfig] = {}
        self._lock = asyncio.Lock()
    
    def configure(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: float,
    ):
        """Configurar throttling para identificador."""
        config = ThrottleConfig(
            max_requests=max_requests,
            window_seconds=window_seconds,
            identifier=identifier,
        )
        self.configs[identifier] = config
        logger.info(f"Configured throttling for {identifier}: {max_requests} requests per {window_seconds}s")
    
    async def check_throttle(self, identifier: str) -> bool:
        """
        Verificar si se permite la solicitud.
        
        Args:
            identifier: Identificador (usuario, IP, etc.)
        
        Returns:
            True si está permitido, False si está throttled
        """
        config = self.configs.get(identifier)
        if not config:
            return True  # Sin throttling configurado
        
        async with self._lock:
            now = datetime.now()
            
            if identifier not in self.requests:
                self.requests[identifier] = deque()
            
            request_times = self.requests[identifier]
            
            # Limpiar requests fuera de la ventana
            cutoff = now - timedelta(seconds=config.window_seconds)
            while request_times and request_times[0] < cutoff:
                request_times.popleft()
            
            # Verificar límite
            if len(request_times) >= config.max_requests:
                logger.warning(f"Throttled request from {identifier}")
                return False
            
            # Registrar solicitud
            request_times.append(now)
            return True
    
    async def get_throttle_status(self, identifier: str) -> Dict:
        """Obtener estado de throttling."""
        config = self.configs.get(identifier)
        if not config:
            return {"configured": False}
        
        async with self._lock:
            request_times = self.requests.get(identifier, deque())
            now = datetime.now()
            cutoff = now - timedelta(seconds=config.window_seconds)
            
            # Limpiar requests antiguos
            while request_times and request_times[0] < cutoff:
                request_times.popleft()
            
            return {
                "configured": True,
                "max_requests": config.max_requests,
                "window_seconds": config.window_seconds,
                "current_requests": len(request_times),
                "remaining": max(0, config.max_requests - len(request_times)),
                "reset_at": (request_times[0] + timedelta(seconds=config.window_seconds)).isoformat() if request_times else None,
            }
















