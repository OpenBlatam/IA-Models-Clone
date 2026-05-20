"""
Rate Limiter
============

Sistema de rate limiting.
"""

import logging
import time
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter."""
    
    def __init__(self):
        """Inicializar rate limiter."""
        self.requests: Dict[str, list] = defaultdict(list)
        self._logger = logger
    
    def is_allowed(
        self,
        key: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> tuple[bool, Optional[float]]:
        """
        Verificar si se permite la solicitud.
        
        Args:
            key: Clave única (usuario, IP, etc.)
            max_requests: Máximo de solicitudes
            window_seconds: Ventana de tiempo en segundos
        
        Returns:
            (permitido, tiempo_restante)
        """
        now = time.time()
        window_start = now - window_seconds
        
        # Limpiar solicitudes antiguas
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        # Verificar límite
        if len(self.requests[key]) >= max_requests:
            # Calcular tiempo hasta que expire la solicitud más antigua
            if self.requests[key]:
                oldest_request = min(self.requests[key])
                time_remaining = window_seconds - (now - oldest_request)
                return False, max(0, time_remaining)
            return False, window_seconds
        
        # Registrar solicitud
        self.requests[key].append(now)
        return True, None
    
    def get_stats(self, key: str, window_seconds: int = 60) -> Dict[str, Any]:
        """
        Obtener estadísticas de rate limiting.
        
        Args:
            key: Clave
            window_seconds: Ventana de tiempo
        
        Returns:
            Estadísticas
        """
        now = time.time()
        window_start = now - window_seconds
        
        requests_in_window = [
            req_time for req_time in self.requests.get(key, [])
            if req_time > window_start
        ]
        
        return {
            "key": key,
            "requests_count": len(requests_in_window),
            "window_seconds": window_seconds,
            "oldest_request": min(requests_in_window) if requests_in_window else None,
            "newest_request": max(requests_in_window) if requests_in_window else None
        }
    
    def reset(self, key: Optional[str] = None):
        """
        Resetear contador.
        
        Args:
            key: Clave específica o None para todas
        """
        if key:
            if key in self.requests:
                del self.requests[key]
        else:
            self.requests.clear()




