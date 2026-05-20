"""
MCP Adaptive Rate Limiting - Rate limiting adaptativo
=======================================================
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """
    Rate limiter adaptativo
    
    Ajusta límites automáticamente según carga del sistema.
    """
    
    def __init__(
        self,
        base_requests_per_minute: int = 60,
        min_requests_per_minute: int = 10,
        max_requests_per_minute: int = 1000,
    ):
        """
        Args:
            base_requests_per_minute: Límite base
            min_requests_per_minute: Límite mínimo
            max_requests_per_minute: Límite máximo
        """
        self.base_requests_per_minute = base_requests_per_minute
        self.min_requests_per_minute = min_requests_per_minute
        self.max_requests_per_minute = max_requests_per_minute
        
        self.current_limit = base_requests_per_minute
        self._limiter = RateLimiter(requests_per_minute=self.current_limit)
        self._system_load_history: deque = deque(maxlen=100)
    
    def update_system_load(self, load: float):
        """
        Actualiza carga del sistema
        
        Args:
            load: Carga del sistema (0.0 - 1.0)
        """
        self._system_load_history.append(load)
        
        # Calcular carga promedio
        if self._system_load_history:
            avg_load = sum(self._system_load_history) / len(self._system_load_history)
            
            # Ajustar límite según carga
            if avg_load > 0.8:
                # Alta carga, reducir límite
                self.current_limit = max(
                    self.min_requests_per_minute,
                    int(self.current_limit * 0.9)
                )
            elif avg_load < 0.3:
                # Baja carga, aumentar límite
                self.current_limit = min(
                    self.max_requests_per_minute,
                    int(self.current_limit * 1.1)
                )
            
            # Actualizar limiter
            self._limiter = RateLimiter(requests_per_minute=self.current_limit)
            
            logger.debug(f"Adaptive rate limit adjusted: {self.current_limit} (load: {avg_load:.2f})")
    
    def check_rate_limit(self) -> tuple[bool, int]:
        """
        Verifica rate limit
        
        Returns:
            Tuple (allowed, remaining)
        """
        return self._limiter.check_rate_limit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        avg_load = (
            sum(self._system_load_history) / len(self._system_load_history)
            if self._system_load_history else 0
        )
        
        return {
            "current_limit": self.current_limit,
            "base_limit": self.base_requests_per_minute,
            "min_limit": self.min_requests_per_minute,
            "max_limit": self.max_requests_per_minute,
            "average_system_load": avg_load,
            "adjustment_factor": self.current_limit / self.base_requests_per_minute,
        }

