"""
Rate Limiter - Control de tasa de peticiones
=============================================
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Limita la tasa de peticiones por usuario/IP.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000
    ):
        """
        Inicializar rate limiter.
        
        Args:
            requests_per_minute: Peticiones por minuto
            requests_per_hour: Peticiones por hora
            requests_per_day: Peticiones por día
        """
        self.limits = {
            "minute": requests_per_minute,
            "hour": requests_per_hour,
            "day": requests_per_day
        }
        
        # Almacenamiento de peticiones por identificador
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Verifica si una petición está permitida.
        
        Args:
            identifier: Identificador único (IP, user_id, etc.)
            
        Returns:
            (allowed, reason) - Si está permitido y razón si no
        """
        now = datetime.now()
        
        # Limpiar peticiones antiguas
        self._cleanup_old_requests(identifier, now)
        
        # Obtener peticiones recientes
        recent_requests = self.requests[identifier]
        
        # Verificar límites
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        requests_last_minute = sum(1 for r in recent_requests if r > minute_ago)
        requests_last_hour = sum(1 for r in recent_requests if r > hour_ago)
        requests_last_day = sum(1 for r in recent_requests if r > day_ago)
        
        # Verificar límites
        if requests_last_minute >= self.limits["minute"]:
            return False, f"Rate limit exceeded: {requests_last_minute}/{self.limits['minute']} requests per minute"
        
        if requests_last_hour >= self.limits["hour"]:
            return False, f"Rate limit exceeded: {requests_last_hour}/{self.limits['hour']} requests per hour"
        
        if requests_last_day >= self.limits["day"]:
            return False, f"Rate limit exceeded: {requests_last_day}/{self.limits['day']} requests per day"
        
        # Registrar petición
        recent_requests.append(now)
        
        return True, None
    
    def _cleanup_old_requests(self, identifier: str, now: datetime):
        """Limpia peticiones antiguas (más de 1 día)"""
        day_ago = now - timedelta(days=1)
        self.requests[identifier] = [
            r for r in self.requests[identifier]
            if r > day_ago
        ]
    
    def get_remaining(self, identifier: str) -> Dict[str, Any]:
        """
        Obtiene peticiones restantes.
        
        Args:
            identifier: Identificador único
            
        Returns:
            Peticiones restantes por período
        """
        now = datetime.now()
        self._cleanup_old_requests(identifier, now)
        
        recent_requests = self.requests[identifier]
        
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        requests_last_minute = sum(1 for r in recent_requests if r > minute_ago)
        requests_last_hour = sum(1 for r in recent_requests if r > hour_ago)
        requests_last_day = sum(1 for r in recent_requests if r > day_ago)
        
        return {
            "minute": {
                "used": requests_last_minute,
                "limit": self.limits["minute"],
                "remaining": max(0, self.limits["minute"] - requests_last_minute)
            },
            "hour": {
                "used": requests_last_hour,
                "limit": self.limits["hour"],
                "remaining": max(0, self.limits["hour"] - requests_last_hour)
            },
            "day": {
                "used": requests_last_day,
                "limit": self.limits["day"],
                "remaining": max(0, self.limits["day"] - requests_last_day)
            }
        }
    
    def reset(self, identifier: Optional[str] = None):
        """
        Resetea contadores.
        
        Args:
            identifier: Identificador específico (None = todos)
        """
        if identifier:
            if identifier in self.requests:
                del self.requests[identifier]
        else:
            self.requests.clear()
        
        logger.info(f"Rate limiter reseteado para: {identifier or 'todos'}")

