"""
Rate Limiter
============

Rate limiting para protección de API.
"""

import logging
import time
from typing import Dict, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter con sliding window."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        per_user: bool = True
    ):
        """
        Inicializar rate limiter.
        
        Args:
            max_requests: Máximo de requests por ventana
            window_seconds: Duración de ventana en segundos
            per_user: Limitar por usuario
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.per_user = per_user
        
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self._logger = logger
    
    def is_allowed(
        self,
        identifier: Optional[str] = None,
        request_time: Optional[float] = None
    ) -> tuple[bool, Optional[Dict[str, any]]]:
        """
        Verificar si request está permitido.
        
        Args:
            identifier: Identificador (user_id, IP, etc.)
            request_time: Tiempo del request (opcional)
        
        Returns:
            (allowed, info)
        """
        if not self.per_user:
            identifier = "global"
        elif identifier is None:
            identifier = "anonymous"
        
        if request_time is None:
            request_time = time.time()
        
        # Limpiar requests antiguos
        window_start = request_time - self.window_seconds
        user_requests = self.requests[identifier]
        
        while user_requests and user_requests[0] < window_start:
            user_requests.popleft()
        
        # Verificar límite
        if len(user_requests) >= self.max_requests:
            retry_after = self.window_seconds - (request_time - user_requests[0])
            return False, {
                "limit": self.max_requests,
                "remaining": 0,
                "reset_at": datetime.fromtimestamp(request_time + retry_after).isoformat(),
                "retry_after": retry_after
            }
        
        # Registrar request
        user_requests.append(request_time)
        
        return True, {
            "limit": self.max_requests,
            "remaining": self.max_requests - len(user_requests),
            "reset_at": datetime.fromtimestamp(request_time + self.window_seconds).isoformat()
        }
    
    def get_stats(self, identifier: Optional[str] = None) -> Dict[str, any]:
        """
        Obtener estadísticas.
        
        Args:
            identifier: Identificador
        
        Returns:
            Estadísticas
        """
        if not self.per_user:
            identifier = "global"
        elif identifier is None:
            identifier = "anonymous"
        
        user_requests = self.requests.get(identifier, deque())
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Contar requests en ventana
        recent_requests = sum(1 for req_time in user_requests if req_time >= window_start)
        
        return {
            "identifier": identifier,
            "requests_in_window": recent_requests,
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "remaining": max(0, self.max_requests - recent_requests)
        }




