"""
Sistema de rate limiting
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time


class RateLimiter:
    """Limita la tasa de requests por usuario/IP"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Inicializa el rate limiter
        
        Args:
            max_requests: Máximo número de requests por ventana
            window_seconds: Duración de la ventana en segundos
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> tuple[bool, Optional[Dict]]:
        """
        Verifica si un request está permitido
        
        Args:
            identifier: Identificador único (user_id, IP, etc.)
            
        Returns:
            Tupla (permitido, info)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Limpiar requests antiguos
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]
        
        # Verificar límite
        if len(self.requests[identifier]) >= self.max_requests:
            oldest_request = min(self.requests[identifier])
            reset_time = oldest_request + self.window_seconds
            
            return False, {
                "limit": self.max_requests,
                "remaining": 0,
                "reset_at": datetime.fromtimestamp(reset_time).isoformat(),
                "retry_after": int(reset_time - now)
            }
        
        # Agregar request
        self.requests[identifier].append(now)
        
        return True, {
            "limit": self.max_requests,
            "remaining": self.max_requests - len(self.requests[identifier]),
            "reset_at": datetime.fromtimestamp(now + self.window_seconds).isoformat()
        }
    
    def reset(self, identifier: str):
        """Resetea el contador para un identificador"""
        if identifier in self.requests:
            del self.requests[identifier]
    
    def get_stats(self, identifier: str) -> Dict:
        """
        Obtiene estadísticas de rate limiting
        
        Args:
            identifier: Identificador
            
        Returns:
            Diccionario con estadísticas
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        requests_in_window = [
            req_time for req_time in self.requests.get(identifier, [])
            if req_time > window_start
        ]
        
        return {
            "identifier": identifier,
            "requests_in_window": len(requests_in_window),
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - len(requests_in_window)),
            "window_seconds": self.window_seconds
        }






