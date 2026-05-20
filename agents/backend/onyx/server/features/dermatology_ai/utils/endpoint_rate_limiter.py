"""
Rate limiting por endpoint
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time


class EndpointRateLimiter:
    """Rate limiter por endpoint"""
    
    def __init__(self):
        """Inicializa el rate limiter"""
        self.limits: Dict[str, Dict] = {}  # endpoint -> {max_requests, window_seconds}
        self.requests: Dict[str, list] = defaultdict(list)  # endpoint -> [timestamps]
    
    def set_limit(self, endpoint: str, max_requests: int, window_seconds: int):
        """
        Establece límite para un endpoint
        
        Args:
            endpoint: Endpoint
            max_requests: Máximo de requests
            window_seconds: Ventana en segundos
        """
        self.limits[endpoint] = {
            "max_requests": max_requests,
            "window_seconds": window_seconds
        }
    
    def is_allowed(self, endpoint: str, identifier: str = "default") -> tuple[bool, Optional[Dict]]:
        """
        Verifica si un request está permitido
        
        Args:
            endpoint: Endpoint
            identifier: Identificador único
            
        Returns:
            Tupla (permitido, info)
        """
        if endpoint not in self.limits:
            return True, None
        
        limit_config = self.limits[endpoint]
        max_requests = limit_config["max_requests"]
        window_seconds = limit_config["window_seconds"]
        
        key = f"{endpoint}:{identifier}"
        now = time.time()
        window_start = now - window_seconds
        
        # Limpiar requests antiguos
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        # Verificar límite
        if len(self.requests[key]) >= max_requests:
            oldest_request = min(self.requests[key]) if self.requests[key] else now
            reset_time = oldest_request + window_seconds
            
            return False, {
                "limit": max_requests,
                "remaining": 0,
                "reset_at": datetime.fromtimestamp(reset_time).isoformat(),
                "retry_after": int(reset_time - now)
            }
        
        # Agregar request
        self.requests[key].append(now)
        
        return True, {
            "limit": max_requests,
            "remaining": max_requests - len(self.requests[key]),
            "reset_at": datetime.fromtimestamp(now + window_seconds).isoformat()
        }
    
    def get_endpoint_stats(self, endpoint: str) -> Dict:
        """Obtiene estadísticas de un endpoint"""
        if endpoint not in self.limits:
            return {}
        
        limit_config = self.limits[endpoint]
        window_seconds = limit_config["window_seconds"]
        now = time.time()
        window_start = now - window_seconds
        
        # Contar requests en ventana
        requests_in_window = sum(
            len([r for r in reqs if r > window_start])
            for key, reqs in self.requests.items()
            if key.startswith(f"{endpoint}:")
        )
        
        return {
            "endpoint": endpoint,
            "limit": limit_config["max_requests"],
            "window_seconds": window_seconds,
            "requests_in_window": requests_in_window,
            "utilization": (requests_in_window / limit_config["max_requests"] * 100) if limit_config["max_requests"] > 0 else 0
        }






