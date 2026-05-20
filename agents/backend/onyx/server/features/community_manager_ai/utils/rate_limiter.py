"""
Rate Limiter - Limitador de Tasa
==================================

Sistema para limitar la tasa de requests a APIs.
"""

import logging
import time
from typing import Dict, List, Optional
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """Limitador de tasa para APIs"""
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Inicializar rate limiter
        
        Args:
            requests_per_minute: Número máximo de requests por minuto
        """
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = Lock()
        logger.info(f"Rate Limiter inicializado: {requests_per_minute} req/min")
    
    def is_allowed(self, key: str) -> bool:
        """
        Verificar si un request está permitido
        
        Args:
            key: Clave única (ej: platform name)
            
        Returns:
            True si está permitido
        """
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Limpiar requests antiguos
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > minute_ago
            ]
            
            # Verificar límite
            if len(self.requests[key]) >= self.requests_per_minute:
                logger.warning(f"Rate limit alcanzado para {key}")
                return False
            
            # Registrar request
            self.requests[key].append(now)
            return True
    
    def wait_if_needed(self, key: str):
        """
        Esperar si es necesario para respetar el rate limit
        
        Args:
            key: Clave única
        """
        if not self.is_allowed(key):
            # Calcular tiempo de espera
            if key in self.requests and self.requests[key]:
                oldest_request = min(self.requests[key])
                wait_time = 60 - (time.time() - oldest_request) + 1
                
                if wait_time > 0:
                    logger.info(f"Esperando {wait_time:.2f}s para rate limit de {key}")
                    time.sleep(wait_time)
    
    def reset(self, key: Optional[str] = None):
        """
        Resetear rate limiter
        
        Args:
            key: Clave específica o None para resetear todas
        """
        with self.lock:
            if key:
                self.requests[key] = []
            else:
                self.requests.clear()
            logger.info(f"Rate limiter reseteado para {key or 'todas las claves'}")
    
    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas del rate limiter
        
        Args:
            key: Clave específica o None para todas
            
        Returns:
            Dict con estadísticas
        """
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            if key:
                requests = [
                    req_time for req_time in self.requests.get(key, [])
                    if req_time > minute_ago
                ]
                return {
                    "key": key,
                    "requests_last_minute": len(requests),
                    "limit": self.requests_per_minute,
                    "remaining": max(0, self.requests_per_minute - len(requests))
                }
            else:
                total_requests = 0
                keys_stats = {}
                
                for k, req_list in self.requests.items():
                    recent = [r for r in req_list if r > minute_ago]
                    keys_stats[k] = {
                        "requests": len(recent),
                        "remaining": max(0, self.requests_per_minute - len(recent))
                    }
                    total_requests += len(recent)
                
                return {
                    "total_keys": len(self.requests),
                    "total_requests_last_minute": total_requests,
                    "limit_per_key": self.requests_per_minute,
                    "keys": keys_stats
                }
    
    def get_wait_time(self, key: str) -> float:
        """
        Obtener tiempo de espera estimado para una clave
        
        Args:
            key: Clave única
            
        Returns:
            Tiempo de espera en segundos (0 si no hay que esperar)
        """
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            recent_requests = [
                req_time for req_time in self.requests.get(key, [])
                if req_time > minute_ago
            ]
            
            if len(recent_requests) < self.requests_per_minute:
                return 0.0
            
            oldest_request = min(recent_requests)
            wait_time = 60 - (now - oldest_request) + 1
            
            return max(0.0, wait_time)

