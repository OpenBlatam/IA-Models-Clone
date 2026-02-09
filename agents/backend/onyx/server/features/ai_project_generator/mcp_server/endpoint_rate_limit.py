"""
MCP Endpoint Rate Limiting - Rate limiting por endpoint
========================================================
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class EndpointRateLimiter:
    """
    Rate limiter por endpoint
    
    Permite límites diferentes para cada endpoint.
    """
    
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._default_limits: Dict[str, int] = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
        }
    
    def set_endpoint_limits(
        self,
        endpoint: str,
        requests_per_minute: int,
        requests_per_hour: Optional[int] = None,
    ):
        """
        Configura límites para un endpoint
        
        Args:
            endpoint: Path del endpoint (ej: "/mcp/v1/resources")
            requests_per_minute: Requests por minuto
            requests_per_hour: Requests por hora (opcional)
        """
        limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour or requests_per_minute * 60,
        )
        
        self._limiters[endpoint] = limiter
        logger.info(f"Set rate limits for endpoint {endpoint}: {requests_per_minute}/min")
    
    def check_rate_limit(self, endpoint: str) -> tuple[bool, Optional[str]]:
        """
        Verifica rate limit para un endpoint
        
        Args:
            endpoint: Path del endpoint
            
        Returns:
            Tuple (allowed, error_message)
        """
        limiter = self._limiters.get(endpoint)
        
        if not limiter:
            # Usar límites por defecto
            limiter = RateLimiter(
                requests_per_minute=self._default_limits["requests_per_minute"],
                requests_per_hour=self._default_limits["requests_per_hour"],
            )
            self._limiters[endpoint] = limiter
        
        allowed, remaining = limiter.check_rate_limit()
        
        if not allowed:
            return False, f"Rate limit exceeded for endpoint {endpoint}"
        
        return True, None
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rate limit para un endpoint
        
        Args:
            endpoint: Path del endpoint
            
        Returns:
            Diccionario con estadísticas
        """
        limiter = self._limiters.get(endpoint)
        
        if not limiter:
            return {
                "endpoint": endpoint,
                "limits": self._default_limits,
                "usage": {},
            }
        
        return {
            "endpoint": endpoint,
            "limits": limiter.get_limits(),
            "usage": limiter.get_usage_stats(),
        }

