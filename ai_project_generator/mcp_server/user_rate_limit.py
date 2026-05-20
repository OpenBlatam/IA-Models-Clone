"""
MCP User Rate Limiting - Rate limiting por usuario
===================================================
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class UserRateLimiter:
    """
    Rate limiter por usuario
    
    Permite límites de rate diferentes por usuario.
    """
    
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._default_limits: Dict[str, int] = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
        }
    
    def set_user_limits(
        self,
        user_id: str,
        requests_per_minute: int,
        requests_per_hour: Optional[int] = None,
    ):
        """
        Configura límites para un usuario
        
        Args:
            user_id: ID del usuario
            requests_per_minute: Requests por minuto
            requests_per_hour: Requests por hora (opcional)
        """
        limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour or requests_per_minute * 60,
        )
        
        self._limiters[user_id] = limiter
        logger.info(f"Set rate limits for user {user_id}: {requests_per_minute}/min")
    
    def check_rate_limit(self, user_id: str) -> tuple[bool, Optional[str]]:
        """
        Verifica rate limit para un usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Tuple (allowed, error_message)
        """
        limiter = self._limiters.get(user_id)
        
        if not limiter:
            # Usar límites por defecto
            limiter = RateLimiter(
                requests_per_minute=self._default_limits["requests_per_minute"],
                requests_per_hour=self._default_limits["requests_per_hour"],
            )
            self._limiters[user_id] = limiter
        
        allowed, remaining = limiter.check_rate_limit()
        
        if not allowed:
            return False, f"Rate limit exceeded for user {user_id}"
        
        return True, None
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rate limit para un usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Diccionario con estadísticas
        """
        limiter = self._limiters.get(user_id)
        
        if not limiter:
            return {
                "user_id": user_id,
                "limits": self._default_limits,
                "usage": {},
            }
        
        return {
            "user_id": user_id,
            "limits": limiter.get_limits(),
            "usage": limiter.get_usage_stats(),
        }

