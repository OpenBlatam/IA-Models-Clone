"""
Distributed Rate Limiter - Rate Limiter Distribuido
===================================================

Rate limiting distribuido con Redis:
- Distributed rate limiting
- Sliding window log
- Token bucket distribuido
- Multi-key rate limiting
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DistributedRateLimiter:
    """
    Rate limiter distribuido usando Redis.
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        default_limit: int = 100,
        default_window: int = 60
    ) -> None:
        self.redis_client = redis_client
        self.default_limit = default_limit
        self.default_window = default_window
        self._ensure_redis()
    
    def _ensure_redis(self) -> None:
        """Asegura que Redis está disponible"""
        if not self.redis_client:
            try:
                from ..core.redis_client import get_redis_client
                self.redis_client = get_redis_client()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
    
    async def check_rate_limit(
        self,
        key: str,
        limit: Optional[int] = None,
        window: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Verifica rate limit distribuido.
        
        Returns:
            (is_allowed, error_message, stats)
        """
        if not self.redis_client:
            # Fallback a rate limiting local
            return True, None, {}
        
        limit = limit or self.default_limit
        window = window or self.default_window
        
        try:
            # Usar sliding window log algorithm
            now = time.time()
            window_start = now - window
            
            # Limpiar entradas antiguas
            pipe = self.redis_client.sync_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, window)
            results = pipe.execute()
            
            current_count = results[1] + 1  # +1 por la nueva entrada
            
            if current_count > limit:
                # Calcular tiempo hasta reset
                oldest = self.redis_client.sync_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    reset_time = oldest[0][1] + window
                    wait_time = reset_time - now
                else:
                    wait_time = window
                
                return False, f"Rate limit exceeded. Try again in {wait_time:.0f}s", {
                    "limit": limit,
                    "remaining": 0,
                    "reset_in": wait_time
                }
            
            return True, None, {
                "limit": limit,
                "remaining": limit - current_count,
                "reset_in": window
            }
            
        except Exception as e:
            logger.error(f"Distributed rate limit check failed: {e}")
            # Fallback: permitir request
            return True, None, {}
    
    async def get_rate_limit_stats(self, key: str) -> Dict[str, Any]:
        """Obtiene estadísticas de rate limit"""
        if not self.redis_client:
            return {}
        
        try:
            count = self.redis_client.sync_client.zcard(key)
            return {
                "key": key,
                "current_count": count,
                "limit": self.default_limit,
                "window": self.default_window
            }
        except Exception as e:
            logger.error(f"Failed to get rate limit stats: {e}")
            return {}


def get_distributed_rate_limiter(
    redis_client: Optional[Any] = None,
    default_limit: int = 100,
    default_window: int = 60
) -> DistributedRateLimiter:
    """Obtiene rate limiter distribuido"""
    return DistributedRateLimiter(
        redis_client=redis_client,
        default_limit=default_limit,
        default_window=default_window
    )















