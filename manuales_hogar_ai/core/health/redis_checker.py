"""
Redis Checker
=============

Checker especializado para Redis.
"""

import asyncio
from typing import Dict, Any
from ...core.base.service_base import BaseService
from ...infrastructure.cache_redis import get_cache


class RedisChecker(BaseService):
    """Checker de Redis."""
    
    def __init__(self):
        """Inicializar checker."""
        super().__init__(logger_name=__name__)
    
    async def check(self) -> Dict[str, Any]:
        """
        Verificar conectividad de Redis.
        
        Returns:
            Diccionario con status y message
        """
        try:
            cache = await get_cache()
            if cache.client:
                await asyncio.wait_for(cache.client.ping(), timeout=2.0)
                return {
                    "status": "healthy",
                    "message": "Redis is accessible",
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Redis not configured, using memory store",
                }
        except asyncio.TimeoutError:
            return {
                "status": "degraded",
                "message": "Redis connection timeout",
            }
        except Exception as e:
            self.log_warning(f"Redis health check failed: {e}")
            return {
                "status": "degraded",
                "message": f"Redis unavailable: {str(e)}",
            }

