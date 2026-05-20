"""
Cache Manager
=============

Gestor especializado para conexiones de cache.
"""

import logging
from typing import Optional

from ...core.base.service_base import BaseService
from ...infrastructure.cache_redis import _cache_instance, get_cache

logger = logging.getLogger(__name__)


class CacheManager(BaseService):
    """Gestor de conexiones de cache."""
    
    def __init__(self):
        """Inicializar gestor."""
        super().__init__(logger_name=__name__)
        self.cache = None
    
    async def initialize(self):
        """Inicializar conexión de cache."""
        try:
            self.cache = await get_cache()
            self.log_info("Cache connection initialized")
        except Exception as e:
            self.log_warning(f"Failed to initialize cache: {e}")
            # Cache no es crítico, continuar sin él
    
    async def health_check(self) -> bool:
        """
        Verificar salud de la conexión.
        
        Returns:
            True si está saludable
        """
        if not _cache_instance or not _cache_instance.client:
            return False
        
        try:
            await _cache_instance.client.ping()
            self.log_debug("Cache health check: OK")
            return True
        except Exception as e:
            self.log_warning(f"Cache health check failed: {e}")
            # Intentar reconectar
            try:
                await _cache_instance.connect()
                return True
            except Exception:
                return False
    
    async def cleanup(self):
        """Limpiar conexión."""
        try:
            if _cache_instance:
                await _cache_instance.disconnect()
            self.log_info("Cache connection closed")
        except Exception as e:
            self.log_error(f"Error closing cache: {e}")

