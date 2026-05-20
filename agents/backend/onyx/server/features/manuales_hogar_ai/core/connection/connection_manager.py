"""
Connection Manager
==================

Gestor principal de conexiones que compone managers especializados.
"""

import logging
import asyncio
from typing import Optional

from ...core.base.service_base import BaseService
from .database_manager import DatabaseManager
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class ConnectionManager(BaseService):
    """Gestor principal de conexiones."""
    
    def __init__(self):
        """Inicializar gestor."""
        super().__init__(logger_name=__name__)
        self.database_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Inicializar todas las conexiones."""
        self.log_info("Initializing connections...")
        
        await self.database_manager.initialize()
        await self.cache_manager.initialize()
    
    async def health_check_loop(self, interval: int = 60):
        """Loop periódico de health checks."""
        while True:
            try:
                await asyncio.sleep(interval)
                
                await self.database_manager.health_check()
                await self.cache_manager.health_check()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Health check loop error: {e}")
    
    async def start_health_checks(self, interval: int = 60):
        """Iniciar health checks periódicos."""
        self._health_check_task = asyncio.create_task(self.health_check_loop(interval))
        self.log_info(f"Started connection health checks (interval: {interval}s)")
    
    async def stop_health_checks(self):
        """Detener health checks periódicos."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self.log_info("Stopped connection health checks")
    
    async def cleanup(self):
        """Limpiar todas las conexiones."""
        self.log_info("Cleaning up connections...")
        
        await self.stop_health_checks()
        await self.database_manager.cleanup()
        await self.cache_manager.cleanup()


# Singleton instance
_connection_manager: Optional[ConnectionManager] = None


async def get_connection_manager() -> ConnectionManager:
    """Obtener instancia singleton del connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
        await _connection_manager.initialize()
    return _connection_manager

