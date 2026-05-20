"""
Connection Helpers - Utilidades para gestión de conexiones
===========================================================

Funciones helper para facilitar la gestión de conexiones y pools.
"""

import logging
import asyncio
from typing import Callable, Any, Optional, Dict, TypeVar, Generic
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConnectionManager:
    """
    Gestor de conexiones con health checks y auto-reconnect.
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        health_check: Optional[Callable[[T], bool]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Inicializar gestor de conexiones.
        
        Args:
            factory: Función que crea conexiones
            health_check: Función de health check (opcional)
            max_retries: Número máximo de reintentos
            retry_delay: Delay entre reintentos en segundos
        """
        self.factory = factory
        self.health_check = health_check
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connection: Optional[T] = None
        self._last_check: Optional[datetime] = None
        self._check_interval = timedelta(seconds=60)
    
    async def get_connection(self, force_new: bool = False) -> T:
        """
        Obtener conexión (reutiliza si está saludable).
        
        Args:
            force_new: Forzar nueva conexión
        
        Returns:
            Conexión válida
        """
        if force_new or not self._connection:
            self._connection = await self._create_with_retry()
            return self._connection
        
        # Verificar salud si es necesario
        if self._needs_health_check():
            if not await self._is_healthy():
                logger.warning("Connection unhealthy, creating new one")
                self._connection = await self._create_with_retry()
        
        return self._connection
    
    async def _create_with_retry(self) -> T:
        """Crear conexión con reintentos"""
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(self.factory):
                    conn = await self.factory()
                else:
                    conn = self.factory()
                
                # Verificar salud si hay health check
                if self.health_check:
                    if asyncio.iscoroutinefunction(self.health_check):
                        is_healthy = await self.health_check(conn)
                    else:
                        is_healthy = self.health_check(conn)
                    
                    if not is_healthy:
                        raise ValueError("Connection created but health check failed")
                
                return conn
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Connection creation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to create connection after {self.max_retries} attempts: {e}")
                    raise
    
    async def _is_healthy(self) -> bool:
        """Verificar si la conexión está saludable"""
        if not self._connection:
            return False
        
        if not self.health_check:
            return True
        
        try:
            if asyncio.iscoroutinefunction(self.health_check):
                return await self.health_check(self._connection)
            return self.health_check(self._connection)
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def _needs_health_check(self) -> bool:
        """Determinar si se necesita health check"""
        if not self._last_check:
            return True
        return datetime.utcnow() - self._last_check > self._check_interval
    
    async def close(self) -> None:
        """Cerrar conexión"""
        if self._connection:
            try:
                if hasattr(self._connection, 'close'):
                    if asyncio.iscoroutinefunction(self._connection.close):
                        await self._connection.close()
                    else:
                        self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._connection = None
    
    @asynccontextmanager
    async def connection(self, force_new: bool = False):
        """
        Context manager para obtener conexión.
        
        Args:
            force_new: Forzar nueva conexión
        
        Yields:
            Conexión
        """
        conn = await self.get_connection(force_new=force_new)
        try:
            yield conn
        finally:
            # No cerrar aquí, reutilizar para próximas operaciones
            pass


def create_connection_manager(
    factory: Callable[[], T],
    health_check: Optional[Callable[[T], bool]] = None,
    **kwargs
) -> ConnectionManager:
    """
    Crear gestor de conexiones.
    
    Args:
        factory: Función que crea conexiones
        health_check: Función de health check (opcional)
        **kwargs: Argumentos adicionales para ConnectionManager
    
    Returns:
        ConnectionManager instance
    """
    return ConnectionManager(factory, health_check, **kwargs)


@asynccontextmanager
async def managed_connection(
    factory: Callable[[], T],
    health_check: Optional[Callable[[T], bool]] = None
):
    """
    Context manager para conexión gestionada.
    
    Args:
        factory: Función que crea conexiones
        health_check: Función de health check (opcional)
    
    Yields:
        Conexión
    """
    manager = ConnectionManager(factory, health_check)
    try:
        async with manager.connection() as conn:
            yield conn
    finally:
        await manager.close()

