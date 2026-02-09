"""
Connection Reuse
Reutilización agresiva de conexiones
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from collections import deque

logger = logging.getLogger(__name__)


class ConnectionReuseManager:
    """Gestor de reutilización de conexiones"""
    
    def __init__(self, max_idle: int = 10, idle_timeout: float = 300.0):
        self.max_idle = max_idle
        self.idle_timeout = idle_timeout
        self._pools: Dict[str, deque] = {}
        self._connection_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get_connection(
        self,
        pool_name: str,
        factory: callable,
        *args,
        **kwargs
    ):
        """
        Obtiene o crea una conexión reutilizable
        
        Args:
            pool_name: Nombre del pool
            factory: Función que crea la conexión
            *args, **kwargs: Argumentos para factory
        """
        async with self._lock:
            # Intentar obtener de pool
            if pool_name in self._pools and self._pools[pool_name]:
                conn = self._pools[pool_name].popleft()
                # Verificar si la conexión sigue válida
                if await self._is_valid(conn):
                    logger.debug(f"Reusing connection from pool: {pool_name}")
                    return conn
                else:
                    logger.debug(f"Connection invalid, creating new: {pool_name}")
            
            # Crear nueva conexión
            conn = await factory(*args, **kwargs)
            return conn
    
    async def return_connection(self, pool_name: str, conn: Any):
        """Devuelve conexión al pool"""
        async with self._lock:
            if pool_name not in self._pools:
                self._pools[pool_name] = deque(maxlen=self.max_idle)
            
            if len(self._pools[pool_name]) < self.max_idle:
                self._pools[pool_name].append(conn)
                self._connection_times[id(conn)] = asyncio.get_event_loop().time()
                logger.debug(f"Returned connection to pool: {pool_name}")
            else:
                # Pool lleno, cerrar conexión
                await self._close_connection(conn)
    
    async def _is_valid(self, conn: Any) -> bool:
        """Verifica si una conexión es válida"""
        try:
            # Intentar operación simple
            if hasattr(conn, 'is_closed'):
                return not conn.is_closed()
            return True
        except Exception:
            return False
    
    async def _close_connection(self, conn: Any):
        """Cierra una conexión"""
        try:
            if hasattr(conn, 'close'):
                await conn.close()
            elif hasattr(conn, 'dispose'):
                await conn.dispose()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    async def cleanup_idle(self):
        """Limpia conexiones idle"""
        import time
        current_time = time.time()
        
        async with self._lock:
            for pool_name, pool in self._pools.items():
                to_remove = []
                for conn in pool:
                    conn_id = id(conn)
                    if conn_id in self._connection_times:
                        idle_time = current_time - self._connection_times[conn_id]
                        if idle_time > self.idle_timeout:
                            to_remove.append(conn)
                
                for conn in to_remove:
                    pool.remove(conn)
                    await self._close_connection(conn)
                    conn_id = id(conn)
                    if conn_id in self._connection_times:
                        del self._connection_times[conn_id]
    
    @asynccontextmanager
    async def connection(self, pool_name: str, factory: callable, *args, **kwargs):
        """Context manager para conexiones"""
        conn = await self.get_connection(pool_name, factory, *args, **kwargs)
        try:
            yield conn
        finally:
            await self.return_connection(pool_name, conn)


# Instancia global
_connection_reuse: Optional[ConnectionReuseManager] = None


def get_connection_reuse() -> ConnectionReuseManager:
    """Obtiene el gestor de reutilización de conexiones"""
    global _connection_reuse
    if _connection_reuse is None:
        _connection_reuse = ConnectionReuseManager()
    return _connection_reuse

