"""
MCP Connection Pooling - Pool de conexiones
============================================
"""

import asyncio
import logging
from typing import Optional, Callable, Any, Generic, TypeVar, Dict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConnectionPool(Generic[T]):
    """
    Pool de conexiones genérico
    
    Permite reutilizar conexiones para mejorar performance.
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time: int = 300,
    ):
        """
        Inicializa el pool de conexiones.
        
        Args:
            factory: Función que crea nuevas conexiones
            min_size: Tamaño mínimo del pool (debe ser >= 0 y <= max_size)
            max_size: Tamaño máximo del pool (debe ser > 0)
            max_idle_time: Tiempo máximo de inactividad en segundos (debe ser > 0)
            
        Raises:
            ValueError: Si los parámetros son inválidos
            TypeError: Si factory no es callable
        """
        if not callable(factory):
            raise TypeError("factory must be callable")
        if not isinstance(min_size, int) or min_size < 0:
            raise ValueError("min_size must be a non-negative integer")
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        if min_size > max_size:
            raise ValueError("min_size cannot be greater than max_size")
        if not isinstance(max_idle_time, int) or max_idle_time <= 0:
            raise ValueError("max_idle_time must be a positive integer")
        
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._active_connections = 0
        self._created_connections = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """
        Inicializa el pool con conexiones mínimas.
        
        Crea las conexiones iniciales según min_size. Si alguna falla,
        se registra el error pero el pool continúa inicializándose.
        """
        for _ in range(self.min_size):
            try:
                conn = await self._create_connection()
                await self._pool.put(conn)
            except Exception as e:
                logger.error(f"Error initializing pool connection: {e}", exc_info=True)
    
    async def _create_connection(self) -> T:
        """
        Crea una nueva conexión usando la factory.
        
        Returns:
            Nueva conexión del tipo T
            
        Raises:
            Exception: Si la factory falla al crear la conexión
        """
        try:
            self._created_connections += 1
            conn = self.factory()
            
            # Si es async, await
            if asyncio.iscoroutine(conn):
                conn = await conn
            
            return conn
        except Exception as e:
            logger.error(f"Error creating connection: {e}", exc_info=True)
            self._created_connections -= 1  # Revertir contador
            raise
    
    @asynccontextmanager
    async def acquire(self):
        """
        Adquiere una conexión del pool
        
        Yields:
            Conexión del pool
        """
        conn = None
        try:
            # Intentar obtener del pool
            try:
                conn = await asyncio.wait_for(
                    self._pool.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                # Crear nueva si no hay disponible y no excedemos max
                async with self._lock:
                    if self._active_connections < self.max_size:
                        conn = await self._create_connection()
                    else:
                        # Esperar a que haya una disponible
                        conn = await self._pool.get()
            
            self._active_connections += 1
            yield conn
            
        finally:
            if conn:
                self._active_connections -= 1
                # Retornar al pool
                try:
                    await self._pool.put_nowait(conn)
                except asyncio.QueueFull:
                    # Pool lleno, descartar conexión
                    await self._close_connection(conn)
    
    async def _close_connection(self, conn: T):
        """Cierra una conexión"""
        try:
            if hasattr(conn, 'close'):
                if asyncio.iscoroutinefunction(conn.close):
                    await conn.close()
                else:
                    conn.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    async def close_all(self) -> None:
        """
        Cierra todas las conexiones del pool.
        
        Itera sobre todas las conexiones en el pool y las cierra
        de forma segura, manejando errores individuales.
        """
        closed_count = 0
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await self._close_connection(conn)
                closed_count += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.warning(f"Error closing connection in pool: {e}")
        
        if closed_count > 0:
            logger.info(f"Closed {closed_count} connections from pool")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del pool
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "pool_size": self._pool.qsize(),
            "active_connections": self._active_connections,
            "created_connections": self._created_connections,
            "max_size": self.max_size,
            "min_size": self.min_size,
        }


class DatabaseConnectionPool(ConnectionPool):
    """Pool especializado para conexiones de base de datos"""
    pass


class HTTPConnectionPool(ConnectionPool):
    """Pool especializado para clientes HTTP"""
    pass

