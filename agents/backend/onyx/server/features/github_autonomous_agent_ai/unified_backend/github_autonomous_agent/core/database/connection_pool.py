"""
Connection Pool Avanzado - Pool de conexiones optimizado para SQLite y PostgreSQL.
"""

import asyncio
from typing import Optional, AsyncContextManager
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import deque
import aiosqlite

from config.logging_config import get_logger
from config.settings import settings
from core.exceptions import StorageError

logger = get_logger(__name__)


class ConnectionPool:
    """Pool avanzado de conexiones con health checks y auto-recovery."""
    
    def __init__(
        self,
        db_path: str,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0,
        health_check_interval: float = 60.0
    ):
        """
        Inicializar pool de conexiones.
        
        Args:
            db_path: Ruta a la base de datos
            min_connections: Número mínimo de conexiones
            max_connections: Número máximo de conexiones
            connection_timeout: Timeout para obtener conexión (segundos)
            idle_timeout: Timeout para conexiones idle (segundos)
            health_check_interval: Intervalo para health checks (segundos)
        """
        self.db_path = db_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.health_check_interval = health_check_interval
        
        self.pool: deque = deque()
        self.active_connections: int = 0
        self.total_connections: int = 0
        self.stats = {
            "created": 0,
            "acquired": 0,
            "released": 0,
            "timeouts": 0,
            "errors": 0,
            "health_checks": 0
        }
        
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self) -> None:
        """Inicializar pool con conexiones mínimas."""
        async with self._lock:
            for _ in range(self.min_connections):
                conn = await self._create_connection()
                if conn:
                    self.pool.append(conn)
                    self.total_connections += 1
            self._running = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info(f"Connection pool inicializado: {len(self.pool)} conexiones")
    
    async def _create_connection(self) -> Optional[aiosqlite.Connection]:
        """Crear nueva conexión."""
        try:
            conn = await aiosqlite.connect(
                self.db_path,
                timeout=10.0,
                isolation_level=None
            )
            # Optimizaciones SQLite
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            self.stats["created"] += 1
            return conn
        except Exception as e:
            logger.error(f"Error creando conexión: {e}", exc_info=True)
            self.stats["errors"] += 1
            return None
    
    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[aiosqlite.Connection]:
        """
        Adquirir conexión del pool.
        
        Yields:
            Conexión aiosqlite
            
        Raises:
            StorageError: Si no se puede obtener conexión
        """
        start_time = datetime.now()
        conn = None
        
        try:
            # Intentar obtener conexión del pool
            while True:
                async with self._lock:
                    if self.pool:
                        conn = self.pool.popleft()
                        self.active_connections += 1
                        self.stats["acquired"] += 1
                        break
                    elif self.total_connections < self.max_connections:
                        # Crear nueva conexión
                        conn = await self._create_connection()
                        if conn:
                            self.total_connections += 1
                            self.active_connections += 1
                            self.stats["acquired"] += 1
                            break
                
                # Esperar con timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= self.connection_timeout:
                    self.stats["timeouts"] += 1
                    raise StorageError(
                        f"Timeout esperando conexión ({self.connection_timeout}s)"
                    )
                
                await asyncio.sleep(0.1)
            
            try:
                yield conn
            finally:
                # Devolver conexión al pool
                async with self._lock:
                    if conn:
                        # Verificar que la conexión sigue viva
                        try:
                            await conn.execute("SELECT 1")
                            self.pool.append(conn)
                        except Exception:
                            # Conexión muerta, crear nueva
                            try:
                                await conn.close()
                            except Exception:
                                pass
                            self.total_connections -= 1
                            conn = await self._create_connection()
                            if conn:
                                self.pool.append(conn)
                                self.total_connections += 1
                        
                        self.active_connections -= 1
                        self.stats["released"] += 1
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error en acquire: {e}", exc_info=True)
            self.stats["errors"] += 1
            raise StorageError(f"Error obteniendo conexión: {e}") from e
    
    async def _health_check_loop(self) -> None:
        """Loop de health checks periódicos."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._health_check()
            except Exception as e:
                logger.error(f"Error en health check loop: {e}", exc_info=True)
    
    async def _health_check(self) -> None:
        """Verificar salud de conexiones en el pool."""
        async with self._lock:
            healthy_connections = []
            for conn in self.pool:
                try:
                    await conn.execute("SELECT 1")
                    healthy_connections.append(conn)
                except Exception:
                    # Conexión muerta, cerrar
                    try:
                        await conn.close()
                    except Exception:
                        pass
                    self.total_connections -= 1
            
            self.pool = deque(healthy_connections)
            
            # Asegurar mínimo de conexiones
            while len(self.pool) < self.min_connections and self.total_connections < self.max_connections:
                conn = await self._create_connection()
                if conn:
                    self.pool.append(conn)
                    self.total_connections += 1
            
            self.stats["health_checks"] += 1
    
    async def close(self) -> None:
        """Cerrar todas las conexiones."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            while self.pool:
                conn = self.pool.popleft()
                try:
                    await conn.close()
                except Exception:
                    pass
            self.pool.clear()
            self.total_connections = 0
            self.active_connections = 0
        
        logger.info("Connection pool cerrado")
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del pool."""
        return {
            "pool_size": len(self.pool),
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "max_connections": self.max_connections,
            "min_connections": self.min_connections,
            **self.stats
        }


# Pool global
_pool: Optional[ConnectionPool] = None


def get_pool() -> ConnectionPool:
    """Obtener pool global."""
    global _pool
    if _pool is None:
        db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
        _pool = ConnectionPool(
            db_path=db_path,
            min_connections=2,
            max_connections=10
        )
    return _pool



