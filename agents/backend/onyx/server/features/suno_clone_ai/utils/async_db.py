"""
Utilidades para operaciones de base de datos async optimizadas

Incluye connection pooling y optimizaciones de queries.
"""

import aiosqlite
from typing import Optional
from contextlib import asynccontextmanager

from ..config.settings import settings


class AsyncDBPool:
    """Pool de conexiones async para SQLite"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def get_connection(self) -> aiosqlite.Connection:
        """Obtiene una conexión reutilizable"""
        if self._connection is None:
            self._connection = await aiosqlite.connect(
                self.db_path,
                check_same_thread=False
            )
            # Optimizaciones de SQLite para mejor rendimiento
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA synchronous=NORMAL")
            await self._connection.execute("PRAGMA cache_size=10000")
            await self._connection.execute("PRAGMA temp_store=MEMORY")
            await self._connection.execute("PRAGMA mmap_size=268435456")  # 256MB
        return self._connection
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager para transacciones"""
        conn = await self.get_connection()
        try:
            await conn.execute("BEGIN")
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    
    async def close(self) -> None:
        """Cierra la conexión"""
        if self._connection:
            await self._connection.close()
            self._connection = None


# Pool global
_db_pool: Optional[AsyncDBPool] = None


def get_db_pool() -> AsyncDBPool:
    """Obtiene el pool de conexiones global"""
    global _db_pool
    if _db_pool is None:
        db_path = settings.database_url.replace("sqlite:///", "")
        _db_pool = AsyncDBPool(db_path)
    return _db_pool

