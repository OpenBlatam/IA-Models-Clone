"""
Sistema de Transacciones con Rollback.
"""

from typing import Any, Callable, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from config.logging_config import get_logger
from core.database.connection_pool import get_pool
from core.exceptions import StorageError

logger = get_logger(__name__)


class Transaction:
    """Representa una transacción."""
    
    def __init__(self, conn):
        """
        Inicializar transacción.
        
        Args:
            conn: Conexión a la base de datos
        """
        self.conn = conn
        self.committed = False
        self.rolled_back = False
        self.savepoints: list = []
    
    async def begin(self) -> None:
        """Iniciar transacción."""
        await self.conn.execute("BEGIN")
        logger.debug("Transacción iniciada")
    
    async def commit(self) -> None:
        """Commit transacción."""
        if self.rolled_back:
            raise StorageError("Cannot commit rolled back transaction")
        await self.conn.commit()
        self.committed = True
        logger.debug("Transacción commiteada")
    
    async def rollback(self) -> None:
        """Rollback transacción."""
        if self.committed:
            raise StorageError("Cannot rollback committed transaction")
        await self.conn.rollback()
        self.rolled_back = True
        logger.debug("Transacción revertida")
    
    async def savepoint(self, name: str) -> None:
        """
        Crear savepoint.
        
        Args:
            name: Nombre del savepoint
        """
        await self.conn.execute(f"SAVEPOINT {name}")
        self.savepoints.append(name)
        logger.debug(f"Savepoint creado: {name}")
    
    async def rollback_to_savepoint(self, name: str) -> None:
        """
        Rollback a savepoint.
        
        Args:
            name: Nombre del savepoint
        """
        if name not in self.savepoints:
            raise StorageError(f"Savepoint {name} no existe")
        await self.conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
        logger.debug(f"Rollback a savepoint: {name}")
    
    async def release_savepoint(self, name: str) -> None:
        """
        Liberar savepoint.
        
        Args:
            name: Nombre del savepoint
        """
        if name not in self.savepoints:
            raise StorageError(f"Savepoint {name} no existe")
        await self.conn.execute(f"RELEASE SAVEPOINT {name}")
        self.savepoints.remove(name)
        logger.debug(f"Savepoint liberado: {name}")


@asynccontextmanager
async def transaction():
    """
    Context manager para transacciones.
    
    Yields:
        Transaction object
        
    Example:
        async with transaction() as tx:
            await tx.conn.execute("INSERT INTO ...")
            await tx.commit()
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        tx = Transaction(conn)
        try:
            await tx.begin()
            yield tx
            if not tx.committed and not tx.rolled_back:
                await tx.commit()
        except Exception as e:
            if not tx.rolled_back:
                await tx.rollback()
            logger.error(f"Error en transacción: {e}", exc_info=True)
            raise


async def transactional(func: Callable) -> Callable:
    """
    Decorador para funciones transaccionales.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    async def wrapper(*args, **kwargs):
        async with transaction() as tx:
            # Pasar transacción como primer argumento
            return await func(tx, *args, **kwargs)
    
    return wrapper



