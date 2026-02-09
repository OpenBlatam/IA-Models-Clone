"""
Connection Pool Manager
Gestión avanzada de connection pools para bases de datos y servicios
"""

import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncio

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """
    Gestor de connection pools
    Optimiza conexiones para alta concurrencia
    """
    
    def __init__(self):
        self._pools: Dict[str, Any] = {}
        self._pool_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_pool(
        self,
        name: str,
        pool_factory,
        min_size: int = 2,
        max_size: int = 10,
        **kwargs
    ):
        """
        Registra un connection pool
        
        Args:
            name: Nombre del pool
            pool_factory: Función que crea el pool
            min_size: Tamaño mínimo del pool
            max_size: Tamaño máximo del pool
        """
        self._pool_configs[name] = {
            "factory": pool_factory,
            "min_size": min_size,
            "max_size": max_size,
            **kwargs
        }
        logger.info(f"Registered connection pool: {name}")
    
    async def get_pool(self, name: str):
        """Obtiene un pool por nombre (lazy initialization)"""
        if name not in self._pools:
            if name not in self._pool_configs:
                raise ValueError(f"Pool {name} not registered")
            
            config = self._pool_configs[name]
            pool = await config["factory"](
                min_size=config["min_size"],
                max_size=config["max_size"],
                **{k: v for k, v in config.items() if k not in ["factory", "min_size", "max_size"]}
            )
            self._pools[name] = pool
            logger.info(f"Initialized connection pool: {name}")
        
        return self._pools[name]
    
    @asynccontextmanager
    async def acquire(self, pool_name: str):
        """Adquiere una conexión del pool"""
        pool = await self.get_pool(pool_name)
        
        if hasattr(pool, "acquire"):
            async with pool.acquire() as connection:
                yield connection
        else:
            # Para pools que no tienen acquire (ej: Redis)
            yield pool
    
    async def close_all(self):
        """Cierra todos los pools"""
        for name, pool in self._pools.items():
            try:
                if hasattr(pool, "close"):
                    await pool.close()
                elif hasattr(pool, "dispose"):
                    await pool.dispose()
                logger.info(f"Closed connection pool: {name}")
            except Exception as e:
                logger.error(f"Error closing pool {name}: {e}")
        
        self._pools.clear()
    
    def get_pool_stats(self, name: str) -> Dict[str, Any]:
        """Obtiene estadísticas de un pool"""
        if name not in self._pools:
            return {"error": "Pool not initialized"}
        
        pool = self._pools[name]
        stats = {
            "name": name,
            "initialized": True
        }
        
        # Estadísticas específicas según el tipo de pool
        if hasattr(pool, "size"):
            stats["size"] = pool.size()
        if hasattr(pool, "idle"):
            stats["idle"] = pool.idle()
        if hasattr(pool, "maxsize"):
            stats["maxsize"] = pool.maxsize
        
        return stats


# Instancia global
_pool_manager: Optional[ConnectionPoolManager] = None


def get_connection_pool_manager() -> ConnectionPoolManager:
    """Obtiene el gestor de connection pools"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager















