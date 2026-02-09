"""
Connection Pool System
======================

Sistema de pool de conexiones para recursos compartidos.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Generic, TypeVar, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConnectionStatus(Enum):
    """Estado de conexión."""
    IDLE = "idle"
    IN_USE = "in_use"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class Connection(Generic[T]):
    """Conexión."""
    connection_id: str
    resource: T
    status: ConnectionStatus = ConnectionStatus.IDLE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used_at: Optional[str] = None
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionPool(Generic[T]):
    """
    Pool de conexiones.
    
    Gestiona pool de conexiones reutilizables.
    """
    
    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: float = 300.0  # 5 minutos
    ):
        """
        Inicializar pool de conexiones.
        
        Args:
            name: Nombre del pool
            factory: Función para crear conexiones
            max_size: Tamaño máximo del pool
            min_size: Tamaño mínimo del pool
            max_idle_time: Tiempo máximo de inactividad (segundos)
        """
        self.name = name
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        
        self.connections: Dict[str, Connection[T]] = {}
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        
        # Inicializar pool mínimo
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self) -> None:
        """Inicializar pool con conexiones mínimas."""
        for _ in range(self.min_size):
            await self._create_connection()
    
    async def _create_connection(self) -> Optional[Connection[T]]:
        """Crear nueva conexión."""
        if len(self.connections) >= self.max_size:
            return None
        
        try:
            connection_id = f"conn_{len(self.connections)}"
            resource = self.factory()
            
            connection = Connection(
                connection_id=connection_id,
                resource=resource,
                status=ConnectionStatus.IDLE
            )
            
            self.connections[connection_id] = connection
            await self.available_connections.put(connection_id)
            
            logger.info(f"Created connection in pool {self.name}: {connection_id}")
            
            return connection
        except Exception as e:
            logger.error(f"Error creating connection in pool {self.name}: {e}")
            return None
    
    async def acquire(self, timeout: float = 10.0) -> Optional[Connection[T]]:
        """
        Adquirir conexión del pool.
        
        Args:
            timeout: Timeout en segundos
            
        Returns:
            Conexión o None si timeout
        """
        try:
            # Intentar obtener conexión disponible
            connection_id = await asyncio.wait_for(
                self.available_connections.get(),
                timeout=timeout
            )
            
            connection = self.connections.get(connection_id)
            if connection:
                connection.status = ConnectionStatus.IN_USE
                connection.last_used_at = datetime.now().isoformat()
                connection.use_count += 1
                return connection
            
            # Si no hay conexión disponible, crear nueva
            connection = await self._create_connection()
            if connection:
                connection.status = ConnectionStatus.IN_USE
                connection.last_used_at = datetime.now().isoformat()
                connection.use_count += 1
                return connection
            
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout acquiring connection from pool {self.name}")
            return None
    
    async def release(self, connection: Connection[T]) -> None:
        """
        Liberar conexión al pool.
        
        Args:
            connection: Conexión a liberar
        """
        if connection.connection_id not in self.connections:
            return
        
        connection.status = ConnectionStatus.IDLE
        await self.available_connections.put(connection.connection_id)
    
    async def close_connection(self, connection: Connection[T]) -> None:
        """
        Cerrar conexión.
        
        Args:
            connection: Conexión a cerrar
        """
        if connection.connection_id in self.connections:
            connection.status = ConnectionStatus.CLOSED
            del self.connections[connection.connection_id]
            logger.info(f"Closed connection in pool {self.name}: {connection.connection_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del pool."""
        total = len(self.connections)
        idle = sum(1 for c in self.connections.values() if c.status == ConnectionStatus.IDLE)
        in_use = sum(1 for c in self.connections.values() if c.status == ConnectionStatus.IN_USE)
        
        return {
            "name": self.name,
            "total_connections": total,
            "idle_connections": idle,
            "in_use_connections": in_use,
            "available_connections": self.available_connections.qsize(),
            "max_size": self.max_size,
            "min_size": self.min_size
        }


# Instancia global de pools
_connection_pools: Dict[str, ConnectionPool] = {}


def create_connection_pool(
    name: str,
    factory: Callable[[], T],
    max_size: int = 10,
    min_size: int = 2
) -> ConnectionPool[T]:
    """
    Crear pool de conexiones.
    
    Args:
        name: Nombre del pool
        factory: Función para crear conexiones
        max_size: Tamaño máximo
        min_size: Tamaño mínimo
        
    Returns:
        Pool de conexiones
    """
    pool = ConnectionPool(name, factory, max_size, min_size)
    _connection_pools[name] = pool
    return pool


def get_connection_pool(name: str) -> Optional[ConnectionPool]:
    """Obtener pool de conexiones por nombre."""
    return _connection_pools.get(name)






