"""
Connection Manager - Gestor de Conexiones
=========================================

Sistema de gestión de conexiones con pooling, keep-alive y estadísticas.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Estado de conexión."""
    IDLE = "idle"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class Connection:
    """Conexión."""
    connection_id: str
    connection_type: str
    state: ConnectionState = ConnectionState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    use_count: int = 0
    total_usage_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionManager:
    """Gestor de conexiones."""
    
    def __init__(self):
        self.connections: Dict[str, Connection] = {}
        self.connections_by_type: Dict[str, List[str]] = defaultdict(list)
        self.idle_connections: Dict[str, deque] = defaultdict(deque)
        self.active_connections: Dict[str, Connection] = {}
        self.connection_history: deque = deque(maxlen=100000)
        self.statistics: Dict[str, Any] = {
            "total_created": 0,
            "total_closed": 0,
            "total_errors": 0,
            "peak_active": 0,
        }
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def register_connection(
        self,
        connection_id: str,
        connection_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar conexión."""
        connection = Connection(
            connection_id=connection_id,
            connection_type=connection_type,
            metadata=metadata or {},
        )
        
        async def save_connection():
            async with self._lock:
                self.connections[connection_id] = connection
                self.connections_by_type[connection_type].append(connection_id)
                self.idle_connections[connection_type].append(connection_id)
                self.statistics["total_created"] += 1
        
        asyncio.create_task(save_connection())
        
        logger.info(f"Registered connection: {connection_id} ({connection_type})")
        return connection_id
    
    async def acquire_connection(
        self,
        connection_type: str,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """Adquirir conexión."""
        timeout = timeout or 30.0
        start_time = datetime.now()
        
        while True:
            async with self._lock:
                # Buscar conexión idle
                if self.idle_connections[connection_type]:
                    connection_id = self.idle_connections[connection_type].popleft()
                    connection = self.connections.get(connection_id)
                    
                    if connection and connection.state == ConnectionState.IDLE:
                        connection.state = ConnectionState.ACTIVE
                        connection.last_used = datetime.now()
                        connection.use_count += 1
                        self.active_connections[connection_id] = connection
                        
                        # Actualizar peak
                        active_count = len([
                            c for c in self.connections.values()
                            if c.state == ConnectionState.ACTIVE
                        ])
                        if active_count > self.statistics["peak_active"]:
                            self.statistics["peak_active"] = active_count
                        
                        return connection_id
            
            # Esperar si hay timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= timeout:
                return None
            
            await asyncio.sleep(0.1)
    
    async def release_connection(self, connection_id: str):
        """Liberar conexión."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        async with self._lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            # Calcular tiempo de uso
            if connection.last_used:
                usage_time = (datetime.now() - connection.last_used).total_seconds()
                connection.total_usage_time += usage_time
            
            connection.state = ConnectionState.IDLE
            connection.last_used = datetime.now()
            self.idle_connections[connection.connection_type].append(connection_id)
        
        logger.debug(f"Released connection: {connection_id}")
    
    async def close_connection(self, connection_id: str):
        """Cerrar conexión."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        async with self._lock:
            connection.state = ConnectionState.CLOSING
            connection.state = ConnectionState.CLOSED
            
            # Remover de todos los lugares
            if connection_id in self.connections:
                del self.connections[connection_id]
            
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            if connection_id in self.idle_connections[connection.connection_type]:
                self.idle_connections[connection.connection_type].remove(connection_id)
            
            if connection_id in self.connections_by_type[connection.connection_type]:
                self.connections_by_type[connection.connection_type].remove(connection_id)
            
            self.statistics["total_closed"] += 1
            
            # Guardar en historial
            self.connection_history.append({
                "connection_id": connection_id,
                "connection_type": connection.connection_type,
                "closed_at": datetime.now().isoformat(),
                "use_count": connection.use_count,
                "total_usage_time": connection.total_usage_time,
            })
        
        logger.info(f"Closed connection: {connection_id}")
    
    async def cleanup_idle_connections(self, max_idle_time: float = 300.0):
        """Limpiar conexiones idle antiguas."""
        now = datetime.now()
        connections_to_close = []
        
        async with self._lock:
            for connection_type, connection_ids in self.idle_connections.items():
                for connection_id in list(connection_ids):
                    connection = self.connections.get(connection_id)
                    if connection and connection.last_used:
                        idle_time = (now - connection.last_used).total_seconds()
                        if idle_time > max_idle_time:
                            connections_to_close.append(connection_id)
        
        for connection_id in connections_to_close:
            await self.close_connection(connection_id)
    
    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de conexión."""
        connection = self.connections.get(connection_id)
        if not connection:
            return None
        
        return {
            "connection_id": connection.connection_id,
            "connection_type": connection.connection_type,
            "state": connection.state.value,
            "created_at": connection.created_at.isoformat(),
            "last_used": connection.last_used.isoformat() if connection.last_used else None,
            "use_count": connection.use_count,
            "total_usage_time": connection.total_usage_time,
        }
    
    def get_connections_by_type(self, connection_type: str) -> List[Dict[str, Any]]:
        """Obtener conexiones por tipo."""
        connection_ids = self.connections_by_type.get(connection_type, [])
        
        return [
            self.get_connection(cid)
            for cid in connection_ids
            if self.get_connection(cid)
        ]
    
    def get_connection_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_type: Dict[str, int] = defaultdict(int)
        by_state: Dict[str, int] = defaultdict(int)
        
        for connection in self.connections.values():
            by_type[connection.connection_type] += 1
            by_state[connection.state.value] += 1
        
        return {
            "total_connections": len(self.connections),
            "connections_by_type": dict(by_type),
            "connections_by_state": dict(by_state),
            "idle_connections": sum(len(dq) for dq in self.idle_connections.values()),
            "active_connections": len(self.active_connections),
            "statistics": self.statistics.copy(),
        }


