"""
Load Balancer System
====================

Sistema de balanceador de carga para distribución de trabajo.
"""

import logging
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """Estrategia de balanceo."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"


@dataclass
class Server:
    """Servidor."""
    server_id: str
    name: str
    address: str
    weight: int = 1
    active: bool = True
    connections: int = 0
    response_time: float = 0.0
    last_health_check: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """
    Balanceador de carga.
    
    Distribuye carga entre múltiples servidores.
    """
    
    def __init__(
        self,
        name: str,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    ):
        """
        Inicializar balanceador de carga.
        
        Args:
            name: Nombre del balanceador
            strategy: Estrategia de balanceo
        """
        self.name = name
        self.strategy = strategy
        self.servers: Dict[str, Server] = {}
        self.round_robin_index = 0
    
    def add_server(
        self,
        server_id: str,
        name: str,
        address: str,
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Server:
        """
        Agregar servidor.
        
        Args:
            server_id: ID único del servidor
            name: Nombre
            address: Dirección del servidor
            weight: Peso (para estrategia weighted)
            metadata: Metadata adicional
            
        Returns:
            Servidor agregado
        """
        server = Server(
            server_id=server_id,
            name=name,
            address=address,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.servers[server_id] = server
        logger.info(f"Added server to load balancer {self.name}: {name} ({server_id})")
        
        return server
    
    def remove_server(self, server_id: str) -> bool:
        """Remover servidor."""
        if server_id in self.servers:
            del self.servers[server_id]
            return True
        return False
    
    def get_server(self) -> Optional[Server]:
        """
        Obtener servidor según estrategia.
        
        Returns:
            Servidor seleccionado o None
        """
        active_servers = [
            s for s in self.servers.values()
            if s.active
        ]
        
        if not active_servers:
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            server = active_servers[self.round_robin_index % len(active_servers)]
            self.round_robin_index += 1
            return server
        
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(active_servers)
        
        elif self.strategy == LoadBalanceStrategy.WEIGHTED:
            total_weight = sum(s.weight for s in active_servers)
            r = random.random() * total_weight
            cumulative = 0
            for server in active_servers:
                cumulative += server.weight
                if r <= cumulative:
                    return server
            return active_servers[0]
        
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(active_servers, key=lambda s: s.connections)
        
        elif self.strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            return min(active_servers, key=lambda s: s.response_time)
        
        return active_servers[0]
    
    def record_connection(self, server_id: str) -> None:
        """Registrar conexión."""
        if server_id in self.servers:
            self.servers[server_id].connections += 1
    
    def release_connection(self, server_id: str) -> None:
        """Liberar conexión."""
        if server_id in self.servers:
            self.servers[server_id].connections = max(0, self.servers[server_id].connections - 1)
    
    def update_response_time(self, server_id: str, response_time: float) -> None:
        """Actualizar tiempo de respuesta."""
        if server_id in self.servers:
            self.servers[server_id].response_time = response_time
            self.servers[server_id].last_health_check = datetime.now().isoformat()
    
    def set_server_active(self, server_id: str, active: bool) -> bool:
        """Establecer estado activo de servidor."""
        if server_id in self.servers:
            self.servers[server_id].active = active
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del balanceador."""
        total_servers = len(self.servers)
        active_servers = sum(1 for s in self.servers.values() if s.active)
        total_connections = sum(s.connections for s in self.servers.values())
        
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "total_servers": total_servers,
            "active_servers": active_servers,
            "total_connections": total_connections,
            "servers": [
                {
                    "server_id": s.server_id,
                    "name": s.name,
                    "address": s.address,
                    "active": s.active,
                    "connections": s.connections,
                    "response_time": s.response_time
                }
                for s in self.servers.values()
            ]
        }


# Instancia global de balanceadores
_load_balancers: Dict[str, LoadBalancer] = {}


def create_load_balancer(
    name: str,
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
) -> LoadBalancer:
    """
    Crear balanceador de carga.
    
    Args:
        name: Nombre del balanceador
        strategy: Estrategia de balanceo
        
    Returns:
        Balanceador de carga
    """
    balancer = LoadBalancer(name, strategy)
    _load_balancers[name] = balancer
    return balancer


def get_load_balancer(name: str) -> Optional[LoadBalancer]:
    """Obtener balanceador de carga por nombre."""
    return _load_balancers.get(name)






