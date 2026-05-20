"""
MCP Load Balancer - Balanceador de carga
==========================================
"""

import logging
import random
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(str, Enum):
    """Estrategias de balanceo de carga"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    IP_HASH = "ip_hash"


class BackendServer(BaseModel):
    """Servidor backend"""
    server_id: str = Field(..., description="ID único del servidor")
    url: str = Field(..., description="URL del servidor")
    weight: int = Field(default=1, description="Peso del servidor")
    healthy: bool = Field(default=True, description="Estado de salud")
    active_connections: int = Field(default=0, description="Conexiones activas")
    last_used: Optional[datetime] = Field(None, description="Última vez usado")


class LoadBalancer:
    """
    Balanceador de carga
    
    Distribuye requests entre múltiples servidores backend.
    """
    
    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    ):
        """
        Args:
            strategy: Estrategia de balanceo
        """
        self.strategy = strategy
        self._servers: Dict[str, BackendServer] = {}
        self._current_index = 0
    
    def add_server(self, server: BackendServer):
        """
        Agrega servidor backend
        
        Args:
            server: Servidor backend
        """
        self._servers[server.server_id] = server
        logger.info(f"Added backend server: {server.server_id} at {server.url}")
    
    def remove_server(self, server_id: str):
        """
        Elimina servidor backend
        
        Args:
            server_id: ID del servidor
        """
        if server_id in self._servers:
            del self._servers[server_id]
            logger.info(f"Removed backend server: {server_id}")
    
    def get_server(self, client_ip: Optional[str] = None) -> Optional[BackendServer]:
        """
        Obtiene servidor según estrategia
        
        Args:
            client_ip: IP del cliente (para IP_HASH)
            
        Returns:
            BackendServer o None
        """
        healthy_servers = [
            s for s in self._servers.values()
            if s.healthy
        ]
        
        if not healthy_servers:
            logger.warning("No healthy servers available")
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            server = healthy_servers[self._current_index % len(healthy_servers)]
            self._current_index += 1
            return server
        
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(healthy_servers, key=lambda s: s.active_connections)
        
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            # Seleccionar según peso
            total_weight = sum(s.weight for s in healthy_servers)
            if total_weight == 0:
                return healthy_servers[0]
            
            # Round robin con pesos
            server = healthy_servers[self._current_index % len(healthy_servers)]
            self._current_index += 1
            return server
        
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(healthy_servers)
        
        elif self.strategy == LoadBalanceStrategy.IP_HASH:
            if client_ip:
                index = hash(client_ip) % len(healthy_servers)
                return healthy_servers[index]
            return random.choice(healthy_servers)
        
        return healthy_servers[0]
    
    def mark_server_healthy(self, server_id: str):
        """Marca servidor como saludable"""
        if server_id in self._servers:
            self._servers[server_id].healthy = True
    
    def mark_server_unhealthy(self, server_id: str):
        """Marca servidor como no saludable"""
        if server_id in self._servers:
            self._servers[server_id].healthy = False
    
    def increment_connections(self, server_id: str):
        """Incrementa conexiones activas"""
        if server_id in self._servers:
            self._servers[server_id].active_connections += 1
            self._servers[server_id].last_used = datetime.utcnow()
    
    def decrement_connections(self, server_id: str):
        """Decrementa conexiones activas"""
        if server_id in self._servers:
            self._servers[server_id].active_connections = max(
                0,
                self._servers[server_id].active_connections - 1
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del load balancer"""
        return {
            "strategy": self.strategy.value,
            "total_servers": len(self._servers),
            "healthy_servers": sum(1 for s in self._servers.values() if s.healthy),
            "servers": [
                {
                    "server_id": s.server_id,
                    "url": s.url,
                    "healthy": s.healthy,
                    "active_connections": s.active_connections,
                    "weight": s.weight,
                }
                for s in self._servers.values()
            ],
        }

