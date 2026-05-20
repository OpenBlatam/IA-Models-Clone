"""
Load Balancing Utilities
========================
Utilidades para balanceo de carga.
"""

from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import random
import hashlib


class LoadBalancingStrategy(str, Enum):
    """Estrategias de balanceo de carga."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"


class Server:
    """Representación de un servidor."""
    
    def __init__(
        self,
        id: str,
        url: str,
        weight: int = 1,
        health_check: Optional[Callable] = None
    ):
        """
        Inicializar servidor.
        
        Args:
            id: ID único del servidor
            url: URL del servidor
            weight: Peso para balanceo (1-100)
            health_check: Función para health check
        """
        self.id = id
        self.url = url
        self.weight = weight
        self.health_check = health_check
        self.active_connections = 0
        self.is_healthy = True
        self.response_times: List[float] = []
    
    def record_connection(self):
        """Registrar nueva conexión."""
        self.active_connections += 1
    
    def release_connection(self):
        """Liberar conexión."""
        self.active_connections = max(0, self.active_connections - 1)
    
    def record_response_time(self, time: float):
        """Registrar tiempo de respuesta."""
        self.response_times.append(time)
        # Mantener solo últimos 100
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def get_avg_response_time(self) -> float:
        """Obtener tiempo promedio de respuesta."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


class LoadBalancer:
    """Balanceador de carga."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        """
        Inicializar balanceador.
        
        Args:
            strategy: Estrategia de balanceo
        """
        self.strategy = strategy
        self.servers: List[Server] = []
        self.current_index = 0
    
    def add_server(self, server: Server):
        """Agregar servidor."""
        self.servers.append(server)
    
    def remove_server(self, server_id: str):
        """Remover servidor."""
        self.servers = [s for s in self.servers if s.id != server_id]
    
    def get_server(self, context: Optional[Dict[str, Any]] = None) -> Optional[Server]:
        """
        Obtener servidor según estrategia.
        
        Args:
            context: Contexto adicional (ej: IP del cliente)
            
        Returns:
            Servidor seleccionado
        """
        # Filtrar servidores saludables
        healthy_servers = [s for s in self.servers if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            server = healthy_servers[self.current_index % len(healthy_servers)]
            self.current_index += 1
            return server
        
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_servers)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            total_weight = sum(s.weight for s in healthy_servers)
            if total_weight == 0:
                return random.choice(healthy_servers)
            
            # Selección basada en peso
            rand = random.uniform(0, total_weight)
            cumulative = 0
            for server in healthy_servers:
                cumulative += server.weight
                if rand <= cumulative:
                    return server
            
            return healthy_servers[0]
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_servers, key=lambda s: s.active_connections)
        
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            if context and "ip" in context:
                ip_hash = int(hashlib.md5(context["ip"].encode()).hexdigest(), 16)
                index = ip_hash % len(healthy_servers)
                return healthy_servers[index]
            return random.choice(healthy_servers)
        
        return healthy_servers[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del balanceador."""
        return {
            "total_servers": len(self.servers),
            "healthy_servers": sum(1 for s in self.servers if s.is_healthy),
            "strategy": self.strategy.value,
            "servers": [
                {
                    "id": s.id,
                    "url": s.url,
                    "weight": s.weight,
                    "active_connections": s.active_connections,
                    "is_healthy": s.is_healthy,
                    "avg_response_time": s.get_avg_response_time()
                }
                for s in self.servers
            ]
        }

