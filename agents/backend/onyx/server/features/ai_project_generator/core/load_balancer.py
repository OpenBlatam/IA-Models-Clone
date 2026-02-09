"""
Load Balancer - Balanceador de carga avanzado
============================================

Balanceador de carga con múltiples estrategias:
- Round Robin
- Least Connections
- Weighted Round Robin
- IP Hash
- Health-based routing
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Protocol
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum

from .types import ServiceName, ServiceURL, IPAddress

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(str, Enum):
    """Estrategias de balanceo de carga"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    HEALTH_BASED = "health_based"


class BackendServer:
    """Servidor backend"""
    
    def __init__(
        self,
        url: ServiceURL,
        weight: int = 1,
        health_check_url: Optional[str] = None
    ) -> None:
        self.url = url
        self.weight = weight
        self.health_check_url = health_check_url or f"{url}/health"
        self.connections: int = 0
        self.is_healthy: bool = True
        self.last_health_check: Optional[datetime] = None
        self.response_times: List[float] = []
    
    def increment_connections(self) -> None:
        """Incrementa contador de conexiones"""
        self.connections += 1
    
    def decrement_connections(self) -> None:
        """Decrementa contador de conexiones"""
        self.connections = max(0, self.connections - 1)
    
    def record_response_time(self, time: float) -> None:
        """Registra tiempo de respuesta"""
        self.response_times.append(time)
        # Mantener solo últimos 100
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def get_avg_response_time(self) -> float:
        """Obtiene tiempo promedio de respuesta"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


class LoadBalancer:
    """
    Balanceador de carga avanzado con múltiples estrategias.
    """
    
    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        health_check_interval: int = 30
    ) -> None:
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.backends: Dict[ServiceName, List[BackendServer]] = defaultdict(list)
        self.current_indices: Dict[ServiceName, int] = defaultdict(int)
        self.connection_counts: Dict[ServiceURL, int] = defaultdict(int)
    
    def add_backend(
        self,
        service_name: ServiceName,
        url: ServiceURL,
        weight: int = 1,
        health_check_url: Optional[str] = None
    ) -> None:
        """Agrega servidor backend"""
        backend = BackendServer(url, weight, health_check_url)
        self.backends[service_name].append(backend)
        logger.info(f"Backend {url} added to service {service_name}")
    
    def remove_backend(
        self,
        service_name: ServiceName,
        url: ServiceURL
    ) -> None:
        """Remueve servidor backend"""
        backends = self.backends[service_name]
        self.backends[service_name] = [
            b for b in backends if b.url != url
        ]
        logger.info(f"Backend {url} removed from service {service_name}")
    
    def get_backend(
        self,
        service_name: ServiceName,
        client_ip: Optional[IPAddress] = None
    ) -> Optional[BackendServer]:
        """
        Obtiene servidor backend según estrategia.
        
        Args:
            service_name: Nombre del servicio
            client_ip: IP del cliente (para IP hash)
        
        Returns:
            Servidor backend seleccionado
        """
        backends = self.backends[service_name]
        if not backends:
            return None
        
        # Filtrar solo backends saludables
        healthy_backends = [b for b in backends if b.is_healthy]
        if not healthy_backends:
            # Si no hay saludables, usar todos
            healthy_backends = backends
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(service_name, healthy_backends)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_backends)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(service_name, healthy_backends)
        elif self.strategy == LoadBalanceStrategy.IP_HASH:
            return self._ip_hash(healthy_backends, client_ip)
        elif self.strategy == LoadBalanceStrategy.HEALTH_BASED:
            return self._health_based(healthy_backends)
        else:
            return healthy_backends[0]
    
    def _round_robin(
        self,
        service_name: ServiceName,
        backends: List[BackendServer]
    ) -> BackendServer:
        """Round robin"""
        index = self.current_indices[service_name]
        backend = backends[index % len(backends)]
        self.current_indices[service_name] = (index + 1) % len(backends)
        return backend
    
    def _least_connections(self, backends: List[BackendServer]) -> BackendServer:
        """Least connections"""
        return min(backends, key=lambda b: b.connections)
    
    def _weighted_round_robin(
        self,
        service_name: ServiceName,
        backends: List[BackendServer]
    ) -> BackendServer:
        """Weighted round robin"""
        total_weight = sum(b.weight for b in backends)
        index = self.current_indices[service_name]
        
        current_weight = 0
        for backend in backends:
            current_weight += backend.weight
            if index < current_weight:
                self.current_indices[service_name] = (index + 1) % total_weight
                return backend
        
        return backends[0]
    
    def _ip_hash(
        self,
        backends: List[BackendServer],
        client_ip: Optional[IPAddress]
    ) -> BackendServer:
        """IP hash"""
        if not client_ip:
            return backends[0]
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(backends)
        return backends[index]
    
    def _health_based(self, backends: List[BackendServer]) -> BackendServer:
        """Health-based routing"""
        # Seleccionar backend con mejor tiempo de respuesta y menos conexiones
        return min(
            backends,
            key=lambda b: (b.get_avg_response_time(), b.connections)
        )
    
    async def check_backend_health(self, backend: BackendServer) -> bool:
        """Verifica salud de un backend"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(backend.health_check_url)
                backend.is_healthy = response.status_code == 200
                backend.last_health_check = datetime.now()
                return backend.is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for {backend.url}: {e}")
            backend.is_healthy = False
            backend.last_health_check = datetime.now()
            return False
    
    async def check_all_backends(self) -> None:
        """Verifica salud de todos los backends"""
        for service_name, backends in self.backends.items():
            for backend in backends:
                await self.check_backend_health(backend)
    
    def get_stats(self, service_name: ServiceName) -> Dict[str, Any]:
        """Obtiene estadísticas de un servicio"""
        backends = self.backends[service_name]
        return {
            "service": service_name,
            "strategy": self.strategy.value,
            "total_backends": len(backends),
            "healthy_backends": sum(1 for b in backends if b.is_healthy),
            "backends": [
                {
                    "url": b.url,
                    "weight": b.weight,
                    "connections": b.connections,
                    "is_healthy": b.is_healthy,
                    "avg_response_time": b.get_avg_response_time()
                }
                for b in backends
            ]
        }


def get_load_balancer(
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
) -> LoadBalancer:
    """Obtiene balanceador de carga"""
    return LoadBalancer(strategy=strategy)















