"""
Sistema de Load Balancing y Distribución de Carga

Proporciona:
- Round-robin
- Least connections
- Weighted round-robin
- Health-based routing
- Failover automático
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Estrategias de load balancing"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"
    RANDOM = "random"


@dataclass
class Backend:
    """Backend del load balancer"""
    id: str
    url: str
    weight: int = 1
    health_check_url: Optional[str] = None
    healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_health_check: Optional[float] = None
    response_times: List[float] = field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Calcula tasa de éxito"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    def get_avg_response_time(self) -> float:
        """Calcula tiempo de respuesta promedio"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


class LoadBalancer:
    """Load balancer avanzado"""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        health_check_interval: int = 30
    ):
        """
        Args:
            strategy: Estrategia de load balancing
            health_check_interval: Intervalo de health checks en segundos
        """
        self.strategy = strategy
        self.backends: Dict[str, Backend] = {}
        self.current_index: Dict[str, int] = defaultdict(int)
        self.health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info(f"LoadBalancer initialized with strategy: {strategy.value}")
    
    def add_backend(
        self,
        backend_id: str,
        url: str,
        weight: int = 1,
        health_check_url: Optional[str] = None
    ):
        """
        Agrega un backend
        
        Args:
            backend_id: ID del backend
            url: URL del backend
            weight: Peso para weighted round-robin
            health_check_url: URL para health check
        """
        backend = Backend(
            id=backend_id,
            url=url,
            weight=weight,
            health_check_url=health_check_url or f"{url}/health"
        )
        
        self.backends[backend_id] = backend
        logger.info(f"Backend added: {backend_id} at {url}")
    
    def remove_backend(self, backend_id: str):
        """Elimina un backend"""
        if backend_id in self.backends:
            del self.backends[backend_id]
            logger.info(f"Backend removed: {backend_id}")
    
    def get_backend(self, service_name: str = "default") -> Optional[Backend]:
        """
        Obtiene un backend según la estrategia
        
        Args:
            service_name: Nombre del servicio
        
        Returns:
            Backend seleccionado
        """
        # Filtrar backends saludables
        healthy_backends = [
            b for b in self.backends.values()
            if b.healthy
        ]
        
        if not healthy_backends:
            # Si no hay backends saludables, usar todos
            healthy_backends = list(self.backends.values())
        
        if not healthy_backends:
            return None
        
        # Seleccionar según estrategia
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            backend = self._round_robin(healthy_backends, service_name)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            backend = self._least_connections(healthy_backends)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            backend = self._weighted_round_robin(healthy_backends, service_name)
        
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            backend = self._health_based(healthy_backends)
        
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            import random
            backend = random.choice(healthy_backends)
        
        else:
            backend = healthy_backends[0]
        
        return backend
    
    def _round_robin(self, backends: List[Backend], service_name: str) -> Backend:
        """Round-robin"""
        index = self.current_index[service_name] % len(backends)
        self.current_index[service_name] += 1
        return backends[index]
    
    def _least_connections(self, backends: List[Backend]) -> Backend:
        """Least connections"""
        return min(backends, key=lambda b: b.active_connections)
    
    def _weighted_round_robin(self, backends: List[Backend], service_name: str) -> Backend:
        """Weighted round-robin"""
        # Calcular peso total
        total_weight = sum(b.weight for b in backends)
        
        # Obtener índice actual
        current = self.current_index[service_name]
        
        # Encontrar backend según peso
        cumulative = 0
        for backend in backends:
            cumulative += backend.weight
            if current < cumulative:
                self.current_index[service_name] = (current + 1) % total_weight
                return backend
        
        # Fallback
        self.current_index[service_name] = (current + 1) % total_weight
        return backends[0]
    
    def _health_based(self, backends: List[Backend]) -> Backend:
        """Health-based (mejor tasa de éxito y menor tiempo de respuesta)"""
        scored_backends = []
        for backend in backends:
            success_rate = backend.get_success_rate()
            avg_time = backend.get_avg_response_time()
            
            # Score: mayor éxito, menor tiempo
            score = success_rate * 100 - avg_time
            scored_backends.append((score, backend))
        
        # Ordenar por score y devolver el mejor
        scored_backends.sort(key=lambda x: x[0], reverse=True)
        return scored_backends[0][1] if scored_backends else backends[0]
    
    def record_request(
        self,
        backend_id: str,
        success: bool,
        response_time: float
    ):
        """Registra una request"""
        backend = self.backends.get(backend_id)
        if backend:
            backend.total_requests += 1
            if not success:
                backend.failed_requests += 1
            backend.response_times.append(response_time)
            
            # Mantener solo últimos 100 tiempos
            if len(backend.response_times) > 100:
                backend.response_times = backend.response_times[-100:]
    
    def increment_connections(self, backend_id: str):
        """Incrementa conexiones activas"""
        backend = self.backends.get(backend_id)
        if backend:
            backend.active_connections += 1
    
    def decrement_connections(self, backend_id: str):
        """Decrementa conexiones activas"""
        backend = self.backends.get(backend_id)
        if backend:
            backend.active_connections = max(0, backend.active_connections - 1)
    
    async def start_health_checks(self):
        """Inicia health checks periódicos"""
        while True:
            await asyncio.sleep(self.health_check_interval)
            await self._perform_health_checks()
    
    async def _perform_health_checks(self):
        """Realiza health checks en todos los backends"""
        import httpx
        
        for backend in self.backends.values():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(backend.health_check_url)
                    backend.healthy = response.status_code == 200
                    backend.last_health_check = time.time()
            except Exception as e:
                backend.healthy = False
                logger.warning(f"Health check failed for {backend.id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del load balancer"""
        return {
            "strategy": self.strategy.value,
            "total_backends": len(self.backends),
            "healthy_backends": sum(1 for b in self.backends.values() if b.healthy),
            "backends": [
                {
                    "id": b.id,
                    "url": b.url,
                    "healthy": b.healthy,
                    "active_connections": b.active_connections,
                    "total_requests": b.total_requests,
                    "success_rate": b.get_success_rate(),
                    "avg_response_time": b.get_avg_response_time()
                }
                for b in self.backends.values()
            ]
        }


# Instancia global
_load_balancer: Optional[LoadBalancer] = None


def get_load_balancer(
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
) -> LoadBalancer:
    """Obtiene la instancia global del load balancer"""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer(strategy=strategy)
    return _load_balancer

