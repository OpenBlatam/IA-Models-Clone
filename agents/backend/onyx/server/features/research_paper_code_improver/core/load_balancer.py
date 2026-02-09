"""
Load Balancer - Balanceador de carga
=====================================
"""

import logging
import random
import time
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Estrategias de balanceo"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


@dataclass
class BackendServer:
    """Servidor backend"""
    id: str
    host: str
    port: int
    protocol: str = "http"
    weight: int = 1
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    health_status: str = "healthy"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "weight": self.weight,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "avg_response_time": self.avg_response_time,
            "health_status": self.health_status,
            "url": f"{self.protocol}://{self.host}:{self.port}"
        }


class LoadBalancer:
    """Balanceador de carga"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.backends: List[BackendServer] = []
        self.current_index: int = 0
        self.stats: Dict[str, Dict[str, Any]] = {}
    
    def add_backend(
        self,
        host: str,
        port: int,
        backend_id: Optional[str] = None,
        protocol: str = "http",
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BackendServer:
        """Agrega un servidor backend"""
        if backend_id is None:
            import uuid
            backend_id = str(uuid.uuid4())
        
        backend = BackendServer(
            id=backend_id,
            host=host,
            port=port,
            protocol=protocol,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.backends.append(backend)
        logger.info(f"Backend {backend_id} agregado: {host}:{port}")
        return backend
    
    def remove_backend(self, backend_id: str) -> bool:
        """Remueve un servidor backend"""
        backend = next((b for b in self.backends if b.id == backend_id), None)
        if backend:
            self.backends.remove(backend)
            logger.info(f"Backend {backend_id} removido")
            return True
        return False
    
    def get_backend(self, client_ip: Optional[str] = None) -> Optional[BackendServer]:
        """Obtiene un backend según la estrategia"""
        # Filtrar solo backends saludables
        healthy_backends = [b for b in self.backends if b.health_status == "healthy"]
        
        if not healthy_backends:
            logger.warning("No hay backends saludables disponibles")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            backend = healthy_backends[self.current_index % len(healthy_backends)]
            self.current_index += 1
            return backend
        
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_backends)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_backends, key=lambda b: b.active_connections)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Seleccionar basado en peso
            total_weight = sum(b.weight for b in healthy_backends)
            if total_weight == 0:
                return random.choice(healthy_backends)
            
            rand = random.uniform(0, total_weight)
            current = 0
            for backend in healthy_backends:
                current += backend.weight
                if rand <= current:
                    return backend
            return healthy_backends[0]
        
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            if client_ip:
                hash_value = hash(client_ip) % len(healthy_backends)
                return healthy_backends[hash_value]
            return random.choice(healthy_backends)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(healthy_backends, key=lambda b: b.avg_response_time)
        
        return healthy_backends[0]
    
    def record_request(
        self,
        backend_id: str,
        success: bool,
        response_time: float
    ):
        """Registra una petición"""
        backend = next((b for b in self.backends if b.id == backend_id), None)
        if not backend:
            return
        
        backend.total_requests += 1
        backend.last_request_time = datetime.now()
        
        if not success:
            backend.failed_requests += 1
        
        # Actualizar tiempo promedio de respuesta
        if backend.total_requests == 1:
            backend.avg_response_time = response_time
        else:
            backend.avg_response_time = (
                (backend.avg_response_time * (backend.total_requests - 1) + response_time) /
                backend.total_requests
            )
    
    def increment_connections(self, backend_id: str):
        """Incrementa conexiones activas"""
        backend = next((b for b in self.backends if b.id == backend_id), None)
        if backend:
            backend.active_connections += 1
    
    def decrement_connections(self, backend_id: str):
        """Decrementa conexiones activas"""
        backend = next((b for b in self.backends if b.id == backend_id), None)
        if backend:
            backend.active_connections = max(0, backend.active_connections - 1)
    
    def mark_unhealthy(self, backend_id: str):
        """Marca un backend como no saludable"""
        backend = next((b for b in self.backends if b.id == backend_id), None)
        if backend:
            backend.health_status = "unhealthy"
            logger.warning(f"Backend {backend_id} marcado como unhealthy")
    
    def mark_healthy(self, backend_id: str):
        """Marca un backend como saludable"""
        backend = next((b for b in self.backends if b.id == backend_id), None)
        if backend:
            backend.health_status = "healthy"
            logger.info(f"Backend {backend_id} marcado como healthy")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del balanceador"""
        total_requests = sum(b.total_requests for b in self.backends)
        total_failed = sum(b.failed_requests for b in self.backends)
        healthy_count = len([b for b in self.backends if b.health_status == "healthy"])
        
        return {
            "total_backends": len(self.backends),
            "healthy_backends": healthy_count,
            "unhealthy_backends": len(self.backends) - healthy_count,
            "total_requests": total_requests,
            "total_failed": total_failed,
            "success_rate": (total_requests - total_failed) / total_requests if total_requests > 0 else 0,
            "backends": [b.to_dict() for b in self.backends]
        }
    
    def list_backends(self) -> List[Dict[str, Any]]:
        """Lista todos los backends"""
        return [b.to_dict() for b in self.backends]




