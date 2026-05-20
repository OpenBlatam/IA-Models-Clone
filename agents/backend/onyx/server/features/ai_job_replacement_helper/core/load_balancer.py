"""
Load Balancer Service - Balanceador de carga
=============================================

Sistema de balanceo de carga con múltiples algoritmos.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(str, Enum):
    """Algoritmos de balanceo"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    RANDOM = "random"


@dataclass
class BackendServer:
    """Servidor backend"""
    id: str
    host: str
    port: int
    weight: int = 1
    active_connections: int = 0
    total_requests: int = 0
    health_status: str = "healthy"
    last_request_time: Optional[datetime] = None


@dataclass
class LoadBalancer:
    """Balanceador de carga"""
    name: str
    algorithm: LoadBalancingAlgorithm
    backends: List[BackendServer]
    current_index: int = 0  # Para round robin


class LoadBalancerService:
    """Servicio de balanceador de carga"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.balancers: Dict[str, LoadBalancer] = {}
        logger.info("LoadBalancerService initialized")
    
    def create_load_balancer(
        self,
        name: str,
        algorithm: LoadBalancingAlgorithm,
        backends: List[Dict[str, Any]]
    ) -> LoadBalancer:
        """Crear balanceador de carga"""
        backend_servers = [
            BackendServer(
                id=f"backend_{i}",
                host=b["host"],
                port=b["port"],
                weight=b.get("weight", 1),
            )
            for i, b in enumerate(backends)
        ]
        
        balancer = LoadBalancer(
            name=name,
            algorithm=algorithm,
            backends=backend_servers,
        )
        
        self.balancers[name] = balancer
        
        logger.info(f"Load balancer created: {name} with {len(backend_servers)} backends")
        return balancer
    
    def select_backend(
        self,
        balancer_name: str,
        client_ip: Optional[str] = None
    ) -> Optional[BackendServer]:
        """Seleccionar backend según algoritmo"""
        balancer = self.balancers.get(balancer_name)
        if not balancer:
            return None
        
        healthy_backends = [
            b for b in balancer.backends
            if b.health_status == "healthy"
        ]
        
        if not healthy_backends:
            return None
        
        if balancer.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            backend = healthy_backends[balancer.current_index % len(healthy_backends)]
            balancer.current_index += 1
            return backend
        
        elif balancer.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return min(healthy_backends, key=lambda b: b.active_connections)
        
        elif balancer.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            # Selección ponderada (simplificada)
            total_weight = sum(b.weight for b in healthy_backends)
            backend = healthy_backends[balancer.current_index % len(healthy_backends)]
            balancer.current_index += 1
            return backend
        
        elif balancer.algorithm == LoadBalancingAlgorithm.IP_HASH:
            if client_ip:
                import hashlib
                hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
                return healthy_backends[hash_value % len(healthy_backends)]
            return healthy_backends[0]
        
        elif balancer.algorithm == LoadBalancingAlgorithm.RANDOM:
            import random
            return random.choice(healthy_backends)
        
        return healthy_backends[0]
    
    def record_request(
        self,
        balancer_name: str,
        backend_id: str
    ):
        """Registrar request en backend"""
        balancer = self.balancers.get(balancer_name)
        if not balancer:
            return
        
        backend = next((b for b in balancer.backends if b.id == backend_id), None)
        if backend:
            backend.active_connections += 1
            backend.total_requests += 1
            backend.last_request_time = datetime.now()
    
    def record_response(
        self,
        balancer_name: str,
        backend_id: str
    ):
        """Registrar respuesta de backend"""
        balancer = self.balancers.get(balancer_name)
        if not balancer:
            return
        
        backend = next((b for b in balancer.backends if b.id == backend_id), None)
        if backend:
            backend.active_connections = max(0, backend.active_connections - 1)
    
    def get_balancer_stats(self, balancer_name: str) -> Dict[str, Any]:
        """Obtener estadísticas del balanceador"""
        balancer = self.balancers.get(balancer_name)
        if not balancer:
            return {"exists": False}
        
        return {
            "name": balancer.name,
            "algorithm": balancer.algorithm.value,
            "total_backends": len(balancer.backends),
            "healthy_backends": sum(1 for b in balancer.backends if b.health_status == "healthy"),
            "total_requests": sum(b.total_requests for b in balancer.backends),
            "active_connections": sum(b.active_connections for b in balancer.backends),
            "backends": [
                {
                    "id": b.id,
                    "host": b.host,
                    "port": b.port,
                    "weight": b.weight,
                    "active_connections": b.active_connections,
                    "total_requests": b.total_requests,
                    "health_status": b.health_status,
                }
                for b in balancer.backends
            ],
        }




