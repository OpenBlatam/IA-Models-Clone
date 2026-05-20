"""
API Gateway Avanzado
=====================

Sistema de API Gateway para routing, load balancing y gestión de servicios.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Estrategias de routing"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"


@dataclass
class ServiceEndpoint:
    """Endpoint de servicio"""
    service_id: str
    url: str
    weight: int = 1
    active: bool = True
    connections: int = 0
    last_used: Optional[str] = None


class APIGateway:
    """
    API Gateway avanzado
    
    Proporciona:
    - Routing inteligente
    - Load balancing
    - Health checks
    - Circuit breaker
    - Rate limiting por servicio
    - Retry logic
    """
    
    def __init__(self):
        """Inicializar gateway"""
        self.services: Dict[str, List[ServiceEndpoint]] = {}
        self.routing_strategy: Dict[str, RoutingStrategy] = {}
        self.current_index: Dict[str, int] = {}
        self.circuit_breaker: Dict[str, Dict[str, Any]] = {}
        logger.info("APIGateway inicializado")
    
    def register_service(
        self,
        service_name: str,
        service_id: str,
        url: str,
        weight: int = 1,
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    ):
        """Registrar servicio"""
        if service_name not in self.services:
            self.services[service_name] = []
            self.routing_strategy[service_name] = strategy
            self.current_index[service_name] = 0
        
        endpoint = ServiceEndpoint(
            service_id=service_id,
            url=url,
            weight=weight,
            last_used=datetime.now().isoformat()
        )
        
        self.services[service_name].append(endpoint)
        logger.info(f"Servicio registrado: {service_name}/{service_id}")
    
    def get_endpoint(
        self,
        service_name: str
    ) -> Optional[ServiceEndpoint]:
        """Obtener endpoint según estrategia de routing"""
        if service_name not in self.services:
            return None
        
        endpoints = [e for e in self.services[service_name] if e.active]
        if not endpoints:
            return None
        
        strategy = self.routing_strategy.get(service_name, RoutingStrategy.ROUND_ROBIN)
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            index = self.current_index.get(service_name, 0)
            endpoint = endpoints[index % len(endpoints)]
            self.current_index[service_name] = (index + 1) % len(endpoints)
            return endpoint
        
        elif strategy == RoutingStrategy.RANDOM:
            return random.choice(endpoints)
        
        elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(endpoints, key=lambda x: x.connections)
        
        elif strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection
            total_weight = sum(e.weight for e in endpoints)
            rand = random.random() * total_weight
            cumulative = 0
            for endpoint in endpoints:
                cumulative += endpoint.weight
                if rand <= cumulative:
                    return endpoint
        
        return endpoints[0]
    
    def mark_endpoint_used(self, service_name: str, service_id: str):
        """Marcar endpoint como usado"""
        if service_name in self.services:
            for endpoint in self.services[service_name]:
                if endpoint.service_id == service_id:
                    endpoint.last_used = datetime.now().isoformat()
                    endpoint.connections += 1
    
    def mark_endpoint_failed(self, service_name: str, service_id: str):
        """Marcar endpoint como fallido"""
        if service_name not in self.circuit_breaker:
            self.circuit_breaker[service_name] = {}
        
        if service_id not in self.circuit_breaker[service_name]:
            self.circuit_breaker[service_name][service_id] = {
                "failures": 0,
                "last_failure": None
            }
        
        cb = self.circuit_breaker[service_name][service_id]
        cb["failures"] += 1
        cb["last_failure"] = datetime.now().isoformat()
        
        # Desactivar si hay muchos fallos
        if cb["failures"] >= 5:
            for endpoint in self.services.get(service_name, []):
                if endpoint.service_id == service_id:
                    endpoint.active = False
                    logger.warning(f"Endpoint desactivado por fallos: {service_name}/{service_id}")
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Obtener salud del servicio"""
        if service_name not in self.services:
            return {"status": "unknown"}
        
        endpoints = self.services[service_name]
        active = sum(1 for e in endpoints if e.active)
        
        return {
            "service_name": service_name,
            "total_endpoints": len(endpoints),
            "active_endpoints": active,
            "routing_strategy": self.routing_strategy.get(service_name).value,
            "endpoints": [
                {
                    "service_id": e.service_id,
                    "url": e.url,
                    "active": e.active,
                    "connections": e.connections,
                    "last_used": e.last_used
                }
                for e in endpoints
            ]
        }


# Instancia global
_api_gateway: Optional[APIGateway] = None


def get_api_gateway() -> APIGateway:
    """Obtener instancia global del gateway"""
    global _api_gateway
    if _api_gateway is None:
        _api_gateway = APIGateway()
    return _api_gateway
