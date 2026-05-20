"""
Load Balancer - Sistema de load balancing interno
==================================================
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum
import time

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(str, Enum):
    """Estrategias de load balancing"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"


class ServerNode:
    """Nodo del servidor"""
    
    def __init__(self, node_id: str, weight: int = 1):
        self.node_id = node_id
        self.weight = weight
        self.active_connections = 0
        self.total_requests = 0
        self.total_response_time = 0.0
        self.healthy = True
        self.last_response_time = 0.0
    
    def get_average_response_time(self) -> float:
        """Obtiene tiempo promedio de respuesta"""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests


class LoadBalancer:
    """Sistema de load balancing interno"""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.nodes: List[ServerNode] = []
        self.current_index = 0
        self.request_count = 0
    
    def add_node(self, node_id: str, weight: int = 1) -> ServerNode:
        """Agrega un nodo"""
        node = ServerNode(node_id, weight)
        self.nodes.append(node)
        logger.info(f"Nodo agregado: {node_id} (peso: {weight})")
        return node
    
    def remove_node(self, node_id: str):
        """Remueve un nodo"""
        self.nodes = [n for n in self.nodes if n.node_id != node_id]
        logger.info(f"Nodo removido: {node_id}")
    
    def get_next_node(self) -> Optional[ServerNode]:
        """Obtiene el siguiente nodo según la estrategia"""
        healthy_nodes = [n for n in self.nodes if n.healthy]
        
        if not healthy_nodes:
            logger.warning("No hay nodos saludables disponibles")
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(healthy_nodes)
        
        return healthy_nodes[0]
    
    def _round_robin(self, nodes: List[ServerNode]) -> ServerNode:
        """Round robin"""
        node = nodes[self.current_index % len(nodes)]
        self.current_index += 1
        return node
    
    def _least_connections(self, nodes: List[ServerNode]) -> ServerNode:
        """Least connections"""
        return min(nodes, key=lambda n: n.active_connections)
    
    def _weighted_round_robin(self, nodes: List[ServerNode]) -> ServerNode:
        """Weighted round robin"""
        total_weight = sum(n.weight for n in nodes)
        current_weight = 0
        selected = nodes[0]
        
        for node in nodes:
            current_weight += node.weight
            if (self.request_count % total_weight) < current_weight:
                selected = node
                break
        
        self.request_count += 1
        return selected
    
    def _least_response_time(self, nodes: List[ServerNode]) -> ServerNode:
        """Least response time"""
        return min(nodes, key=lambda n: n.get_average_response_time())
    
    def record_request(self, node_id: str, response_time: float):
        """Registra una solicitud"""
        node = next((n for n in self.nodes if n.node_id == node_id), None)
        if node:
            node.active_connections = max(0, node.active_connections - 1)
            node.total_requests += 1
            node.total_response_time += response_time
            node.last_response_time = response_time
    
    def start_request(self, node_id: str):
        """Marca inicio de solicitud"""
        node = next((n for n in self.nodes if n.node_id == node_id), None)
        if node:
            node.active_connections += 1
    
    def set_node_health(self, node_id: str, healthy: bool):
        """Establece salud de un nodo"""
        node = next((n for n in self.nodes if n.node_id == node_id), None)
        if node:
            node.healthy = healthy
            logger.info(f"Nodo {node_id} marcado como {'saludable' if healthy else 'no saludable'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return {
            "strategy": self.strategy.value,
            "total_nodes": len(self.nodes),
            "healthy_nodes": sum(1 for n in self.nodes if n.healthy),
            "nodes": [
                {
                    "id": n.node_id,
                    "weight": n.weight,
                    "active_connections": n.active_connections,
                    "total_requests": n.total_requests,
                    "avg_response_time": n.get_average_response_time(),
                    "healthy": n.healthy
                }
                for n in self.nodes
            ]
        }




