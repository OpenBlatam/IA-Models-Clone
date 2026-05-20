"""
Intelligent Load Balancer - Balanceador de Carga Inteligente
=============================================================

Sistema de balanceo de carga inteligente con algoritmos adaptativos y detección de salud de nodos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import statistics
import random

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Algoritmo de balanceo."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE = "adaptive"


@dataclass
class Node:
    """Nodo del balanceador."""
    node_id: str
    address: str
    weight: float = 1.0
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time_ms: float = 0.0
    health_status: str = "healthy"
    last_health_check: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    enabled: bool = True


class IntelligentLoadBalancer:
    """Balanceador de carga inteligente."""
    
    def __init__(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE,
        health_check_interval: float = 30.0,
    ):
        self.nodes: Dict[str, Node] = {}
        self.algorithm = algorithm
        self.health_check_interval = health_check_interval
        self.current_index: Dict[str, int] = {}
        self.node_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = asyncio.Lock()
    
    def add_node(
        self,
        node_id: str,
        address: str,
        weight: float = 1.0,
    ):
        """Agregar nodo al balanceador."""
        node = Node(
            node_id=node_id,
            address=address,
            weight=weight,
        )
        
        self.nodes[node_id] = node
        self.current_index[node_id] = 0
        
        logger.info(f"Added node {node_id} at {address} with weight {weight}")
    
    def remove_node(self, node_id: str):
        """Remover nodo del balanceador."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            if node_id in self.current_index:
                del self.current_index[node_id]
            logger.info(f"Removed node {node_id}")
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Obtener nodo por ID."""
        return self.nodes.get(node_id)
    
    def select_node(self, request_id: Optional[str] = None) -> Optional[Node]:
        """
        Seleccionar nodo usando algoritmo de balanceo.
        
        Args:
            request_id: ID de la petición (opcional)
        
        Returns:
            Nodo seleccionado
        """
        healthy_nodes = [
            node for node in self.nodes.values()
            if node.enabled and node.health_status == "healthy"
        ]
        
        if not healthy_nodes:
            logger.warning("No healthy nodes available")
            return None
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_select(healthy_nodes)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_nodes)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_nodes)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(healthy_nodes)
        elif self.algorithm == LoadBalancingAlgorithm.ADAPTIVE:
            return self._adaptive_select(healthy_nodes)
        
        return healthy_nodes[0]
    
    def _round_robin_select(self, nodes: List[Node]) -> Node:
        """Selección round-robin."""
        if not nodes:
            return None
        
        # Usar índice global o por algoritmo
        if "round_robin" not in self.current_index:
            self.current_index["round_robin"] = 0
        
        node = nodes[self.current_index["round_robin"] % len(nodes)]
        self.current_index["round_robin"] = (self.current_index["round_robin"] + 1) % len(nodes)
        
        return node
    
    def _least_connections_select(self, nodes: List[Node]) -> Node:
        """Selección por menor número de conexiones."""
        return min(nodes, key=lambda n: n.active_connections)
    
    def _weighted_round_robin_select(self, nodes: List[Node]) -> Node:
        """Selección round-robin con pesos."""
        total_weight = sum(n.weight for n in nodes)
        if total_weight == 0:
            return self._round_robin_select(nodes)
        
        # Selección probabilística basada en pesos
        r = random.random() * total_weight
        cumulative = 0.0
        
        for node in nodes:
            cumulative += node.weight
            if r <= cumulative:
                return node
        
        return nodes[-1]
    
    def _least_response_time_select(self, nodes: List[Node]) -> Node:
        """Selección por menor tiempo de respuesta."""
        return min(nodes, key=lambda n: n.avg_response_time_ms if n.avg_response_time_ms > 0 else float('inf'))
    
    def _adaptive_select(self, nodes: List[Node]) -> Node:
        """Selección adaptativa combinando múltiples factores."""
        # Calcular score para cada nodo
        scores = []
        
        for node in nodes:
            # Factor de conexiones (menor es mejor)
            connection_score = 1.0 / (node.active_connections + 1)
            
            # Factor de tiempo de respuesta (menor es mejor)
            if node.avg_response_time_ms > 0:
                response_score = 1000.0 / (node.avg_response_time_ms + 1)
            else:
                response_score = 1.0
            
            # Factor de errores (menor es mejor)
            error_rate = node.total_errors / max(node.total_requests, 1)
            error_score = 1.0 - error_rate
            
            # Factor de peso
            weight_score = node.weight
            
            # Score combinado
            total_score = (
                connection_score * 0.3 +
                response_score * 0.3 +
                error_score * 0.2 +
                weight_score * 0.2
            )
            
            scores.append((total_score, node))
        
        # Seleccionar nodo con mayor score
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1] if scores else None
    
    async def record_request(
        self,
        node_id: str,
        response_time_ms: float,
        success: bool = True,
    ):
        """Registrar petición procesada."""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        node.total_requests += 1
        if not success:
            node.total_errors += 1
        
        # Actualizar tiempo de respuesta promedio
        node.response_times.append(response_time_ms)
        if node.response_times:
            node.avg_response_time_ms = statistics.mean(node.response_times)
        
        # Registrar en estadísticas
        self.node_stats[node_id].append({
            "timestamp": datetime.now(),
            "response_time_ms": response_time_ms,
            "success": success,
        })
    
    async def update_node_health(
        self,
        node_id: str,
        health_status: str,
    ):
        """Actualizar estado de salud del nodo."""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        node.health_status = health_status
        node.last_health_check = datetime.now()
        
        if health_status != "healthy":
            node.enabled = False
        else:
            node.enabled = True
    
    def get_node_stats(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de nodos."""
        if node_id:
            node = self.nodes.get(node_id)
            if not node:
                return {}
            
            stats = list(self.node_stats[node_id])
            recent_stats = stats[-100:] if len(stats) > 100 else stats
            
            return {
                "node_id": node_id,
                "address": node.address,
                "weight": node.weight,
                "active_connections": node.active_connections,
                "total_requests": node.total_requests,
                "total_errors": node.total_errors,
                "error_rate": node.total_errors / max(node.total_requests, 1),
                "avg_response_time_ms": node.avg_response_time_ms,
                "health_status": node.health_status,
                "enabled": node.enabled,
                "recent_requests": len(recent_stats),
            }
        
        return {
            node_id: self.get_node_stats(node_id)
            for node_id in self.nodes.keys()
        }
    
    def get_load_balancer_summary(self) -> Dict[str, Any]:
        """Obtener resumen del balanceador."""
        healthy_nodes = sum(1 for n in self.nodes.values() if n.health_status == "healthy")
        total_requests = sum(n.total_requests for n in self.nodes.values())
        total_errors = sum(n.total_errors for n in self.nodes.values())
        
        return {
            "algorithm": self.algorithm.value,
            "total_nodes": len(self.nodes),
            "healthy_nodes": healthy_nodes,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
            "avg_response_time_ms": statistics.mean([
                n.avg_response_time_ms for n in self.nodes.values()
                if n.avg_response_time_ms > 0
            ]) if any(n.avg_response_time_ms > 0 for n in self.nodes.values()) else 0.0,
        }
















