"""
Sistema de Edge Computing
==========================

Sistema para procesamiento en el edge (borde de la red).
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EdgeNodeStatus(Enum):
    """Estado de nodo edge"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"


@dataclass
class EdgeNode:
    """Nodo edge"""
    node_id: str
    location: str
    capacity: int
    latency_ms: float
    status: EdgeNodeStatus
    last_heartbeat: str


class EdgeComputingSystem:
    """
    Sistema de edge computing
    
    Proporciona:
    - Distribución de procesamiento a nodos edge
    - Selección de nodo más cercano
    - Balanceo de carga en edge
    - Sincronización con cloud
    - Procesamiento offline
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.nodes: Dict[str, EdgeNode] = {}
        self.task_queue: List[Dict[str, Any]] = []
        logger.info("EdgeComputingSystem inicializado")
    
    def register_node(
        self,
        node_id: str,
        location: str,
        capacity: int = 100,
        latency_ms: float = 10.0
    ) -> EdgeNode:
        """Registrar nodo edge"""
        node = EdgeNode(
            node_id=node_id,
            location=location,
            capacity=capacity,
            latency_ms=latency_ms,
            status=EdgeNodeStatus.ONLINE,
            last_heartbeat=datetime.now().isoformat()
        )
        
        self.nodes[node_id] = node
        logger.info(f"Nodo edge registrado: {node_id} en {location}")
        
        return node
    
    def select_best_node(
        self,
        user_location: Optional[str] = None
    ) -> Optional[EdgeNode]:
        """Seleccionar mejor nodo edge"""
        online_nodes = [
            n for n in self.nodes.values()
            if n.status == EdgeNodeStatus.ONLINE
        ]
        
        if not online_nodes:
            return None
        
        # Seleccionar por menor latencia
        best_node = min(online_nodes, key=lambda x: x.latency_ms)
        
        return best_node
    
    def submit_task(
        self,
        task: Dict[str, Any],
        preferred_node: Optional[str] = None
    ) -> str:
        """
        Enviar tarea a nodo edge
        
        Args:
            task: Tarea a procesar
            preferred_node: Nodo preferido (opcional)
        
        Returns:
            ID de tarea
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        if preferred_node and preferred_node in self.nodes:
            node = self.nodes[preferred_node]
            if node.status == EdgeNodeStatus.ONLINE:
                task["assigned_node"] = preferred_node
                task["task_id"] = task_id
                self.task_queue.append(task)
                logger.info(f"Tarea {task_id} asignada a nodo {preferred_node}")
                return task_id
        
        # Seleccionar mejor nodo
        best_node = self.select_best_node()
        if best_node:
            task["assigned_node"] = best_node.node_id
            task["task_id"] = task_id
            self.task_queue.append(task)
            logger.info(f"Tarea {task_id} asignada a nodo {best_node.node_id}")
            return task_id
        
        # Sin nodos disponibles, agregar a cola
        task["task_id"] = task_id
        self.task_queue.append(task)
        logger.warning(f"Tarea {task_id} agregada a cola (sin nodos disponibles)")
        
        return task_id
    
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de nodo"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        
        return {
            "node_id": node_id,
            "location": node.location,
            "status": node.status.value,
            "capacity": node.capacity,
            "latency_ms": node.latency_ms,
            "last_heartbeat": node.last_heartbeat
        }
    
    def list_nodes(self) -> List[Dict[str, Any]]:
        """Listar todos los nodos"""
        return [
            {
                "node_id": n.node_id,
                "location": n.location,
                "status": n.status.value,
                "capacity": n.capacity,
                "latency_ms": n.latency_ms
            }
            for n in self.nodes.values()
        ]


# Instancia global
_edge_computing: Optional[EdgeComputingSystem] = None


def get_edge_computing() -> EdgeComputingSystem:
    """Obtener instancia global del sistema"""
    global _edge_computing
    if _edge_computing is None:
        _edge_computing = EdgeComputingSystem()
    return _edge_computing














