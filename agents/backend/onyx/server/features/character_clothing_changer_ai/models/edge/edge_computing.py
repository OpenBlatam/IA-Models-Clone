"""
Edge Computing System
=====================
Sistema de edge computing para procesamiento distribuido
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class EdgeNodeStatus(Enum):
    """Estados de nodo edge"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Prioridad de tarea"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EdgeNode:
    """Nodo edge"""
    id: str
    name: str
    location: str
    capabilities: List[str]
    status: EdgeNodeStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    last_heartbeat: float = 0.0
    active_tasks: int = 0


@dataclass
class EdgeTask:
    """Tarea en edge"""
    id: str
    task_type: str
    data: Dict[str, Any]
    priority: TaskPriority
    assigned_node: Optional[str] = None
    status: str = "pending"
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class EdgeComputing:
    """
    Sistema de edge computing
    """
    
    def __init__(self):
        self.nodes: Dict[str, EdgeNode] = {}
        self.tasks: Dict[str, EdgeTask] = {}
        self.task_queue: List[str] = []  # task_ids ordenados por prioridad
    
    def register_node(
        self,
        name: str,
        location: str,
        capabilities: List[str]
    ) -> EdgeNode:
        """
        Registrar nodo edge
        
        Args:
            name: Nombre del nodo
            location: Ubicación geográfica
            capabilities: Capacidades del nodo
        """
        node_id = f"edge_{int(time.time())}"
        
        node = EdgeNode(
            id=node_id,
            name=name,
            location=location,
            capabilities=capabilities,
            status=EdgeNodeStatus.ONLINE,
            last_heartbeat=time.time()
        )
        
        self.nodes[node_id] = node
        return node
    
    def update_node_status(
        self,
        node_id: str,
        status: EdgeNodeStatus,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        network_latency: Optional[float] = None
    ):
        """Actualizar estado de nodo"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.status = status
            node.last_heartbeat = time.time()
            
            if cpu_usage is not None:
                node.cpu_usage = cpu_usage
            if memory_usage is not None:
                node.memory_usage = memory_usage
            if network_latency is not None:
                node.network_latency = network_latency
    
    def submit_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        preferred_location: Optional[str] = None
    ) -> EdgeTask:
        """
        Enviar tarea a edge
        
        Args:
            task_type: Tipo de tarea
            data: Datos de la tarea
            priority: Prioridad
            preferred_location: Ubicación preferida
        """
        task_id = f"task_{int(time.time() * 1000)}"
        
        task = EdgeTask(
            id=task_id,
            task_type=task_type,
            data=data,
            priority=priority,
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # Ordenar por prioridad
        self.task_queue.sort(
            key=lambda tid: self.tasks[tid].priority.value,
            reverse=True
        )
        
        # Asignar a nodo
        node = self._select_node(task, preferred_location)
        if node:
            self._assign_task(task_id, node.id)
        
        return task
    
    def _select_node(
        self,
        task: EdgeTask,
        preferred_location: Optional[str]
    ) -> Optional[EdgeNode]:
        """Seleccionar nodo para tarea"""
        # Filtrar nodos disponibles
        available_nodes = [
            node for node in self.nodes.values()
            if node.status == EdgeNodeStatus.ONLINE
            and node.active_tasks < 10  # Límite de tareas
        ]
        
        if not available_nodes:
            return None
        
        # Si hay ubicación preferida, priorizar nodos cercanos
        if preferred_location:
            location_nodes = [
                node for node in available_nodes
                if node.location == preferred_location
            ]
            if location_nodes:
                available_nodes = location_nodes
        
        # Seleccionar nodo con menor carga
        best_node = min(
            available_nodes,
            key=lambda n: (n.cpu_usage + n.memory_usage) / 2
        )
        
        return best_node
    
    def _assign_task(self, task_id: str, node_id: str):
        """Asignar tarea a nodo"""
        if task_id not in self.tasks or node_id not in self.nodes:
            return
        
        task = self.tasks[task_id]
        node = self.nodes[node_id]
        
        task.assigned_node = node_id
        task.status = "assigned"
        node.active_tasks += 1
        
        # Simular ejecución
        task.started_at = time.time()
        task.status = "running"
        
        # En implementación real, enviar tarea al nodo
        # Por ahora, simular
        time.sleep(0.1)
        
        task.completed_at = time.time()
        task.status = "completed"
        task.result = {"result": "processed", "node": node_id}
        
        node.active_tasks -= 1
    
    def get_node_tasks(self, node_id: str) -> List[EdgeTask]:
        """Obtener tareas de un nodo"""
        return [
            task for task in self.tasks.values()
            if task.assigned_node == node_id
        ]
    
    def get_available_nodes(self) -> List[EdgeNode]:
        """Obtener nodos disponibles"""
        return [
            node for node in self.nodes.values()
            if node.status == EdgeNodeStatus.ONLINE
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de edge computing"""
        return {
            'total_nodes': len(self.nodes),
            'online_nodes': len([n for n in self.nodes.values() if n.status == EdgeNodeStatus.ONLINE]),
            'total_tasks': len(self.tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == 'pending']),
            'running_tasks': len([t for t in self.tasks.values() if t.status == 'running']),
            'completed_tasks': len([t for t in self.tasks.values() if t.status == 'completed']),
            'average_node_load': (
                sum((n.cpu_usage + n.memory_usage) / 2 for n in self.nodes.values()) / len(self.nodes)
                if self.nodes else 0
            )
        }


# Instancia global
edge_computing = EdgeComputing()

