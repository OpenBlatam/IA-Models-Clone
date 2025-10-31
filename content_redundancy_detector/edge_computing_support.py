"""
Edge Computing Support for Distributed Processing
Sistema de Edge Computing para procesamiento distribuido ultra-optimizado
"""

import asyncio
import logging
import time
import json
import socket
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import psutil
import platform

logger = logging.getLogger(__name__)


class EdgeNodeType(Enum):
    """Tipos de nodos edge"""
    MOBILE_DEVICE = "mobile_device"
    IOT_SENSOR = "iot_sensor"
    EDGE_SERVER = "edge_server"
    GATEWAY = "gateway"
    FOG_NODE = "fog_node"
    MICRO_DATACENTER = "micro_datacenter"


class EdgeNodeStatus(Enum):
    """Estados de nodos edge"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class TaskType(Enum):
    """Tipos de tareas"""
    CONTENT_ANALYSIS = "content_analysis"
    SIMILARITY_DETECTION = "similarity_detection"
    QUALITY_ASSESSMENT = "quality_assessment"
    DATA_PROCESSING = "data_processing"
    ML_INFERENCE = "ml_inference"
    REAL_TIME_ANALYSIS = "real_time_analysis"


class TaskPriority(Enum):
    """Prioridades de tareas"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EdgeNodeInfo:
    """Información de nodo edge"""
    id: str
    name: str
    type: EdgeNodeType
    status: EdgeNodeStatus
    ip_address: str
    port: int
    capabilities: List[str]
    resources: Dict[str, Any]
    location: Dict[str, float]  # lat, lon
    last_heartbeat: float
    created_at: float
    metadata: Dict[str, Any]


@dataclass
class EdgeTask:
    """Tarea edge"""
    id: str
    type: TaskType
    priority: TaskPriority
    data: Dict[str, Any]
    node_id: Optional[str]
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    status: str
    result: Optional[Dict[str, Any]]
    error: Optional[str]


@dataclass
class EdgeCluster:
    """Cluster edge"""
    id: str
    name: str
    nodes: List[str]
    leader_node: Optional[str]
    created_at: float
    metadata: Dict[str, Any]


@dataclass
class EdgeResource:
    """Recurso edge"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_bandwidth: float
    battery_level: Optional[float]
    temperature: Optional[float]
    timestamp: float


class EdgeNodeManager:
    """Manager de nodos edge"""
    
    def __init__(self):
        self.nodes: Dict[str, EdgeNodeInfo] = {}
        self.clusters: Dict[str, EdgeCluster] = {}
        self.tasks: Dict[str, EdgeTask] = {}
        self.resources: Dict[str, List[EdgeResource]] = defaultdict(list)
        self._lock = threading.Lock()
        self._heartbeat_interval = 30.0
        self._cleanup_interval = 300.0  # 5 minutos
    
    async def register_node(self, node_info: EdgeNodeInfo) -> bool:
        """Registrar nodo edge"""
        async with self._lock:
            try:
                # Verificar conectividad
                if not await self._check_node_connectivity(node_info):
                    return False
                
                self.nodes[node_info.id] = node_info
                logger.info(f"Edge node registered: {node_info.id} ({node_info.name})")
                return True
                
            except Exception as e:
                logger.error(f"Error registering edge node: {e}")
                return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """Desregistrar nodo edge"""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                
                # Remover de clusters
                for cluster_id, cluster in self.clusters.items():
                    if node_id in cluster.nodes:
                        cluster.nodes.remove(node_id)
                        if cluster.leader_node == node_id:
                            cluster.leader_node = None
                
                logger.info(f"Edge node unregistered: {node_id}")
                return True
            return False
    
    async def update_node_status(self, node_id: str, status: EdgeNodeStatus, 
                               resources: Optional[EdgeResource] = None) -> bool:
        """Actualizar estado de nodo"""
        async with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                self.nodes[node_id].last_heartbeat = time.time()
                
                if resources:
                    self.resources[node_id].append(resources)
                    # Mantener solo los últimos 100 recursos
                    if len(self.resources[node_id]) > 100:
                        self.resources[node_id] = self.resources[node_id][-100:]
                
                return True
            return False
    
    async def get_available_nodes(self, task_type: TaskType, 
                                min_resources: Optional[Dict[str, float]] = None) -> List[EdgeNodeInfo]:
        """Obtener nodos disponibles para tarea"""
        async with self._lock:
            available_nodes = []
            
            for node in self.nodes.values():
                if (node.status == EdgeNodeStatus.ONLINE and 
                    task_type.value in node.capabilities):
                    
                    # Verificar recursos mínimos
                    if min_resources:
                        latest_resources = self._get_latest_resources(node.id)
                        if latest_resources:
                            if (latest_resources.cpu_usage > min_resources.get("cpu_usage", 100) or
                                latest_resources.memory_usage > min_resources.get("memory_usage", 100)):
                                continue
                    
                    available_nodes.append(node)
            
            # Ordenar por recursos disponibles (menor uso = mejor)
            available_nodes.sort(key=lambda n: self._get_node_load_score(n.id))
            return available_nodes
    
    def _get_latest_resources(self, node_id: str) -> Optional[EdgeResource]:
        """Obtener recursos más recientes del nodo"""
        if node_id in self.resources and self.resources[node_id]:
            return self.resources[node_id][-1]
        return None
    
    def _get_node_load_score(self, node_id: str) -> float:
        """Obtener score de carga del nodo"""
        latest_resources = self._get_latest_resources(node_id)
        if not latest_resources:
            return 0.0
        
        # Score basado en uso de recursos (menor es mejor)
        return (latest_resources.cpu_usage + 
                latest_resources.memory_usage + 
                latest_resources.disk_usage) / 3.0
    
    async def _check_node_connectivity(self, node_info: EdgeNodeInfo) -> bool:
        """Verificar conectividad del nodo"""
        try:
            # Intentar conectar al nodo
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{node_info.ip_address}:{node_info.port}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def cleanup_inactive_nodes(self):
        """Limpiar nodos inactivos"""
        async with self._lock:
            current_time = time.time()
            inactive_nodes = []
            
            for node_id, node in self.nodes.items():
                if current_time - node.last_heartbeat > self._cleanup_interval:
                    inactive_nodes.append(node_id)
            
            for node_id in inactive_nodes:
                await self.unregister_node(node_id)
                logger.info(f"Cleaned up inactive edge node: {node_id}")


class EdgeTaskManager:
    """Manager de tareas edge"""
    
    def __init__(self, node_manager: EdgeNodeManager):
        self.node_manager = node_manager
        self.tasks: Dict[str, EdgeTask] = {}
        self.task_queue: deque = deque()
        self._lock = threading.Lock()
        self._max_retries = 3
        self._task_timeout = 300.0  # 5 minutos
    
    async def submit_task(self, task_type: TaskType, data: Dict[str, Any], 
                         priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Enviar tarea"""
        task_id = f"task_{int(time.time())}_{id(data)}"
        
        task = EdgeTask(
            id=task_id,
            type=task_type,
            priority=priority,
            data=data,
            node_id=None,
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            status="pending",
            result=None,
            error=None
        )
        
        async with self._lock:
            self.tasks[task_id] = task
            self.task_queue.append(task_id)
        
        # Intentar asignar inmediatamente
        await self._process_task_queue()
        
        return task_id
    
    async def _process_task_queue(self):
        """Procesar cola de tareas"""
        async with self._lock:
            if not self.task_queue:
                return
            
            # Ordenar por prioridad
            sorted_tasks = sorted(
                self.task_queue,
                key=lambda tid: self._get_priority_score(self.tasks[tid].priority),
                reverse=True
            )
            
            for task_id in sorted_tasks:
                task = self.tasks[task_id]
                if task.status == "pending":
                    await self._assign_task(task)
                    break
    
    def _get_priority_score(self, priority: TaskPriority) -> int:
        """Obtener score de prioridad"""
        priority_scores = {
            TaskPriority.LOW: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.HIGH: 3,
            TaskPriority.CRITICAL: 4
        }
        return priority_scores.get(priority, 2)
    
    async def _assign_task(self, task: EdgeTask):
        """Asignar tarea a nodo"""
        try:
            # Obtener nodos disponibles
            available_nodes = await self.node_manager.get_available_nodes(task.type)
            
            if not available_nodes:
                logger.warning(f"No available nodes for task {task.id}")
                return
            
            # Seleccionar mejor nodo
            selected_node = available_nodes[0]
            task.node_id = selected_node.id
            task.status = "assigned"
            
            # Enviar tarea al nodo
            success = await self._send_task_to_node(task, selected_node)
            
            if success:
                task.status = "running"
                task.started_at = time.time()
                self.task_queue.remove(task.id)
                logger.info(f"Task {task.id} assigned to node {selected_node.id}")
            else:
                task.status = "failed"
                task.error = "Failed to send task to node"
                self.task_queue.remove(task.id)
                
        except Exception as e:
            logger.error(f"Error assigning task {task.id}: {e}")
            task.status = "failed"
            task.error = str(e)
            if task.id in self.task_queue:
                self.task_queue.remove(task.id)
    
    async def _send_task_to_node(self, task: EdgeTask, node: EdgeNodeInfo) -> bool:
        """Enviar tarea a nodo"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{node.ip_address}:{node.port}/tasks",
                    json={
                        "task_id": task.id,
                        "type": task.type.value,
                        "data": task.data
                    }
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending task to node {node.id}: {e}")
            return False
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener resultado de tarea"""
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                return {
                    "id": task.id,
                    "type": task.type.value,
                    "status": task.status,
                    "result": task.result,
                    "error": task.error,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "node_id": task.node_id
                }
            return None
    
    async def get_task_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de tareas"""
        async with self._lock:
            total_tasks = len(self.tasks)
            pending_tasks = sum(1 for t in self.tasks.values() if t.status == "pending")
            running_tasks = sum(1 for t in self.tasks.values() if t.status == "running")
            completed_tasks = sum(1 for t in self.tasks.values() if t.status == "completed")
            failed_tasks = sum(1 for t in self.tasks.values() if t.status == "failed")
            
            return {
                "total_tasks": total_tasks,
                "pending_tasks": pending_tasks,
                "running_tasks": running_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "queue_size": len(self.task_queue),
                "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            }


class EdgeClusterManager:
    """Manager de clusters edge"""
    
    def __init__(self, node_manager: EdgeNodeManager):
        self.node_manager = node_manager
        self.clusters: Dict[str, EdgeCluster] = {}
        self._lock = threading.Lock()
    
    async def create_cluster(self, cluster_id: str, name: str, 
                           node_ids: List[str]) -> bool:
        """Crear cluster"""
        async with self._lock:
            try:
                # Verificar que todos los nodos existen
                for node_id in node_ids:
                    if node_id not in self.node_manager.nodes:
                        return False
                
                # Seleccionar líder (nodo con mejor recursos)
                leader_node = self._select_leader_node(node_ids)
                
                cluster = EdgeCluster(
                    id=cluster_id,
                    name=name,
                    nodes=node_ids,
                    leader_node=leader_node,
                    created_at=time.time(),
                    metadata={}
                )
                
                self.clusters[cluster_id] = cluster
                logger.info(f"Edge cluster created: {cluster_id} with {len(node_ids)} nodes")
                return True
                
            except Exception as e:
                logger.error(f"Error creating edge cluster: {e}")
                return False
    
    def _select_leader_node(self, node_ids: List[str]) -> Optional[str]:
        """Seleccionar nodo líder"""
        best_node = None
        best_score = float('inf')
        
        for node_id in node_ids:
            score = self.node_manager._get_node_load_score(node_id)
            if score < best_score:
                best_score = score
                best_node = node_id
        
        return best_node
    
    async def get_cluster_info(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de cluster"""
        async with self._lock:
            if cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                return {
                    "id": cluster.id,
                    "name": cluster.name,
                    "nodes": cluster.nodes,
                    "leader_node": cluster.leader_node,
                    "created_at": cluster.created_at,
                    "metadata": cluster.metadata
                }
            return None
    
    async def get_all_clusters(self) -> List[Dict[str, Any]]:
        """Obtener todos los clusters"""
        async with self._lock:
            return [
                {
                    "id": cluster.id,
                    "name": cluster.name,
                    "nodes": cluster.nodes,
                    "leader_node": cluster.leader_node,
                    "created_at": cluster.created_at
                }
                for cluster in self.clusters.values()
            ]


class EdgeComputingManager:
    """Manager principal de edge computing"""
    
    def __init__(self):
        self.node_manager = EdgeNodeManager()
        self.task_manager = EdgeTaskManager(self.node_manager)
        self.cluster_manager = EdgeClusterManager(self.node_manager)
        self.is_running = False
        self._cleanup_task = None
        self._heartbeat_task = None
    
    async def start(self):
        """Iniciar edge computing manager"""
        try:
            self.is_running = True
            
            # Iniciar tareas de limpieza
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info("Edge computing manager started")
            
        except Exception as e:
            logger.error(f"Error starting edge computing manager: {e}")
            raise
    
    async def stop(self):
        """Detener edge computing manager"""
        try:
            self.is_running = False
            
            # Detener tareas
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Edge computing manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping edge computing manager: {e}")
    
    async def _cleanup_loop(self):
        """Loop de limpieza"""
        while self.is_running:
            try:
                await self.node_manager.cleanup_inactive_nodes()
                await asyncio.sleep(60)  # Limpiar cada minuto
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _heartbeat_loop(self):
        """Loop de heartbeat"""
        while self.is_running:
            try:
                # Procesar cola de tareas
                await self.task_manager._process_task_queue()
                await asyncio.sleep(5)  # Procesar cada 5 segundos
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        node_stats = {
            "total_nodes": len(self.node_manager.nodes),
            "online_nodes": sum(1 for n in self.node_manager.nodes.values() 
                              if n.status == EdgeNodeStatus.ONLINE),
            "offline_nodes": sum(1 for n in self.node_manager.nodes.values() 
                               if n.status == EdgeNodeStatus.OFFLINE),
            "busy_nodes": sum(1 for n in self.node_manager.nodes.values() 
                            if n.status == EdgeNodeStatus.BUSY)
        }
        
        task_stats = await self.task_manager.get_task_stats()
        cluster_stats = {
            "total_clusters": len(self.cluster_manager.clusters),
            "clusters": await self.cluster_manager.get_all_clusters()
        }
        
        return {
            "is_running": self.is_running,
            "nodes": node_stats,
            "tasks": task_stats,
            "clusters": cluster_stats
        }


# Instancia global del manager de edge computing
edge_computing_manager = EdgeComputingManager()


# Router para endpoints de edge computing
edge_computing_router = APIRouter()


@edge_computing_router.post("/edge/nodes/register")
async def register_edge_node_endpoint(node_data: dict):
    """Registrar nodo edge"""
    try:
        node_info = EdgeNodeInfo(
            id=node_data["id"],
            name=node_data["name"],
            type=EdgeNodeType(node_data["type"]),
            status=EdgeNodeStatus.ONLINE,
            ip_address=node_data["ip_address"],
            port=node_data["port"],
            capabilities=node_data.get("capabilities", []),
            resources=node_data.get("resources", {}),
            location=node_data.get("location", {"lat": 0.0, "lon": 0.0}),
            last_heartbeat=time.time(),
            created_at=time.time(),
            metadata=node_data.get("metadata", {})
        )
        
        success = await edge_computing_manager.node_manager.register_node(node_info)
        
        if success:
            return {
                "message": "Edge node registered successfully",
                "node_id": node_info.id,
                "name": node_info.name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to register edge node")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid node type: {e}")
    except Exception as e:
        logger.error(f"Error registering edge node: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register edge node: {str(e)}")


@edge_computing_router.delete("/edge/nodes/{node_id}")
async def unregister_edge_node_endpoint(node_id: str):
    """Desregistrar nodo edge"""
    try:
        success = await edge_computing_manager.node_manager.unregister_node(node_id)
        
        if success:
            return {"message": "Edge node unregistered successfully", "node_id": node_id}
        else:
            raise HTTPException(status_code=404, detail="Edge node not found")
            
    except Exception as e:
        logger.error(f"Error unregistering edge node: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unregister edge node: {str(e)}")


@edge_computing_router.get("/edge/nodes")
async def get_edge_nodes_endpoint():
    """Obtener nodos edge"""
    try:
        nodes = edge_computing_manager.node_manager.nodes
        return {
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type.value,
                    "status": node.status.value,
                    "ip_address": node.ip_address,
                    "port": node.port,
                    "capabilities": node.capabilities,
                    "location": node.location,
                    "last_heartbeat": node.last_heartbeat,
                    "created_at": node.created_at
                }
                for node in nodes.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting edge nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge nodes: {str(e)}")


@edge_computing_router.get("/edge/nodes/{node_id}")
async def get_edge_node_endpoint(node_id: str):
    """Obtener nodo edge específico"""
    try:
        if node_id not in edge_computing_manager.node_manager.nodes:
            raise HTTPException(status_code=404, detail="Edge node not found")
        
        node = edge_computing_manager.node_manager.nodes[node_id]
        latest_resources = edge_computing_manager.node_manager._get_latest_resources(node_id)
        
        return {
            "id": node.id,
            "name": node.name,
            "type": node.type.value,
            "status": node.status.value,
            "ip_address": node.ip_address,
            "port": node.port,
            "capabilities": node.capabilities,
            "resources": node.resources,
            "location": node.location,
            "last_heartbeat": node.last_heartbeat,
            "created_at": node.created_at,
            "metadata": node.metadata,
            "latest_resources": {
                "cpu_usage": latest_resources.cpu_usage if latest_resources else None,
                "memory_usage": latest_resources.memory_usage if latest_resources else None,
                "disk_usage": latest_resources.disk_usage if latest_resources else None,
                "network_bandwidth": latest_resources.network_bandwidth if latest_resources else None,
                "battery_level": latest_resources.battery_level if latest_resources else None,
                "temperature": latest_resources.temperature if latest_resources else None,
                "timestamp": latest_resources.timestamp if latest_resources else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting edge node: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge node: {str(e)}")


@edge_computing_router.post("/edge/tasks/submit")
async def submit_edge_task_endpoint(task_data: dict):
    """Enviar tarea edge"""
    try:
        task_type = TaskType(task_data["type"])
        data = task_data["data"]
        priority = TaskPriority(task_data.get("priority", "medium"))
        
        task_id = await edge_computing_manager.task_manager.submit_task(
            task_type, data, priority
        )
        
        return {
            "message": "Edge task submitted successfully",
            "task_id": task_id,
            "type": task_type.value,
            "priority": priority.value
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid task type or priority: {e}")
    except Exception as e:
        logger.error(f"Error submitting edge task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit edge task: {str(e)}")


@edge_computing_router.get("/edge/tasks/{task_id}")
async def get_edge_task_endpoint(task_id: str):
    """Obtener tarea edge"""
    try:
        result = await edge_computing_manager.task_manager.get_task_result(task_id)
        
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="Edge task not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting edge task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge task: {str(e)}")


@edge_computing_router.get("/edge/tasks")
async def get_edge_tasks_endpoint():
    """Obtener tareas edge"""
    try:
        tasks = edge_computing_manager.task_manager.tasks
        return {
            "tasks": [
                {
                    "id": task.id,
                    "type": task.type.value,
                    "priority": task.priority.value,
                    "status": task.status,
                    "node_id": task.node_id,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "error": task.error
                }
                for task in tasks.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting edge tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge tasks: {str(e)}")


@edge_computing_router.post("/edge/clusters")
async def create_edge_cluster_endpoint(cluster_data: dict):
    """Crear cluster edge"""
    try:
        cluster_id = cluster_data["id"]
        name = cluster_data["name"]
        node_ids = cluster_data["node_ids"]
        
        success = await edge_computing_manager.cluster_manager.create_cluster(
            cluster_id, name, node_ids
        )
        
        if success:
            return {
                "message": "Edge cluster created successfully",
                "cluster_id": cluster_id,
                "name": name,
                "nodes": node_ids
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create edge cluster")
            
    except Exception as e:
        logger.error(f"Error creating edge cluster: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create edge cluster: {str(e)}")


@edge_computing_router.get("/edge/clusters")
async def get_edge_clusters_endpoint():
    """Obtener clusters edge"""
    try:
        clusters = await edge_computing_manager.cluster_manager.get_all_clusters()
        return {"clusters": clusters}
    except Exception as e:
        logger.error(f"Error getting edge clusters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge clusters: {str(e)}")


@edge_computing_router.get("/edge/clusters/{cluster_id}")
async def get_edge_cluster_endpoint(cluster_id: str):
    """Obtener cluster edge específico"""
    try:
        cluster_info = await edge_computing_manager.cluster_manager.get_cluster_info(cluster_id)
        
        if cluster_info:
            return cluster_info
        else:
            raise HTTPException(status_code=404, detail="Edge cluster not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting edge cluster: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge cluster: {str(e)}")


@edge_computing_router.get("/edge/stats")
async def get_edge_computing_stats_endpoint():
    """Obtener estadísticas de edge computing"""
    try:
        stats = await edge_computing_manager.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting edge computing stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge computing stats: {str(e)}")


# Funciones de utilidad para integración
async def start_edge_computing():
    """Iniciar edge computing"""
    await edge_computing_manager.start()


async def stop_edge_computing():
    """Detener edge computing"""
    await edge_computing_manager.stop()


async def register_edge_node(node_info: EdgeNodeInfo) -> bool:
    """Registrar nodo edge"""
    return await edge_computing_manager.node_manager.register_node(node_info)


async def submit_edge_task(task_type: TaskType, data: Dict[str, Any], 
                          priority: TaskPriority = TaskPriority.MEDIUM) -> str:
    """Enviar tarea edge"""
    return await edge_computing_manager.task_manager.submit_task(task_type, data, priority)


async def get_edge_computing_stats() -> Dict[str, Any]:
    """Obtener estadísticas de edge computing"""
    return await edge_computing_manager.get_system_stats()


logger.info("Edge computing support module loaded successfully")

