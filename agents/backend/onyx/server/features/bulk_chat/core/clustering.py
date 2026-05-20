"""
Clustering - Sistema de Clustering Distribuido
=============================================

Sistema de clustering para escalabilidad horizontal.
"""

import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Nodo en el cluster."""
    node_id: str
    host: str
    port: int
    status: str = "active"  # active, inactive, failed
    last_heartbeat: datetime = field(default_factory=datetime.now)
    load: float = 0.0  # Carga del nodo (0.0 - 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClusterManager:
    """Gestor de cluster para distribución de carga."""
    
    def __init__(self, node_id: str, host: str = "localhost", port: int = 8006):
        """
        Inicializar gestor de cluster.
        
        Args:
            node_id: ID único del nodo
            host: Host del nodo
            port: Puerto del nodo
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.nodes: Dict[str, Node] = {}
        self.local_node = Node(
            node_id=node_id,
            host=host,
            port=port,
        )
        self.nodes[node_id] = self.local_node
        
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def register_node(self, node: Node):
        """Registrar un nodo en el cluster."""
        async with self._lock:
            self.nodes[node.node_id] = node
            logger.info(f"Registered node: {node.node_id} ({node.host}:{node.port})")
    
    async def unregister_node(self, node_id: str):
        """Desregistrar un nodo."""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered node: {node_id}")
    
    async def get_node_for_session(self, session_id: str) -> Node:
        """
        Obtener nodo para una sesión (consistent hashing).
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            Nodo asignado
        """
        async with self._lock:
            active_nodes = [n for n in self.nodes.values() if n.status == "active"]
            
            if not active_nodes:
                return self.local_node
            
            # Consistent hashing simple
            hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
            node_index = hash_value % len(active_nodes)
            
            return active_nodes[node_index]
    
    async def get_least_loaded_node(self) -> Node:
        """Obtener nodo con menor carga."""
        async with self._lock:
            active_nodes = [n for n in self.nodes.values() if n.status == "active"]
            
            if not active_nodes:
                return self.local_node
            
            # Ordenar por carga
            sorted_nodes = sorted(active_nodes, key=lambda n: n.load)
            return sorted_nodes[0]
    
    async def update_node_load(self, node_id: str, load: float):
        """Actualizar carga de un nodo."""
        async with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].load = load
    
    async def heartbeat(self, node_id: str):
        """Actualizar heartbeat de un nodo."""
        async with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = datetime.now()
                self.nodes[node_id].status = "active"
    
    async def check_health(self, timeout_seconds: float = 30.0):
        """Verificar salud de nodos."""
        async with self._lock:
            now = datetime.now()
            for node in self.nodes.values():
                if node.node_id == self.node_id:
                    continue  # No verificar el nodo local
                
                time_since_heartbeat = (now - node.last_heartbeat).total_seconds()
                if time_since_heartbeat > timeout_seconds:
                    node.status = "failed"
                    logger.warning(f"Node {node.node_id} marked as failed (no heartbeat)")
    
    async def start_heartbeat_monitor(self, interval: float = 10.0):
        """Iniciar monitor de heartbeat."""
        async def monitor_loop():
            while True:
                try:
                    await self.check_health()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in heartbeat monitor: {e}")
                    await asyncio.sleep(interval)
        
        self._heartbeat_task = asyncio.create_task(monitor_loop())
        logger.info("Heartbeat monitor started")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Obtener información del cluster."""
        active_nodes = [n for n in self.nodes.values() if n.status == "active"]
        failed_nodes = [n for n in self.nodes.values() if n.status == "failed"]
        
        return {
            "node_id": self.node_id,
            "total_nodes": len(self.nodes),
            "active_nodes": len(active_nodes),
            "failed_nodes": len(failed_nodes),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "host": n.host,
                    "port": n.port,
                    "status": n.status,
                    "load": n.load,
                    "last_heartbeat": n.last_heartbeat.isoformat(),
                }
                for n in self.nodes.values()
            ],
        }
































