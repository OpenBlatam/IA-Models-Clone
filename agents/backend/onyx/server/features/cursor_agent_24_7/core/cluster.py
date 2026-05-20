"""
Cluster - Sistema de clustering
=================================

Sistema para ejecutar múltiples instancias del agente en cluster.
"""

import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Estado de un nodo"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNREACHABLE = "unreachable"


@dataclass
class ClusterNode:
    """Nodo del cluster"""
    id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.INACTIVE
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClusterManager:
    """Gestor de cluster"""
    
    def __init__(self, agent, node_id: Optional[str] = None):
        self.agent = agent
        self.node_id = node_id or self._generate_node_id()
        self.nodes: Dict[str, ClusterNode] = {}
        self.leader_id: Optional[str] = None
        self.running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
    
    def _generate_node_id(self) -> str:
        """Generar ID único para el nodo"""
        import socket
        hostname = socket.gethostname()
        timestamp = datetime.now().timestamp()
        unique_str = f"{hostname}_{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    async def start(self):
        """Iniciar cluster manager"""
        if self.running:
            return
        
        self.running = True
        
        # Agregar este nodo al cluster
        self.add_node(self.node_id, "localhost", 8024)
        
        # Iniciar heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info(f"🌐 Cluster manager started (node_id: {self.node_id})")
    
    async def stop(self):
        """Detener cluster manager"""
        self.running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🌐 Cluster manager stopped")
    
    def add_node(self, node_id: str, host: str, port: int, metadata: Optional[Dict] = None):
        """Agregar nodo al cluster"""
        node = ClusterNode(
            id=node_id,
            host=host,
            port=port,
            status=NodeStatus.ACTIVE,
            last_heartbeat=datetime.now(),
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        logger.info(f"🌐 Node added to cluster: {node_id} ({host}:{port})")
    
    def remove_node(self, node_id: str):
        """Remover nodo del cluster"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            if self.leader_id == node_id:
                self.leader_id = None
            logger.info(f"🌐 Node removed from cluster: {node_id}")
    
    async def _heartbeat_loop(self):
        """Loop de heartbeat"""
        while self.running:
            try:
                # Actualizar heartbeat de este nodo
                if self.node_id in self.nodes:
                    self.nodes[self.node_id].last_heartbeat = datetime.now()
                    self.nodes[self.node_id].status = NodeStatus.ACTIVE
                
                # Verificar nodos inactivos
                now = datetime.now()
                for node_id, node in list(self.nodes.items()):
                    if node_id != self.node_id:
                        # Si no hay heartbeat en 30 segundos, marcar como inactivo
                        if node.last_heartbeat:
                            time_since_heartbeat = (now - node.last_heartbeat).total_seconds()
                            if time_since_heartbeat > 30:
                                node.status = NodeStatus.UNREACHABLE
                
                # Elegir líder (el nodo con ID más bajo)
                active_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
                if active_nodes:
                    leader = min(active_nodes, key=lambda n: n.id)
                    self.leader_id = leader.id
                
                await asyncio.sleep(5)  # Heartbeat cada 5 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    def is_leader(self) -> bool:
        """Verificar si este nodo es el líder"""
        return self.leader_id == self.node_id
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Obtener información del cluster"""
        return {
            "node_id": self.node_id,
            "is_leader": self.is_leader(),
            "leader_id": self.leader_id,
            "total_nodes": len(self.nodes),
            "active_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.ACTIVE),
            "nodes": [
                {
                    "id": node.id,
                    "host": node.host,
                    "port": node.port,
                    "status": node.status.value,
                    "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None
                }
                for node in self.nodes.values()
            ]
        }



