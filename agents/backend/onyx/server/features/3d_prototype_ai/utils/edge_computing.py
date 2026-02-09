"""
Edge Computing - Sistema de edge computing
===========================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EdgeNodeStatus(str, Enum):
    """Estados de nodo edge"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class EdgeComputing:
    """Sistema de edge computing"""
    
    def __init__(self):
        self.edge_nodes: Dict[str, Dict[str, Any]] = {}
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self.node_metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_edge_node(self, node_id: str, name: str, location: str,
                          capabilities: List[str], max_workload: int = 100) -> Dict[str, Any]:
        """Registra un nodo edge"""
        node = {
            "id": node_id,
            "name": name,
            "location": location,
            "capabilities": capabilities,
            "max_workload": max_workload,
            "current_workload": 0,
            "status": EdgeNodeStatus.ONLINE.value,
            "registered_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat()
        }
        
        self.edge_nodes[node_id] = node
        
        logger.info(f"Nodo edge registrado: {node_id} - {location}")
        return node
    
    def deploy_to_edge(self, deployment_id: str, node_id: str,
                      workload: Dict[str, Any]) -> Dict[str, Any]:
        """Despliega workload a un nodo edge"""
        node = self.edge_nodes.get(node_id)
        if not node:
            raise ValueError(f"Nodo edge no encontrado: {node_id}")
        
        if node["status"] != EdgeNodeStatus.ONLINE.value:
            raise ValueError(f"Nodo edge no está online: {node_id}")
        
        current_workload = node["current_workload"]
        if current_workload + workload.get("size", 1) > node["max_workload"]:
            raise ValueError(f"Nodo edge sobrecargado: {node_id}")
        
        deployment = {
            "id": deployment_id,
            "node_id": node_id,
            "workload": workload,
            "status": "deployed",
            "deployed_at": datetime.now().isoformat()
        }
        
        self.deployments[deployment_id] = deployment
        node["current_workload"] += workload.get("size", 1)
        
        logger.info(f"Workload desplegado en nodo edge: {deployment_id} -> {node_id}")
        return deployment
    
    def record_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Registra métricas de un nodo"""
        node = self.edge_nodes.get(node_id)
        if not node:
            return
        
        metric_entry = {
            "node_id": node_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = []
        
        self.node_metrics[node_id].append(metric_entry)
        
        # Mantener solo últimos 1000 métricas
        if len(self.node_metrics[node_id]) > 1000:
            self.node_metrics[node_id] = self.node_metrics[node_id][-1000:]
        
        # Actualizar estado basado en métricas
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)
        
        if cpu_usage > 90 or memory_usage > 90:
            node["status"] = EdgeNodeStatus.DEGRADED.value
        else:
            node["status"] = EdgeNodeStatus.ONLINE.value
        
        node["last_heartbeat"] = datetime.now().isoformat()
    
    def find_best_node(self, required_capabilities: List[str],
                      workload_size: int = 1) -> Optional[str]:
        """Encuentra el mejor nodo para un workload"""
        suitable_nodes = [
            node for node in self.edge_nodes.values()
            if (node["status"] == EdgeNodeStatus.ONLINE.value and
                all(cap in node["capabilities"] for cap in required_capabilities) and
                (node["current_workload"] + workload_size) <= node["max_workload"])
        ]
        
        if not suitable_nodes:
            return None
        
        # Seleccionar nodo con menor carga
        best_node = min(suitable_nodes, key=lambda n: n["current_workload"])
        
        return best_node["id"]
    
    def get_edge_network_status(self) -> Dict[str, Any]:
        """Obtiene estado de la red edge"""
        total_nodes = len(self.edge_nodes)
        online_nodes = sum(1 for n in self.edge_nodes.values() if n["status"] == EdgeNodeStatus.ONLINE.value)
        
        total_workload = sum(n["current_workload"] for n in self.edge_nodes.values())
        total_capacity = sum(n["max_workload"] for n in self.edge_nodes.values())
        
        return {
            "total_nodes": total_nodes,
            "online_nodes": online_nodes,
            "offline_nodes": total_nodes - online_nodes,
            "total_workload": total_workload,
            "total_capacity": total_capacity,
            "utilization_percent": (total_workload / total_capacity * 100) if total_capacity > 0 else 0,
            "active_deployments": len([d for d in self.deployments.values() if d["status"] == "deployed"])
        }




