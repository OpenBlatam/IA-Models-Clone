"""
Gamma App - Real Improvement Edge
Edge computing system for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import socket
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import psutil
import requests
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)

class EdgeNodeType(Enum):
    """Edge node types"""
    GATEWAY = "gateway"
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    HYBRID = "hybrid"

class EdgeProtocol(Enum):
    """Edge protocols"""
    HTTP = "http"
    MQTT = "mqtt"
    COAP = "coap"
    WEBSOCKET = "websocket"
    TCP = "tcp"
    UDP = "udp"
    GRPC = "grpc"

@dataclass
class EdgeNode:
    """Edge computing node"""
    node_id: str
    name: str
    type: EdgeNodeType
    protocol: EdgeProtocol
    ip_address: str
    port: int
    status: str
    capabilities: List[str]
    resources: Dict[str, Any]
    location: Dict[str, float]
    created_at: datetime = None
    last_heartbeat: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()

@dataclass
class EdgeTask:
    """Edge computing task"""
    task_id: str
    node_id: str
    task_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = None
    status: str = "pending"
    priority: int = 1
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RealImprovementEdge:
    """
    Edge computing system for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize edge computing system"""
        self.project_root = Path(project_root)
        self.nodes: Dict[str, EdgeNode] = {}
        self.tasks: Dict[str, EdgeTask] = {}
        self.edge_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.task_queue = queue.PriorityQueue()
        self.worker_threads = []
        self.heartbeat_interval = 30
        self.load_balancer = None
        
        # Initialize with default nodes
        self._initialize_default_nodes()
        
        # Start edge services
        self._start_edge_services()
        
        logger.info(f"Real Improvement Edge initialized for {self.project_root}")
    
    def _initialize_default_nodes(self):
        """Initialize default edge nodes"""
        # Gateway node
        gateway_node = EdgeNode(
            node_id="gateway_001",
            name="Edge Gateway 001",
            type=EdgeNodeType.GATEWAY,
            protocol=EdgeProtocol.HTTP,
            ip_address="192.168.1.10",
            port=8080,
            status="online",
            capabilities=["routing", "load_balancing", "security"],
            resources={
                "cpu_cores": 4,
                "memory_gb": 8,
                "storage_gb": 100,
                "network_bandwidth_mbps": 1000
            },
            location={"lat": 40.7128, "lon": -74.0060}
        )
        self.nodes[gateway_node.node_id] = gateway_node
        
        # Compute node
        compute_node = EdgeNode(
            node_id="compute_001",
            name="Edge Compute 001",
            type=EdgeNodeType.COMPUTE,
            protocol=EdgeProtocol.HTTP,
            ip_address="192.168.1.11",
            port=8081,
            status="online",
            capabilities=["processing", "ai_inference", "data_analysis"],
            resources={
                "cpu_cores": 8,
                "memory_gb": 16,
                "storage_gb": 200,
                "gpu_cores": 2
            },
            location={"lat": 40.7128, "lon": -74.0060}
        )
        self.nodes[compute_node.node_id] = compute_node
        
        # Storage node
        storage_node = EdgeNode(
            node_id="storage_001",
            name="Edge Storage 001",
            type=EdgeNodeType.STORAGE,
            protocol=EdgeProtocol.HTTP,
            ip_address="192.168.1.12",
            port=8082,
            status="online",
            capabilities=["data_storage", "backup", "replication"],
            resources={
                "cpu_cores": 2,
                "memory_gb": 4,
                "storage_gb": 1000,
                "network_bandwidth_mbps": 500
            },
            location={"lat": 40.7128, "lon": -74.0060}
        )
        self.nodes[storage_node.node_id] = storage_node
        
        # Sensor node
        sensor_node = EdgeNode(
            node_id="sensor_001",
            name="Edge Sensor 001",
            type=EdgeNodeType.SENSOR,
            protocol=EdgeProtocol.MQTT,
            ip_address="192.168.1.13",
            port=1883,
            status="online",
            capabilities=["data_collection", "preprocessing", "filtering"],
            resources={
                "cpu_cores": 1,
                "memory_gb": 1,
                "storage_gb": 10,
                "battery_level": 85.0
            },
            location={"lat": 40.7128, "lon": -74.0060}
        )
        self.nodes[sensor_node.node_id] = sensor_node
    
    def _start_edge_services(self):
        """Start edge computing services"""
        try:
            # Start task processor
            task_processor = threading.Thread(target=self._process_tasks, daemon=True)
            task_processor.start()
            self.worker_threads.append(task_processor)
            
            # Start heartbeat monitor
            heartbeat_monitor = threading.Thread(target=self._monitor_heartbeats, daemon=True)
            heartbeat_monitor.start()
            self.worker_threads.append(heartbeat_monitor)
            
            # Start load balancer
            self._start_load_balancer()
            
            self._log_edge("services_started", "Edge services started")
            
        except Exception as e:
            logger.error(f"Failed to start edge services: {e}")
    
    def _process_tasks(self):
        """Process edge computing tasks"""
        while True:
            try:
                if not self.task_queue.empty():
                    priority, task = self.task_queue.get()
                    
                    # Find best node for task
                    best_node = self._find_best_node_for_task(task)
                    
                    if best_node:
                        # Execute task on node
                        self._execute_task_on_node(task, best_node)
                    else:
                        # No available nodes
                        task.status = "failed"
                        self._log_edge("task_failed", f"No available nodes for task {task.task_id}")
                    
                    self.task_queue.task_done()
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to process tasks: {e}")
                time.sleep(1)
    
    def _monitor_heartbeats(self):
        """Monitor node heartbeats"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for node_id, node in self.nodes.items():
                    # Check if node is responsive
                    if self._check_node_health(node):
                        node.last_heartbeat = current_time
                        if node.status != "online":
                            node.status = "online"
                            self._log_edge("node_online", f"Node {node.name} came online")
                    else:
                        if node.status == "online":
                            node.status = "offline"
                            self._log_edge("node_offline", f"Node {node.name} went offline")
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Failed to monitor heartbeats: {e}")
                time.sleep(60)
    
    def _check_node_health(self, node: EdgeNode) -> bool:
        """Check node health"""
        try:
            if node.protocol == EdgeProtocol.HTTP:
                # Check HTTP health endpoint
                response = requests.get(f"http://{node.ip_address}:{node.port}/health", timeout=5)
                return response.status_code == 200
            elif node.protocol == EdgeProtocol.MQTT:
                # Check MQTT connectivity
                return True  # Simplified check
            else:
                # Check TCP connectivity
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((node.ip_address, node.port))
                sock.close()
                return result == 0
                
        except Exception:
            return False
    
    def _find_best_node_for_task(self, task: EdgeTask) -> Optional[EdgeNode]:
        """Find best node for task execution"""
        try:
            available_nodes = [
                node for node in self.nodes.values()
                if node.status == "online" and self._can_handle_task(node, task)
            ]
            
            if not available_nodes:
                return None
            
            # Simple load balancing - find node with least load
            best_node = min(available_nodes, key=lambda n: self._calculate_node_load(n))
            
            return best_node
            
        except Exception as e:
            logger.error(f"Failed to find best node: {e}")
            return None
    
    def _can_handle_task(self, node: EdgeNode, task: EdgeTask) -> bool:
        """Check if node can handle task"""
        try:
            # Check if node has required capabilities
            required_capabilities = self._get_required_capabilities(task)
            if not all(cap in node.capabilities for cap in required_capabilities):
                return False
            
            # Check resource availability
            if not self._check_resource_availability(node, task):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check task compatibility: {e}")
            return False
    
    def _get_required_capabilities(self, task: EdgeTask) -> List[str]:
        """Get required capabilities for task"""
        capability_map = {
            "data_processing": ["processing"],
            "ai_inference": ["ai_inference", "processing"],
            "data_storage": ["data_storage"],
            "data_collection": ["data_collection"],
            "routing": ["routing"],
            "security": ["security"]
        }
        
        return capability_map.get(task.task_type, ["processing"])
    
    def _check_resource_availability(self, node: EdgeNode, task: EdgeTask) -> bool:
        """Check if node has sufficient resources"""
        try:
            # Get current system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Check CPU availability
            if cpu_percent > 90:
                return False
            
            # Check memory availability
            if memory_percent > 90:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check resource availability: {e}")
            return False
    
    def _calculate_node_load(self, node: EdgeNode) -> float:
        """Calculate node load"""
        try:
            # Count active tasks on node
            active_tasks = len([
                task for task in self.tasks.values()
                if task.node_id == node.node_id and task.status in ["pending", "running"]
            ])
            
            # Get system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Calculate load score
            load_score = (active_tasks * 0.4) + (cpu_percent * 0.3) + (memory_percent * 0.3)
            
            return load_score
            
        except Exception as e:
            logger.error(f"Failed to calculate node load: {e}")
            return 100.0  # High load if calculation fails
    
    def _execute_task_on_node(self, task: EdgeTask, node: EdgeNode):
        """Execute task on edge node"""
        try:
            task.node_id = node.node_id
            task.status = "running"
            task.started_at = datetime.utcnow()
            
            self._log_edge("task_started", f"Task {task.task_id} started on node {node.name}")
            
            # Simulate task execution
            result = self._simulate_task_execution(task, node)
            
            task.output_data = result
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.execution_time = (task.completed_at - task.started_at).total_seconds()
            
            self._log_edge("task_completed", f"Task {task.task_id} completed in {task.execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            task.status = "failed"
            task.completed_at = datetime.utcnow()
    
    def _simulate_task_execution(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Simulate task execution"""
        try:
            # Simulate different task types
            if task.task_type == "data_processing":
                return {
                    "processed_records": 1000,
                    "processing_time": 2.5,
                    "result": "Data processed successfully"
                }
            elif task.task_type == "ai_inference":
                return {
                    "inference_time": 1.2,
                    "confidence": 0.95,
                    "result": "AI inference completed"
                }
            elif task.task_type == "data_storage":
                return {
                    "stored_bytes": 1024000,
                    "storage_time": 0.8,
                    "result": "Data stored successfully"
                }
            elif task.task_type == "data_collection":
                return {
                    "collected_samples": 500,
                    "collection_time": 1.5,
                    "result": "Data collected successfully"
                }
            else:
                return {
                    "execution_time": 1.0,
                    "result": "Task executed successfully"
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _start_load_balancer(self):
        """Start load balancer"""
        try:
            # Simple round-robin load balancer
            self.load_balancer = {
                "algorithm": "round_robin",
                "current_node_index": 0,
                "active": True
            }
            
            self._log_edge("load_balancer_started", "Load balancer started")
            
        except Exception as e:
            logger.error(f"Failed to start load balancer: {e}")
    
    def add_edge_node(self, name: str, type: EdgeNodeType, protocol: EdgeProtocol,
                     ip_address: str, port: int, capabilities: List[str],
                     resources: Dict[str, Any], location: Dict[str, float]) -> str:
        """Add edge node"""
        try:
            node_id = f"node_{int(time.time() * 1000)}"
            
            node = EdgeNode(
                node_id=node_id,
                name=name,
                type=type,
                protocol=protocol,
                ip_address=ip_address,
                port=port,
                status="offline",
                capabilities=capabilities,
                resources=resources,
                location=location
            )
            
            self.nodes[node_id] = node
            
            self._log_edge("node_added", f"Added edge node {name} with ID {node_id}")
            
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to add edge node: {e}")
            raise
    
    def remove_edge_node(self, node_id: str) -> bool:
        """Remove edge node"""
        try:
            if node_id in self.nodes:
                node_name = self.nodes[node_id].name
                del self.nodes[node_id]
                
                # Cancel tasks on this node
                for task in self.tasks.values():
                    if task.node_id == node_id and task.status in ["pending", "running"]:
                        task.status = "cancelled"
                
                self._log_edge("node_removed", f"Removed edge node {node_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove edge node: {e}")
            return False
    
    def submit_task(self, task_type: str, input_data: Dict[str, Any], priority: int = 1) -> str:
        """Submit task to edge computing system"""
        try:
            task_id = f"task_{int(time.time() * 1000)}"
            
            task = EdgeTask(
                task_id=task_id,
                node_id="",
                task_type=task_type,
                input_data=input_data,
                priority=priority
            )
            
            self.tasks[task_id] = task
            
            # Add to task queue
            self.task_queue.put((priority, task))
            
            self._log_edge("task_submitted", f"Task {task_id} submitted")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        try:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            
            return {
                "task_id": task_id,
                "node_id": task.node_id,
                "task_type": task.task_type,
                "status": task.status,
                "priority": task.priority,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "execution_time": task.execution_time,
                "input_data": task.input_data,
                "output_data": task.output_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return None
    
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node information"""
        try:
            if node_id not in self.nodes:
                return None
            
            node = self.nodes[node_id]
            
            # Calculate node load
            load = self._calculate_node_load(node)
            
            # Count tasks on node
            task_count = len([
                task for task in self.tasks.values()
                if task.node_id == node_id
            ])
            
            return {
                "node_id": node_id,
                "name": node.name,
                "type": node.type.value,
                "protocol": node.protocol.value,
                "ip_address": node.ip_address,
                "port": node.port,
                "status": node.status,
                "capabilities": node.capabilities,
                "resources": node.resources,
                "location": node.location,
                "created_at": node.created_at.isoformat(),
                "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None,
                "load": load,
                "task_count": task_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get node info: {e}")
            return None
    
    def get_edge_summary(self) -> Dict[str, Any]:
        """Get edge computing summary"""
        total_nodes = len(self.nodes)
        online_nodes = len([n for n in self.nodes.values() if n.status == "online"])
        offline_nodes = total_nodes - online_nodes
        
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        running_tasks = len([t for t in self.tasks.values() if t.status == "running"])
        pending_tasks = len([t for t in self.tasks.values() if t.status == "pending"])
        
        # Count by type
        type_counts = {}
        for node in self.nodes.values():
            node_type = node.type.value
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        # Calculate average execution time
        completed_task_times = [t.execution_time for t in self.tasks.values() if t.status == "completed"]
        avg_execution_time = sum(completed_task_times) / len(completed_task_times) if completed_task_times else 0
        
        return {
            "total_nodes": total_nodes,
            "online_nodes": online_nodes,
            "offline_nodes": offline_nodes,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "running_tasks": running_tasks,
            "pending_tasks": pending_tasks,
            "type_distribution": type_counts,
            "avg_execution_time": avg_execution_time,
            "queue_size": self.task_queue.qsize(),
            "load_balancer_active": self.load_balancer["active"] if self.load_balancer else False
        }
    
    def _log_edge(self, event: str, message: str):
        """Log edge event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if "edge_logs" not in self.edge_logs:
            self.edge_logs["edge_logs"] = []
        
        self.edge_logs["edge_logs"].append(log_entry)
        
        logger.info(f"Edge: {event} - {message}")
    
    def get_edge_logs(self) -> List[Dict[str, Any]]:
        """Get edge logs"""
        return self.edge_logs.get("edge_logs", [])
    
    def shutdown(self):
        """Shutdown edge computing system"""
        try:
            # Stop load balancer
            if self.load_balancer:
                self.load_balancer["active"] = False
            
            # Wait for worker threads to finish
            for thread in self.worker_threads:
                thread.join(timeout=5)
            
            self._log_edge("shutdown", "Edge computing system shutdown completed")
            
        except Exception as e:
            logger.error(f"Failed to shutdown edge system: {e}")

# Global edge instance
improvement_edge = None

def get_improvement_edge() -> RealImprovementEdge:
    """Get improvement edge instance"""
    global improvement_edge
    if not improvement_edge:
        improvement_edge = RealImprovementEdge()
    return improvement_edge













