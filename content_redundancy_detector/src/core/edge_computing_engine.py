"""
Edge Computing Engine - Advanced edge computing and distributed processing capabilities
"""

import asyncio
import logging
import time
import json
import hashlib
import socket
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import psutil
import requests
import aiohttp
import numpy as np
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class EdgeConfig:
    """Edge computing configuration"""
    enable_edge_nodes: bool = True
    enable_fog_computing: bool = True
    enable_mobile_edge: bool = True
    enable_iot_edge: bool = True
    enable_5g_edge: bool = True
    enable_edge_ai: bool = True
    enable_edge_ml: bool = True
    enable_edge_analytics: bool = True
    enable_edge_storage: bool = True
    enable_edge_networking: bool = True
    enable_edge_security: bool = True
    enable_edge_orchestration: bool = True
    enable_edge_monitoring: bool = True
    enable_edge_optimization: bool = True
    enable_edge_scaling: bool = True
    max_edge_nodes: int = 100
    max_concurrent_tasks: int = 50
    task_timeout: int = 300
    data_sync_interval: int = 60
    health_check_interval: int = 30
    load_balancing_strategy: str = "round_robin"  # round_robin, least_connections, weighted
    enable_auto_scaling: bool = True
    enable_fault_tolerance: bool = True
    enable_data_replication: bool = True
    enable_edge_caching: bool = True
    enable_edge_compression: bool = True
    enable_edge_encryption: bool = True


@dataclass
class EdgeNode:
    """Edge node data class"""
    node_id: str
    timestamp: datetime
    name: str
    ip_address: str
    port: int
    node_type: str  # edge, fog, mobile, iot, 5g
    capabilities: List[str]
    resources: Dict[str, Any]
    status: str
    location: Dict[str, float]  # lat, lon
    last_heartbeat: datetime
    tasks_assigned: int
    tasks_completed: int
    cpu_usage: float
    memory_usage: float
    network_bandwidth: float
    storage_available: float
    latency_ms: float


@dataclass
class EdgeTask:
    """Edge task data class"""
    task_id: str
    timestamp: datetime
    task_type: str
    priority: int
    data: Dict[str, Any]
    requirements: Dict[str, Any]
    assigned_node: Optional[str]
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    retry_count: int
    max_retries: int


@dataclass
class EdgeData:
    """Edge data data class"""
    data_id: str
    timestamp: datetime
    source_node: str
    data_type: str
    size_bytes: int
    content: Any
    metadata: Dict[str, Any]
    replication_nodes: List[str]
    compression_ratio: float
    encryption_enabled: bool
    ttl_seconds: int
    access_count: int


class EdgeNodeManager:
    """Edge node management system"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.nodes = {}
        self.node_capabilities = defaultdict(list)
        self.node_resources = {}
        self.node_status = {}
        self._initialize_edge_nodes()
    
    def _initialize_edge_nodes(self):
        """Initialize edge nodes"""
        try:
            # Create mock edge nodes for demonstration
            self._create_mock_edge_nodes()
            
            logger.info("Edge node manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing edge nodes: {e}")
    
    def _create_mock_edge_nodes(self):
        """Create mock edge nodes for demonstration"""
        try:
            node_types = ["edge", "fog", "mobile", "iot", "5g"]
            capabilities_map = {
                "edge": ["compute", "storage", "networking", "ai", "ml"],
                "fog": ["compute", "storage", "networking", "analytics", "caching"],
                "mobile": ["compute", "networking", "sensors", "location"],
                "iot": ["sensors", "actuators", "networking", "data_collection"],
                "5g": ["networking", "low_latency", "high_bandwidth", "mobility"]
            }
            
            for i in range(10):  # Create 10 mock nodes
                node_type = node_types[i % len(node_types)]
                node_id = f"edge_node_{i+1}"
                
                node = EdgeNode(
                    node_id=node_id,
                    timestamp=datetime.now(),
                    name=f"Edge Node {i+1}",
                    ip_address=f"192.168.1.{100+i}",
                    port=8000 + i,
                    node_type=node_type,
                    capabilities=capabilities_map[node_type],
                    resources={
                        "cpu_cores": 4 + (i % 4),
                        "memory_gb": 8 + (i % 8),
                        "storage_gb": 100 + (i % 100),
                        "network_mbps": 100 + (i % 100)
                    },
                    status="active",
                    location={"lat": 40.7128 + (i * 0.01), "lon": -74.0060 + (i * 0.01)},
                    last_heartbeat=datetime.now(),
                    tasks_assigned=0,
                    tasks_completed=0,
                    cpu_usage=20.0 + (i * 5),
                    memory_usage=30.0 + (i * 3),
                    network_bandwidth=80.0 + (i * 2),
                    storage_available=90.0 - (i * 2),
                    latency_ms=10.0 + (i * 2)
                )
                
                self.nodes[node_id] = node
                self.node_capabilities[node_type].append(node_id)
                self.node_resources[node_id] = node.resources
                self.node_status[node_id] = "active"
                
        except Exception as e:
            logger.error(f"Error creating mock edge nodes: {e}")
    
    async def register_node(self, node_data: Dict[str, Any]) -> EdgeNode:
        """Register a new edge node"""
        try:
            node_id = hashlib.md5(f"{node_data['ip_address']}_{time.time()}".encode()).hexdigest()
            
            node = EdgeNode(
                node_id=node_id,
                timestamp=datetime.now(),
                name=node_data.get("name", f"Edge Node {node_id[:8]}"),
                ip_address=node_data["ip_address"],
                port=node_data.get("port", 8000),
                node_type=node_data.get("node_type", "edge"),
                capabilities=node_data.get("capabilities", ["compute"]),
                resources=node_data.get("resources", {}),
                status="active",
                location=node_data.get("location", {"lat": 0.0, "lon": 0.0}),
                last_heartbeat=datetime.now(),
                tasks_assigned=0,
                tasks_completed=0,
                cpu_usage=0.0,
                memory_usage=0.0,
                network_bandwidth=0.0,
                storage_available=0.0,
                latency_ms=0.0
            )
            
            self.nodes[node_id] = node
            self.node_capabilities[node.node_type].append(node_id)
            self.node_resources[node_id] = node.resources
            self.node_status[node_id] = "active"
            
            logger.info(f"Edge node {node_id} registered successfully")
            
            return node
            
        except Exception as e:
            logger.error(f"Error registering edge node: {e}")
            raise
    
    async def get_available_nodes(self, requirements: Dict[str, Any]) -> List[EdgeNode]:
        """Get available edge nodes based on requirements"""
        try:
            available_nodes = []
            
            for node_id, node in self.nodes.items():
                if node.status != "active":
                    continue
                
                # Check if node meets requirements
                if self._node_meets_requirements(node, requirements):
                    available_nodes.append(node)
            
            # Sort by load (tasks assigned)
            available_nodes.sort(key=lambda x: x.tasks_assigned)
            
            return available_nodes
            
        except Exception as e:
            logger.error(f"Error getting available nodes: {e}")
            return []
    
    def _node_meets_requirements(self, node: EdgeNode, requirements: Dict[str, Any]) -> bool:
        """Check if node meets requirements"""
        try:
            # Check capabilities
            required_capabilities = requirements.get("capabilities", [])
            if not all(cap in node.capabilities for cap in required_capabilities):
                return False
            
            # Check resources
            required_resources = requirements.get("resources", {})
            for resource, required_value in required_resources.items():
                if resource == "cpu_cores" and node.resources.get("cpu_cores", 0) < required_value:
                    return False
                elif resource == "memory_gb" and node.resources.get("memory_gb", 0) < required_value:
                    return False
                elif resource == "storage_gb" and node.resources.get("storage_gb", 0) < required_value:
                    return False
            
            # Check node type
            required_type = requirements.get("node_type")
            if required_type and node.node_type != required_type:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking node requirements: {e}")
            return False
    
    async def update_node_status(self, node_id: str, status: str):
        """Update node status"""
        try:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                self.nodes[node_id].last_heartbeat = datetime.now()
                self.node_status[node_id] = status
                
        except Exception as e:
            logger.error(f"Error updating node status: {e}")
    
    async def get_node_metrics(self, node_id: str) -> Dict[str, Any]:
        """Get node performance metrics"""
        try:
            if node_id not in self.nodes:
                return {}
            
            node = self.nodes[node_id]
            
            return {
                "node_id": node_id,
                "status": node.status,
                "tasks_assigned": node.tasks_assigned,
                "tasks_completed": node.tasks_completed,
                "cpu_usage": node.cpu_usage,
                "memory_usage": node.memory_usage,
                "network_bandwidth": node.network_bandwidth,
                "storage_available": node.storage_available,
                "latency_ms": node.latency_ms,
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "uptime_seconds": (datetime.now() - node.timestamp).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error getting node metrics: {e}")
            return {}


class EdgeTaskManager:
    """Edge task management system"""
    
    def __init__(self, config: EdgeConfig, node_manager: EdgeNodeManager):
        self.config = config
        self.node_manager = node_manager
        self.tasks = {}
        self.task_queue = deque()
        self.completed_tasks = deque(maxlen=1000)
        self.failed_tasks = deque(maxlen=1000)
        self.task_history = deque(maxlen=10000)
    
    async def submit_task(self, task_data: Dict[str, Any]) -> EdgeTask:
        """Submit a new edge task"""
        try:
            task_id = hashlib.md5(f"{task_data['task_type']}_{time.time()}".encode()).hexdigest()
            
            task = EdgeTask(
                task_id=task_id,
                timestamp=datetime.now(),
                task_type=task_data["task_type"],
                priority=task_data.get("priority", 5),
                data=task_data.get("data", {}),
                requirements=task_data.get("requirements", {}),
                assigned_node=None,
                status="pending",
                start_time=None,
                end_time=None,
                result=None,
                error_message=None,
                retry_count=0,
                max_retries=task_data.get("max_retries", 3)
            )
            
            self.tasks[task_id] = task
            self.task_queue.append(task_id)
            
            # Try to assign task immediately
            await self._assign_task(task_id)
            
            return task
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            raise
    
    async def _assign_task(self, task_id: str):
        """Assign task to available node"""
        try:
            if task_id not in self.tasks:
                return
            
            task = self.tasks[task_id]
            
            # Get available nodes
            available_nodes = await self.node_manager.get_available_nodes(task.requirements)
            
            if not available_nodes:
                logger.warning(f"No available nodes for task {task_id}")
                return
            
            # Select best node based on load balancing strategy
            selected_node = self._select_node(available_nodes)
            
            if selected_node:
                task.assigned_node = selected_node.node_id
                task.status = "assigned"
                selected_node.tasks_assigned += 1
                
                # Execute task
                await self._execute_task(task_id)
            
        except Exception as e:
            logger.error(f"Error assigning task: {e}")
    
    def _select_node(self, available_nodes: List[EdgeNode]) -> Optional[EdgeNode]:
        """Select best node based on load balancing strategy"""
        try:
            if not available_nodes:
                return None
            
            if self.config.load_balancing_strategy == "round_robin":
                # Simple round robin
                return available_nodes[0]
            elif self.config.load_balancing_strategy == "least_connections":
                # Select node with least tasks assigned
                return min(available_nodes, key=lambda x: x.tasks_assigned)
            elif self.config.load_balancing_strategy == "weighted":
                # Weighted selection based on resources
                weights = []
                for node in available_nodes:
                    weight = (
                        node.resources.get("cpu_cores", 1) * 0.3 +
                        node.resources.get("memory_gb", 1) * 0.2 +
                        (100 - node.cpu_usage) * 0.3 +
                        (100 - node.memory_usage) * 0.2
                    )
                    weights.append(weight)
                
                # Select node with highest weight
                max_weight_idx = weights.index(max(weights))
                return available_nodes[max_weight_idx]
            else:
                return available_nodes[0]
                
        except Exception as e:
            logger.error(f"Error selecting node: {e}")
            return available_nodes[0] if available_nodes else None
    
    async def _execute_task(self, task_id: str):
        """Execute assigned task"""
        try:
            if task_id not in self.tasks:
                return
            
            task = self.tasks[task_id]
            
            if not task.assigned_node:
                return
            
            task.status = "running"
            task.start_time = datetime.now()
            
            # Simulate task execution
            await asyncio.sleep(1)  # Simulate processing time
            
            # Mock task execution
            if task.task_type == "data_processing":
                result = await self._execute_data_processing_task(task)
            elif task.task_type == "ai_inference":
                result = await self._execute_ai_inference_task(task)
            elif task.task_type == "analytics":
                result = await self._execute_analytics_task(task)
            else:
                result = {"status": "completed", "message": "Task executed successfully"}
            
            task.result = result
            task.status = "completed"
            task.end_time = datetime.now()
            
            # Update node metrics
            if task.assigned_node in self.node_manager.nodes:
                node = self.node_manager.nodes[task.assigned_node]
                node.tasks_assigned -= 1
                node.tasks_completed += 1
            
            # Move to completed tasks
            self.completed_tasks.append(task_id)
            self.task_history.append({
                "task_id": task_id,
                "timestamp": task.timestamp.isoformat(),
                "task_type": task.task_type,
                "assigned_node": task.assigned_node,
                "execution_time": (task.end_time - task.start_time).total_seconds(),
                "status": "completed"
            })
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = "failed"
                task.error_message = str(e)
                task.retry_count += 1
                
                if task.retry_count < task.max_retries:
                    # Retry task
                    task.status = "pending"
                    task.assigned_node = None
                    self.task_queue.append(task_id)
                else:
                    # Move to failed tasks
                    self.failed_tasks.append(task_id)
                    self.task_history.append({
                        "task_id": task_id,
                        "timestamp": task.timestamp.isoformat(),
                        "task_type": task.task_type,
                        "assigned_node": task.assigned_node,
                        "execution_time": 0,
                        "status": "failed",
                        "error": str(e)
                    })
    
    async def _execute_data_processing_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute data processing task"""
        try:
            data = task.data.get("data", [])
            
            # Simulate data processing
            processed_data = []
            for item in data:
                processed_item = {
                    "original": item,
                    "processed": item * 2 if isinstance(item, (int, float)) else str(item).upper(),
                    "timestamp": datetime.now().isoformat()
                }
                processed_data.append(processed_item)
            
            return {
                "status": "completed",
                "processed_count": len(processed_data),
                "processed_data": processed_data[:10],  # Return first 10 items
                "processing_time_ms": 100
            }
            
        except Exception as e:
            logger.error(f"Error executing data processing task: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _execute_ai_inference_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute AI inference task"""
        try:
            input_data = task.data.get("input_data", {})
            
            # Simulate AI inference
            predictions = []
            for i in range(5):  # Mock 5 predictions
                prediction = {
                    "class": f"class_{i}",
                    "confidence": 0.8 + (i * 0.05),
                    "timestamp": datetime.now().isoformat()
                }
                predictions.append(prediction)
            
            return {
                "status": "completed",
                "predictions": predictions,
                "inference_time_ms": 50,
                "model_used": "mock_model_v1"
            }
            
        except Exception as e:
            logger.error(f"Error executing AI inference task: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _execute_analytics_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute analytics task"""
        try:
            data = task.data.get("data", [])
            
            # Simulate analytics
            if data:
                analytics_result = {
                    "count": len(data),
                    "mean": statistics.mean(data) if all(isinstance(x, (int, float)) for x in data) else 0,
                    "median": statistics.median(data) if all(isinstance(x, (int, float)) for x in data) else 0,
                    "std_dev": statistics.stdev(data) if len(data) > 1 and all(isinstance(x, (int, float)) for x in data) else 0,
                    "min": min(data) if data else 0,
                    "max": max(data) if data else 0
                }
            else:
                analytics_result = {
                    "count": 0,
                    "mean": 0,
                    "median": 0,
                    "std_dev": 0,
                    "min": 0,
                    "max": 0
                }
            
            return {
                "status": "completed",
                "analytics": analytics_result,
                "processing_time_ms": 75
            }
            
        except Exception as e:
            logger.error(f"Error executing analytics task: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        try:
            if task_id not in self.tasks:
                return {"status": "not_found"}
            
            task = self.tasks[task_id]
            
            return {
                "task_id": task_id,
                "status": task.status,
                "task_type": task.task_type,
                "priority": task.priority,
                "assigned_node": task.assigned_node,
                "start_time": task.start_time.isoformat() if task.start_time else None,
                "end_time": task.end_time.isoformat() if task.end_time else None,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "error_message": task.error_message,
                "result": task.result
            }
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {"status": "error", "error": str(e)}


class EdgeDataManager:
    """Edge data management system"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.data_store = {}
        self.data_replication = defaultdict(list)
        self.data_access_log = deque(maxlen=10000)
        self.cache = {}
        self._initialize_data_manager()
    
    def _initialize_data_manager(self):
        """Initialize data manager"""
        try:
            logger.info("Edge data manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing data manager: {e}")
    
    async def store_data(self, data: Any, data_type: str, source_node: str, 
                        metadata: Dict[str, Any] = None) -> EdgeData:
        """Store data in edge network"""
        try:
            data_id = hashlib.md5(f"{data_type}_{source_node}_{time.time()}".encode()).hexdigest()
            
            # Calculate data size
            if isinstance(data, (str, bytes)):
                size_bytes = len(data)
            elif isinstance(data, (list, dict)):
                size_bytes = len(json.dumps(data).encode())
            else:
                size_bytes = 100  # Default size
            
            edge_data = EdgeData(
                data_id=data_id,
                timestamp=datetime.now(),
                source_node=source_node,
                data_type=data_type,
                size_bytes=size_bytes,
                content=data,
                metadata=metadata or {},
                replication_nodes=[],
                compression_ratio=1.0,
                encryption_enabled=self.config.enable_edge_encryption,
                ttl_seconds=3600,  # 1 hour default TTL
                access_count=0
            )
            
            self.data_store[data_id] = edge_data
            
            # Log data access
            self.data_access_log.append({
                "data_id": data_id,
                "action": "store",
                "timestamp": datetime.now().isoformat(),
                "source_node": source_node
            })
            
            return edge_data
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise
    
    async def retrieve_data(self, data_id: str, requesting_node: str) -> Optional[EdgeData]:
        """Retrieve data from edge network"""
        try:
            if data_id not in self.data_store:
                return None
            
            edge_data = self.data_store[data_id]
            
            # Check TTL
            if edge_data.ttl_seconds > 0:
                age_seconds = (datetime.now() - edge_data.timestamp).total_seconds()
                if age_seconds > edge_data.ttl_seconds:
                    # Data expired
                    del self.data_store[data_id]
                    return None
            
            # Update access count
            edge_data.access_count += 1
            
            # Log data access
            self.data_access_log.append({
                "data_id": data_id,
                "action": "retrieve",
                "timestamp": datetime.now().isoformat(),
                "requesting_node": requesting_node
            })
            
            return edge_data
            
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return None
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """Get data storage statistics"""
        try:
            total_data = len(self.data_store)
            total_size = sum(data.size_bytes for data in self.data_store.values())
            
            data_types = defaultdict(int)
            for data in self.data_store.values():
                data_types[data.data_type] += 1
            
            return {
                "total_data_items": total_data,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "data_types": dict(data_types),
                "cache_hit_rate": 0.85,  # Mock cache hit rate
                "compression_ratio": 0.7,  # Mock compression ratio
                "encryption_enabled": self.config.enable_edge_encryption
            }
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}


class EdgeComputingEngine:
    """Main Edge Computing Engine"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.node_manager = EdgeNodeManager(config)
        self.task_manager = EdgeTaskManager(config, self.node_manager)
        self.data_manager = EdgeDataManager(config)
        
        self.performance_metrics = {}
        self.health_status = {}
        
        self._initialize_edge_engine()
    
    def _initialize_edge_engine(self):
        """Initialize edge computing engine"""
        try:
            logger.info("Edge Computing Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing edge engine: {e}")
    
    async def submit_edge_task(self, task_data: Dict[str, Any]) -> EdgeTask:
        """Submit task to edge network"""
        try:
            return await self.task_manager.submit_task(task_data)
            
        except Exception as e:
            logger.error(f"Error submitting edge task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        try:
            return await self.task_manager.get_task_status(task_id)
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_available_nodes(self, requirements: Dict[str, Any]) -> List[EdgeNode]:
        """Get available edge nodes"""
        try:
            return await self.node_manager.get_available_nodes(requirements)
            
        except Exception as e:
            logger.error(f"Error getting available nodes: {e}")
            return []
    
    async def store_edge_data(self, data: Any, data_type: str, source_node: str, 
                            metadata: Dict[str, Any] = None) -> EdgeData:
        """Store data in edge network"""
        try:
            return await self.data_manager.store_data(data, data_type, source_node, metadata)
            
        except Exception as e:
            logger.error(f"Error storing edge data: {e}")
            raise
    
    async def retrieve_edge_data(self, data_id: str, requesting_node: str) -> Optional[EdgeData]:
        """Retrieve data from edge network"""
        try:
            return await self.data_manager.retrieve_data(data_id, requesting_node)
            
        except Exception as e:
            logger.error(f"Error retrieving edge data: {e}")
            return None
    
    async def get_edge_capabilities(self) -> Dict[str, Any]:
        """Get edge computing capabilities"""
        try:
            capabilities = {
                "supported_node_types": ["edge", "fog", "mobile", "iot", "5g"],
                "supported_task_types": ["data_processing", "ai_inference", "analytics", "caching", "networking"],
                "supported_data_types": ["sensor_data", "video_stream", "audio_stream", "text_data", "binary_data"],
                "load_balancing_strategies": ["round_robin", "least_connections", "weighted"],
                "max_nodes": self.config.max_edge_nodes,
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "task_timeout": self.config.task_timeout,
                "features": {
                    "edge_ai": self.config.enable_edge_ai,
                    "edge_ml": self.config.enable_edge_ml,
                    "edge_analytics": self.config.enable_edge_analytics,
                    "edge_storage": self.config.enable_edge_storage,
                    "edge_networking": self.config.enable_edge_networking,
                    "edge_security": self.config.enable_edge_security,
                    "edge_orchestration": self.config.enable_edge_orchestration,
                    "edge_monitoring": self.config.enable_edge_monitoring,
                    "edge_optimization": self.config.enable_edge_optimization,
                    "edge_scaling": self.config.enable_edge_scaling,
                    "auto_scaling": self.config.enable_auto_scaling,
                    "fault_tolerance": self.config.enable_fault_tolerance,
                    "data_replication": self.config.enable_data_replication,
                    "edge_caching": self.config.enable_edge_caching,
                    "edge_compression": self.config.enable_edge_compression,
                    "edge_encryption": self.config.enable_edge_encryption
                }
            }
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Error getting edge capabilities: {e}")
            return {}
    
    async def get_edge_performance_metrics(self) -> Dict[str, Any]:
        """Get edge computing performance metrics"""
        try:
            metrics = {
                "total_nodes": len(self.node_manager.nodes),
                "active_nodes": len([n for n in self.node_manager.nodes.values() if n.status == "active"]),
                "total_tasks": len(self.task_manager.tasks),
                "pending_tasks": len(self.task_manager.task_queue),
                "completed_tasks": len(self.task_manager.completed_tasks),
                "failed_tasks": len(self.task_manager.failed_tasks),
                "total_data_items": len(self.data_manager.data_store),
                "average_task_execution_time": 0.0,
                "task_success_rate": 0.0,
                "node_utilization": {},
                "data_access_patterns": {}
            }
            
            # Calculate task metrics
            if self.task_manager.task_history:
                execution_times = [h["execution_time"] for h in self.task_manager.task_history if h["execution_time"] > 0]
                if execution_times:
                    metrics["average_task_execution_time"] = statistics.mean(execution_times)
                
                completed_count = len([h for h in self.task_manager.task_history if h["status"] == "completed"])
                total_count = len(self.task_manager.task_history)
                if total_count > 0:
                    metrics["task_success_rate"] = completed_count / total_count
            
            # Calculate node utilization
            for node_id, node in self.node_manager.nodes.items():
                metrics["node_utilization"][node_id] = {
                    "cpu_usage": node.cpu_usage,
                    "memory_usage": node.memory_usage,
                    "tasks_assigned": node.tasks_assigned,
                    "tasks_completed": node.tasks_completed
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting edge performance metrics: {e}")
            return {}


# Global instance
edge_computing_engine: Optional[EdgeComputingEngine] = None


async def initialize_edge_computing_engine(config: Optional[EdgeConfig] = None) -> None:
    """Initialize edge computing engine"""
    global edge_computing_engine
    
    if config is None:
        config = EdgeConfig()
    
    edge_computing_engine = EdgeComputingEngine(config)
    logger.info("Edge Computing Engine initialized successfully")


async def get_edge_computing_engine() -> Optional[EdgeComputingEngine]:
    """Get edge computing engine instance"""
    return edge_computing_engine

















