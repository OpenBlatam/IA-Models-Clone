"""
BUL Edge Computing System
========================

Edge computing implementation for distributed document processing and real-time optimization.
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import aiohttp
import websockets
import redis.asyncio as redis
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import uuid

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class EdgeNodeType(str, Enum):
    """Types of edge nodes"""
    MOBILE_DEVICE = "mobile_device"
    IOT_SENSOR = "iot_sensor"
    EDGE_SERVER = "edge_server"
    FOG_NODE = "fog_node"
    MICRO_DATACENTER = "micro_datacenter"
    SMART_GATEWAY = "smart_gateway"
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"
    DRONE = "drone"

class ProcessingCapability(str, Enum):
    """Processing capabilities of edge nodes"""
    DOCUMENT_PROCESSING = "document_processing"
    REAL_TIME_ANALYTICS = "real_time_analytics"
    MACHINE_LEARNING = "machine_learning"
    IMAGE_PROCESSING = "image_processing"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    DATA_COMPRESSION = "data_compression"
    ENCRYPTION = "encryption"
    STREAMING = "streaming"

class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class EdgeNode:
    """Edge node information"""
    id: str
    name: str
    node_type: EdgeNodeType
    location: Dict[str, float]  # lat, lon, altitude
    capabilities: List[ProcessingCapability]
    resources: Dict[str, Any]  # CPU, memory, storage, network
    status: str  # online, offline, busy, maintenance
    last_heartbeat: datetime
    processing_power: float  # 0.0 to 1.0
    network_latency: float  # milliseconds
    battery_level: Optional[float] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class EdgeTask:
    """Edge computing task"""
    id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    requirements: Dict[str, Any]
    assigned_node: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class EdgeNetwork:
    """Edge computing network topology"""
    nodes: Dict[str, EdgeNode]
    connections: List[Tuple[str, str, float]]  # node1, node2, latency
    clusters: Dict[str, List[str]]  # cluster_id -> node_ids
    routing_table: Dict[str, Dict[str, str]]  # source -> destination -> next_hop

class EdgeComputingManager:
    """Edge computing management system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        
        # Edge network
        self.network = EdgeNetwork(
            nodes={},
            connections=[],
            clusters={},
            routing_table={}
        )
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.running_tasks: Dict[str, EdgeTask] = {}
        self.completed_tasks: Dict[str, EdgeTask] = {}
        
        # Communication
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.redis_client: Optional[redis.Redis] = None
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Initialize edge computing
        self._initialize_edge_services()
    
    def _initialize_edge_services(self):
        """Initialize edge computing services"""
        try:
            # Initialize Redis for distributed coordination
            self.redis_client = redis.from_url("redis://localhost:6379/1")
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._task_scheduler())
            asyncio.create_task(self._network_optimizer())
            
            self.logger.info("Edge computing services initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize edge services: {e}")
    
    async def register_edge_node(
        self,
        node_id: str,
        name: str,
        node_type: EdgeNodeType,
        location: Dict[str, float],
        capabilities: List[ProcessingCapability],
        resources: Dict[str, Any]
    ) -> EdgeNode:
        """Register a new edge node"""
        try:
            node = EdgeNode(
                id=node_id,
                name=name,
                node_type=node_type,
                location=location,
                capabilities=capabilities,
                resources=resources,
                status="online",
                last_heartbeat=datetime.now(),
                processing_power=self._calculate_processing_power(resources),
                network_latency=0.0,
                metadata={}
            )
            
            # Add to network
            self.network.nodes[node_id] = node
            
            # Update routing table
            await self._update_routing_table()
            
            # Notify other nodes
            await self._broadcast_node_registration(node)
            
            self.logger.info(f"Edge node registered: {node_id} ({node_type.value})")
            return node
        
        except Exception as e:
            self.logger.error(f"Error registering edge node: {e}")
            raise
    
    def _calculate_processing_power(self, resources: Dict[str, Any]) -> float:
        """Calculate processing power score for a node"""
        try:
            cpu_cores = resources.get('cpu_cores', 1)
            cpu_frequency = resources.get('cpu_frequency', 1.0)  # GHz
            memory_gb = resources.get('memory_gb', 1)
            storage_gb = resources.get('storage_gb', 1)
            
            # Normalize and combine metrics
            cpu_score = (cpu_cores * cpu_frequency) / 8.0  # Normalize to 8-core 1GHz
            memory_score = memory_gb / 16.0  # Normalize to 16GB
            storage_score = min(storage_gb / 100.0, 1.0)  # Normalize to 100GB
            
            processing_power = (cpu_score * 0.5 + memory_score * 0.3 + storage_score * 0.2)
            return min(1.0, processing_power)
        
        except Exception as e:
            self.logger.error(f"Error calculating processing power: {e}")
            return 0.5
    
    async def submit_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        requirements: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> EdgeTask:
        """Submit a task for edge processing"""
        try:
            task = EdgeTask(
                id=str(uuid.uuid4()),
                task_type=task_type,
                priority=priority,
                data=data,
                requirements=requirements,
                created_at=datetime.now()
            )
            
            # Add to task queue
            await self.task_queue.put(task)
            
            self.logger.info(f"Task submitted: {task.id} ({task_type})")
            return task
        
        except Exception as e:
            self.logger.error(f"Error submitting task: {e}")
            raise
    
    async def _task_scheduler(self):
        """Background task scheduler"""
        while True:
            try:
                # Get next task from queue
                task = await self.task_queue.get()
                
                # Find suitable node for task
                suitable_node = await self._find_suitable_node(task)
                
                if suitable_node:
                    # Assign task to node
                    task.assigned_node = suitable_node.id
                    task.status = "running"
                    task.started_at = datetime.now()
                    
                    self.running_tasks[task.id] = task
                    
                    # Execute task on node
                    asyncio.create_task(self._execute_task_on_node(task, suitable_node))
                else:
                    # No suitable node found, retry later
                    await asyncio.sleep(1)
                    await self.task_queue.put(task)
                
                # Mark task as done
                self.task_queue.task_done()
            
            except Exception as e:
                self.logger.error(f"Error in task scheduler: {e}")
                await asyncio.sleep(1)
    
    async def _find_suitable_node(self, task: EdgeTask) -> Optional[EdgeNode]:
        """Find suitable edge node for task execution"""
        try:
            suitable_nodes = []
            
            for node in self.network.nodes.values():
                if node.status != "online":
                    continue
                
                # Check if node has required capabilities
                if not self._node_has_capabilities(node, task.requirements):
                    continue
                
                # Check if node has sufficient resources
                if not self._node_has_resources(node, task.requirements):
                    continue
                
                # Calculate suitability score
                score = self._calculate_node_suitability(node, task)
                suitable_nodes.append((node, score))
            
            if not suitable_nodes:
                return None
            
            # Sort by suitability score (higher is better)
            suitable_nodes.sort(key=lambda x: x[1], reverse=True)
            return suitable_nodes[0][0]
        
        except Exception as e:
            self.logger.error(f"Error finding suitable node: {e}")
            return None
    
    def _node_has_capabilities(self, node: EdgeNode, requirements: Dict[str, Any]) -> bool:
        """Check if node has required capabilities"""
        try:
            required_capabilities = requirements.get('capabilities', [])
            
            for capability in required_capabilities:
                if capability not in node.capabilities:
                    return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error checking node capabilities: {e}")
            return False
    
    def _node_has_resources(self, node: EdgeNode, requirements: Dict[str, Any]) -> bool:
        """Check if node has sufficient resources"""
        try:
            required_cpu = requirements.get('cpu_cores', 1)
            required_memory = requirements.get('memory_gb', 1)
            required_storage = requirements.get('storage_gb', 1)
            
            available_cpu = node.resources.get('cpu_cores', 1)
            available_memory = node.resources.get('memory_gb', 1)
            available_storage = node.resources.get('storage_gb', 1)
            
            return (available_cpu >= required_cpu and
                    available_memory >= required_memory and
                    available_storage >= required_storage)
        
        except Exception as e:
            self.logger.error(f"Error checking node resources: {e}")
            return False
    
    def _calculate_node_suitability(self, node: EdgeNode, task: EdgeTask) -> float:
        """Calculate node suitability score for task"""
        try:
            score = 0.0
            
            # Processing power (40% weight)
            score += node.processing_power * 0.4
            
            # Network latency (20% weight)
            latency_score = max(0, 1.0 - (node.network_latency / 100.0))  # Normalize to 100ms
            score += latency_score * 0.2
            
            # Resource availability (20% weight)
            resource_score = min(1.0, sum(node.resources.values()) / 100.0)
            score += resource_score * 0.2
            
            # Priority matching (20% weight)
            priority_score = 1.0 if task.priority == TaskPriority.CRITICAL else 0.8
            score += priority_score * 0.2
            
            return score
        
        except Exception as e:
            self.logger.error(f"Error calculating node suitability: {e}")
            return 0.0
    
    async def _execute_task_on_node(self, task: EdgeTask, node: EdgeNode):
        """Execute task on edge node"""
        try:
            # Simulate task execution
            execution_time = self._estimate_execution_time(task, node)
            
            # Execute task based on type
            if task.task_type == "document_processing":
                result = await self._execute_document_processing(task, node)
            elif task.task_type == "real_time_analytics":
                result = await self._execute_real_time_analytics(task, node)
            elif task.task_type == "machine_learning":
                result = await self._execute_machine_learning(task, node)
            else:
                result = await self._execute_generic_task(task, node)
            
            # Update task status
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            # Move to completed tasks
            self.completed_tasks[task.id] = task
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            
            self.logger.info(f"Task completed: {task.id} on node {node.id}")
        
        except Exception as e:
            # Handle task failure
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            self.completed_tasks[task.id] = task
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            
            self.logger.error(f"Task failed: {task.id} on node {node.id}: {e}")
    
    def _estimate_execution_time(self, task: EdgeTask, node: EdgeNode) -> float:
        """Estimate task execution time"""
        try:
            # Base execution time based on task type
            base_times = {
                "document_processing": 5.0,
                "real_time_analytics": 2.0,
                "machine_learning": 10.0,
                "image_processing": 3.0,
                "nlp": 4.0
            }
            
            base_time = base_times.get(task.task_type, 5.0)
            
            # Adjust based on node processing power
            adjusted_time = base_time / max(0.1, node.processing_power)
            
            # Adjust based on data size
            data_size = len(str(task.data))
            size_factor = 1.0 + (data_size / 10000.0)  # 10KB baseline
            
            return adjusted_time * size_factor
        
        except Exception as e:
            self.logger.error(f"Error estimating execution time: {e}")
            return 5.0
    
    async def _execute_document_processing(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Execute document processing task"""
        try:
            document_content = task.data.get('content', '')
            
            # Simulate document processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            result = {
                'processed_content': document_content.upper(),
                'word_count': len(document_content.split()),
                'processing_node': node.id,
                'processing_time': 0.1
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing document processing: {e}")
            raise
    
    async def _execute_real_time_analytics(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Execute real-time analytics task"""
        try:
            data_points = task.data.get('data_points', [])
            
            # Simulate analytics processing
            await asyncio.sleep(0.05)
            
            result = {
                'average': np.mean(data_points) if data_points else 0,
                'max': max(data_points) if data_points else 0,
                'min': min(data_points) if data_points else 0,
                'count': len(data_points),
                'processing_node': node.id
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing real-time analytics: {e}")
            raise
    
    async def _execute_machine_learning(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Execute machine learning task"""
        try:
            training_data = task.data.get('training_data', [])
            model_type = task.data.get('model_type', 'linear')
            
            # Simulate ML training
            await asyncio.sleep(0.2)
            
            result = {
                'model_type': model_type,
                'training_samples': len(training_data),
                'accuracy': np.random.uniform(0.8, 0.95),
                'processing_node': node.id,
                'model_size': np.random.uniform(1.0, 10.0)  # MB
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing machine learning: {e}")
            raise
    
    async def _execute_generic_task(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Execute generic task"""
        try:
            # Simulate generic processing
            await asyncio.sleep(0.1)
            
            result = {
                'task_type': task.task_type,
                'processing_node': node.id,
                'input_data_size': len(str(task.data)),
                'processing_time': 0.1
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing generic task: {e}")
            raise
    
    async def _heartbeat_monitor(self):
        """Monitor edge node heartbeats"""
        while True:
            try:
                current_time = datetime.now()
                offline_nodes = []
                
                for node_id, node in self.network.nodes.items():
                    # Check if node is offline (no heartbeat for 30 seconds)
                    if (current_time - node.last_heartbeat).total_seconds() > 30:
                        if node.status == "online":
                            node.status = "offline"
                            offline_nodes.append(node_id)
                            
                            # Reassign running tasks from offline nodes
                            await self._reassign_tasks_from_offline_node(node_id)
                
                if offline_nodes:
                    self.logger.warning(f"Nodes went offline: {offline_nodes}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
            
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(10)
    
    async def _reassign_tasks_from_offline_node(self, node_id: str):
        """Reassign tasks from offline node to other nodes"""
        try:
            tasks_to_reassign = []
            
            for task_id, task in self.running_tasks.items():
                if task.assigned_node == node_id:
                    tasks_to_reassign.append(task)
            
            for task in tasks_to_reassign:
                # Remove from running tasks
                del self.running_tasks[task.id]
                
                # Reset task status
                task.assigned_node = None
                task.status = "pending"
                task.started_at = None
                
                # Re-queue task
                await self.task_queue.put(task)
            
            if tasks_to_reassign:
                self.logger.info(f"Reassigned {len(tasks_to_reassign)} tasks from offline node {node_id}")
        
        except Exception as e:
            self.logger.error(f"Error reassigning tasks: {e}")
    
    async def _network_optimizer(self):
        """Optimize edge network topology"""
        while True:
            try:
                # Update network latency measurements
                await self._update_network_latency()
                
                # Optimize routing table
                await self._update_routing_table()
                
                # Balance load across nodes
                await self._balance_load()
                
                await asyncio.sleep(60)  # Optimize every minute
            
            except Exception as e:
                self.logger.error(f"Error in network optimizer: {e}")
                await asyncio.sleep(60)
    
    async def _update_network_latency(self):
        """Update network latency measurements"""
        try:
            # Simulate latency measurement
            for node in self.network.nodes.values():
                # Simulate latency based on distance and network conditions
                base_latency = np.random.uniform(1, 10)  # 1-10ms base latency
                node.network_latency = base_latency
        
        except Exception as e:
            self.logger.error(f"Error updating network latency: {e}")
    
    async def _update_routing_table(self):
        """Update network routing table"""
        try:
            # Simple routing table update
            # In practice, this would use more sophisticated algorithms
            self.network.routing_table = {}
            
            for source_id in self.network.nodes.keys():
                self.network.routing_table[source_id] = {}
                for dest_id in self.network.nodes.keys():
                    if source_id != dest_id:
                        # Simple routing: direct connection if possible
                        self.network.routing_table[source_id][dest_id] = dest_id
        
        except Exception as e:
            self.logger.error(f"Error updating routing table: {e}")
    
    async def _balance_load(self):
        """Balance load across edge nodes"""
        try:
            # Calculate current load for each node
            node_loads = {}
            
            for node_id in self.network.nodes.keys():
                running_tasks_count = sum(
                    1 for task in self.running_tasks.values()
                    if task.assigned_node == node_id
                )
                node_loads[node_id] = running_tasks_count
            
            # Find overloaded and underloaded nodes
            avg_load = sum(node_loads.values()) / max(len(node_loads), 1)
            
            overloaded_nodes = [node_id for node_id, load in node_loads.items() if load > avg_load * 1.5]
            underloaded_nodes = [node_id for node_id, load in node_loads.items() if load < avg_load * 0.5]
            
            # Balance load (simplified implementation)
            if overloaded_nodes and underloaded_nodes:
                self.logger.info(f"Load balancing: {len(overloaded_nodes)} overloaded, {len(underloaded_nodes)} underloaded nodes")
        
        except Exception as e:
            self.logger.error(f"Error balancing load: {e}")
    
    async def _broadcast_node_registration(self, node: EdgeNode):
        """Broadcast node registration to other nodes"""
        try:
            message = {
                'type': 'node_registered',
                'node_id': node.id,
                'node_type': node.node_type.value,
                'capabilities': [cap.value for cap in node.capabilities],
                'timestamp': datetime.now().isoformat()
            }
            
            # Broadcast to all connected nodes
            for connection in self.websocket_connections.values():
                try:
                    await connection.send(json.dumps(message))
                except Exception as e:
                    self.logger.debug(f"Failed to send broadcast message: {e}")
        
        except Exception as e:
            self.logger.error(f"Error broadcasting node registration: {e}")
    
    async def get_edge_network_status(self) -> Dict[str, Any]:
        """Get edge network status"""
        try:
            total_nodes = len(self.network.nodes)
            online_nodes = len([n for n in self.network.nodes.values() if n.status == "online"])
            
            total_tasks = len(self.running_tasks) + len(self.completed_tasks)
            running_tasks = len(self.running_tasks)
            completed_tasks = len(self.completed_tasks)
            
            # Calculate average processing power
            avg_processing_power = np.mean([
                node.processing_power for node in self.network.nodes.values()
            ]) if self.network.nodes else 0.0
            
            # Calculate average network latency
            avg_latency = np.mean([
                node.network_latency for node in self.network.nodes.values()
            ]) if self.network.nodes else 0.0
            
            return {
                'total_nodes': total_nodes,
                'online_nodes': online_nodes,
                'offline_nodes': total_nodes - online_nodes,
                'total_tasks': total_tasks,
                'running_tasks': running_tasks,
                'completed_tasks': completed_tasks,
                'average_processing_power': round(avg_processing_power, 3),
                'average_network_latency': round(avg_latency, 2),
                'network_topology': {
                    'nodes': len(self.network.nodes),
                    'connections': len(self.network.connections),
                    'clusters': len(self.network.clusters)
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error getting edge network status: {e}")
            return {}

# Global edge computing manager
_edge_computing_manager: Optional[EdgeComputingManager] = None

def get_edge_computing_manager() -> EdgeComputingManager:
    """Get the global edge computing manager"""
    global _edge_computing_manager
    if _edge_computing_manager is None:
        _edge_computing_manager = EdgeComputingManager()
    return _edge_computing_manager

# Edge computing router
edge_router = APIRouter(prefix="/edge", tags=["Edge Computing"])

@edge_router.post("/register-node")
async def register_edge_node_endpoint(
    node_id: str = Field(..., description="Unique node identifier"),
    name: str = Field(..., description="Node name"),
    node_type: EdgeNodeType = Field(..., description="Type of edge node"),
    location: Dict[str, float] = Field(..., description="Node location (lat, lon, altitude)"),
    capabilities: List[ProcessingCapability] = Field(..., description="Processing capabilities"),
    resources: Dict[str, Any] = Field(..., description="Available resources")
):
    """Register a new edge node"""
    try:
        manager = get_edge_computing_manager()
        node = await manager.register_edge_node(
            node_id, name, node_type, location, capabilities, resources
        )
        return {"node": asdict(node), "success": True}
    
    except Exception as e:
        logger.error(f"Error registering edge node: {e}")
        raise HTTPException(status_code=500, detail="Failed to register edge node")

@edge_router.post("/submit-task")
async def submit_task_endpoint(
    task_type: str = Field(..., description="Type of task to execute"),
    data: Dict[str, Any] = Field(..., description="Task data"),
    requirements: Dict[str, Any] = Field(..., description="Task requirements"),
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
):
    """Submit a task for edge processing"""
    try:
        manager = get_edge_computing_manager()
        task = await manager.submit_task(task_type, data, requirements, priority)
        return {"task": asdict(task), "success": True}
    
    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit task")

@edge_router.get("/network-status")
async def get_network_status_endpoint():
    """Get edge network status"""
    try:
        manager = get_edge_computing_manager()
        status = await manager.get_edge_network_status()
        return {"status": status}
    
    except Exception as e:
        logger.error(f"Error getting network status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get network status")

@edge_router.get("/nodes")
async def get_edge_nodes_endpoint():
    """Get all edge nodes"""
    try:
        manager = get_edge_computing_manager()
        nodes = [asdict(node) for node in manager.network.nodes.values()]
        return {"nodes": nodes, "count": len(nodes)}
    
    except Exception as e:
        logger.error(f"Error getting edge nodes: {e}")
        raise HTTPException(status_code=500, detail="Failed to get edge nodes")

@edge_router.get("/tasks")
async def get_tasks_endpoint():
    """Get all tasks"""
    try:
        manager = get_edge_computing_manager()
        running_tasks = [asdict(task) for task in manager.running_tasks.values()]
        completed_tasks = [asdict(task) for task in manager.completed_tasks.values()]
        
        return {
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "total_running": len(running_tasks),
            "total_completed": len(completed_tasks)
        }
    
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tasks")


