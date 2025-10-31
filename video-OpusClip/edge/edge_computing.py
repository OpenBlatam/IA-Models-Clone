#!/usr/bin/env python3
"""
Edge Computing System

Advanced edge computing with:
- Edge node management
- Distributed processing
- Edge analytics
- Edge storage
- Edge AI/ML inference
- Edge-to-cloud synchronization
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import aiohttp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

logger = structlog.get_logger("edge_computing")

# =============================================================================
# EDGE COMPUTING MODELS
# =============================================================================

class EdgeNodeType(Enum):
    """Edge node types."""
    GATEWAY = "gateway"
    EDGE_SERVER = "edge_server"
    MOBILE_DEVICE = "mobile_device"
    IOT_DEVICE = "iot_device"
    CLOUD_EDGE = "cloud_edge"
    FOG_NODE = "fog_node"

class EdgeNodeStatus(Enum):
    """Edge node status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OVERLOADED = "overloaded"

class ProcessingTaskType(Enum):
    """Processing task types."""
    VIDEO_PROCESSING = "video_processing"
    IMAGE_ANALYSIS = "image_analysis"
    DATA_ANALYTICS = "data_analytics"
    ML_INFERENCE = "ml_inference"
    DATA_COMPRESSION = "data_compression"
    REAL_TIME_ANALYSIS = "real_time_analysis"
    BATCH_PROCESSING = "batch_processing"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class EdgeNode:
    """Edge node information."""
    node_id: str
    name: str
    node_type: EdgeNodeType
    status: EdgeNodeStatus
    location: Dict[str, float]  # lat, lng, alt
    capabilities: List[str]
    resources: Dict[str, Any]  # CPU, memory, storage, etc.
    network_info: Dict[str, Any]
    last_heartbeat: datetime
    created_at: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.last_heartbeat:
            self.last_heartbeat = datetime.utcnow()
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "location": self.location,
            "capabilities": self.capabilities,
            "resources": self.resources,
            "network_info": self.network_info,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class ProcessingTask:
    """Edge processing task."""
    task_id: str
    task_type: ProcessingTaskType
    priority: TaskPriority
    input_data: Dict[str, Any]
    parameters: Dict[str, Any]
    assigned_node: Optional[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Any]
    error_message: Optional[str]
    estimated_duration: int
    actual_duration: Optional[int]
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "priority": self.priority.value,
            "input_data": self.input_data,
            "parameters": self.parameters,
            "assigned_node": self.assigned_node,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error_message": self.error_message,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration
        }

@dataclass
class EdgeStorage:
    """Edge storage information."""
    storage_id: str
    node_id: str
    storage_type: str
    capacity: int
    used_space: int
    available_space: int
    access_speed: float
    reliability: float
    last_sync: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage_id": self.storage_id,
            "node_id": self.node_id,
            "storage_type": self.storage_type,
            "capacity": self.capacity,
            "used_space": self.used_space,
            "available_space": self.available_space,
            "access_speed": self.access_speed,
            "reliability": self.reliability,
            "last_sync": self.last_sync.isoformat()
        }

# =============================================================================
# EDGE COMPUTING MANAGER
# =============================================================================

class EdgeComputingManager:
    """Edge computing management system."""
    
    def __init__(self):
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.processing_tasks: Dict[str, ProcessingTask] = {}
        self.task_queue: List[ProcessingTask] = []
        self.edge_storage: Dict[str, EdgeStorage] = {}
        self.node_capabilities: Dict[str, List[str]] = defaultdict(list)
        
        # Task execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'online_nodes': 0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.task_scheduler_task: Optional[asyncio.Task] = None
        self.sync_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start the edge computing manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.task_scheduler_task = asyncio.create_task(self._task_scheduler_loop())
        self.sync_task = asyncio.create_task(self._sync_loop())
        
        logger.info("Edge Computing Manager started")
    
    async def stop(self) -> None:
        """Stop the edge computing manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.task_scheduler_task:
            self.task_scheduler_task.cancel()
        if self.sync_task:
            self.sync_task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Edge Computing Manager stopped")
    
    def register_edge_node(self, node: EdgeNode) -> str:
        """Register a new edge node."""
        self.edge_nodes[node.node_id] = node
        self.node_capabilities[node.node_id] = node.capabilities
        
        self.stats['total_nodes'] += 1
        if node.status == EdgeNodeStatus.ONLINE:
            self.stats['online_nodes'] += 1
        
        logger.info(
            "Edge node registered",
            node_id=node.node_id,
            name=node.name,
            type=node.node_type.value,
            capabilities=node.capabilities
        )
        
        return node.node_id
    
    def unregister_edge_node(self, node_id: str) -> bool:
        """Unregister an edge node."""
        if node_id in self.edge_nodes:
            node = self.edge_nodes[node_id]
            
            # Update statistics
            self.stats['total_nodes'] -= 1
            if node.status == EdgeNodeStatus.ONLINE:
                self.stats['online_nodes'] -= 1
            
            # Clean up
            del self.edge_nodes[node_id]
            if node_id in self.node_capabilities:
                del self.node_capabilities[node_id]
            
            # Cancel any running tasks on this node
            for task_id, task in list(self.running_tasks.items()):
                if task_id in self.processing_tasks:
                    processing_task = self.processing_tasks[task_id]
                    if processing_task.assigned_node == node_id:
                        task.cancel()
                        processing_task.status = "cancelled"
                        processing_task.error_message = "Node unregistered"
            
            logger.info("Edge node unregistered", node_id=node_id)
            return True
        
        return False
    
    def update_node_status(self, node_id: str, status: EdgeNodeStatus, resources: Optional[Dict[str, Any]] = None) -> bool:
        """Update edge node status."""
        if node_id not in self.edge_nodes:
            return False
        
        node = self.edge_nodes[node_id]
        old_status = node.status
        
        node.status = status
        node.last_heartbeat = datetime.utcnow()
        
        if resources:
            node.resources.update(resources)
        
        # Update statistics
        if old_status == EdgeNodeStatus.ONLINE and status != EdgeNodeStatus.ONLINE:
            self.stats['online_nodes'] -= 1
        elif old_status != EdgeNodeStatus.ONLINE and status == EdgeNodeStatus.ONLINE:
            self.stats['online_nodes'] += 1
        
        logger.info(
            "Node status updated",
            node_id=node_id,
            old_status=old_status.value,
            new_status=status.value
        )
        
        return True
    
    async def submit_task(self, task: ProcessingTask) -> str:
        """Submit a processing task."""
        self.processing_tasks[task.task_id] = task
        self.task_queue.append(task)
        
        # Sort queue by priority
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        self.stats['total_tasks'] += 1
        
        logger.info(
            "Task submitted",
            task_id=task.task_id,
            task_type=task.task_type.value,
            priority=task.priority.value
        )
        
        return task.task_id
    
    async def _task_scheduler_loop(self) -> None:
        """Task scheduler loop."""
        while self.is_running:
            try:
                # Process task queue
                if self.task_queue:
                    task = self.task_queue[0]
                    
                    # Find suitable node
                    suitable_node = await self._find_suitable_node(task)
                    
                    if suitable_node:
                        # Assign task to node
                        task.assigned_node = suitable_node
                        task.status = "assigned"
                        self.task_queue.pop(0)
                        
                        # Start task execution
                        asyncio.create_task(self._execute_task(task))
                        
                        logger.info(
                            "Task assigned to node",
                            task_id=task.task_id,
                            node_id=suitable_node
                        )
                    else:
                        # No suitable node available, wait
                        await asyncio.sleep(1)
                else:
                    await asyncio.sleep(1)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Task scheduler error", error=str(e))
                await asyncio.sleep(1)
    
    async def _find_suitable_node(self, task: ProcessingTask) -> Optional[str]:
        """Find a suitable node for the task."""
        suitable_nodes = []
        
        for node_id, node in self.edge_nodes.items():
            if node.status != EdgeNodeStatus.ONLINE:
                continue
            
            # Check if node has required capabilities
            if not self._node_has_capabilities(node_id, task.task_type):
                continue
            
            # Check if node has enough resources
            if not self._node_has_resources(node_id, task):
                continue
            
            # Calculate node score (lower is better)
            score = self._calculate_node_score(node_id, task)
            suitable_nodes.append((node_id, score))
        
        if not suitable_nodes:
            return None
        
        # Return node with lowest score
        suitable_nodes.sort(key=lambda x: x[1])
        return suitable_nodes[0][0]
    
    def _node_has_capabilities(self, node_id: str, task_type: ProcessingTaskType) -> bool:
        """Check if node has required capabilities."""
        capabilities = self.node_capabilities.get(node_id, [])
        
        capability_mapping = {
            ProcessingTaskType.VIDEO_PROCESSING: ["video_processing", "gpu"],
            ProcessingTaskType.IMAGE_ANALYSIS: ["image_processing", "ml_inference"],
            ProcessingTaskType.DATA_ANALYTICS: ["data_processing", "analytics"],
            ProcessingTaskType.ML_INFERENCE: ["ml_inference", "gpu"],
            ProcessingTaskType.DATA_COMPRESSION: ["data_processing", "compression"],
            ProcessingTaskType.REAL_TIME_ANALYSIS: ["real_time_processing", "low_latency"],
            ProcessingTaskType.BATCH_PROCESSING: ["batch_processing", "high_throughput"]
        }
        
        required_capabilities = capability_mapping.get(task_type, [])
        return all(cap in capabilities for cap in required_capabilities)
    
    def _node_has_resources(self, node_id: str, task: ProcessingTask) -> bool:
        """Check if node has enough resources."""
        node = self.edge_nodes.get(node_id)
        if not node:
            return False
        
        resources = node.resources
        
        # Check CPU usage
        cpu_usage = resources.get('cpu_usage', 0)
        if cpu_usage > 80:  # 80% threshold
            return False
        
        # Check memory usage
        memory_usage = resources.get('memory_usage', 0)
        if memory_usage > 85:  # 85% threshold
            return False
        
        # Check storage
        storage_usage = resources.get('storage_usage', 0)
        if storage_usage > 90:  # 90% threshold
            return False
        
        return True
    
    def _calculate_node_score(self, node_id: str, task: ProcessingTask) -> float:
        """Calculate node score for task assignment."""
        node = self.edge_nodes.get(node_id)
        if not node:
            return float('inf')
        
        resources = node.resources
        
        # Base score from resource usage
        cpu_usage = resources.get('cpu_usage', 0)
        memory_usage = resources.get('memory_usage', 0)
        storage_usage = resources.get('storage_usage', 0)
        
        resource_score = (cpu_usage + memory_usage + storage_usage) / 3
        
        # Distance penalty (if location is available)
        distance_penalty = 0
        if 'location' in task.input_data and node.location:
            distance = self._calculate_distance(
                task.input_data['location'],
                node.location
            )
            distance_penalty = distance * 0.1  # 0.1 penalty per km
        
        # Network latency penalty
        network_penalty = resources.get('network_latency', 0) * 0.01
        
        total_score = resource_score + distance_penalty + network_penalty
        return total_score
    
    def _calculate_distance(self, loc1: Dict[str, float], loc2: Dict[str, float]) -> float:
        """Calculate distance between two locations."""
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1 = loc1.get('lat', 0), loc1.get('lng', 0)
        lat2, lon2 = loc2.get('lat', 0), loc2.get('lng', 0)
        
        # Haversine formula
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth's radius in kilometers
        return c * r
    
    async def _execute_task(self, task: ProcessingTask) -> None:
        """Execute a processing task."""
        try:
            task.status = "running"
            task.started_at = datetime.utcnow()
            
            # Run task in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_task_sync,
                task
            )
            
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            
            # Calculate actual duration
            if task.started_at:
                task.actual_duration = int((task.completed_at - task.started_at).total_seconds())
            
            # Update statistics
            self.stats['completed_tasks'] += 1
            if task.actual_duration:
                self._update_average_processing_time(task.actual_duration)
            
            logger.info(
                "Task completed",
                task_id=task.task_id,
                duration=task.actual_duration,
                node_id=task.assigned_node
            )
        
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            
            # Update statistics
            self.stats['failed_tasks'] += 1
            
            logger.error(
                "Task failed",
                task_id=task.task_id,
                error=str(e),
                node_id=task.assigned_node
            )
    
    def _run_task_sync(self, task: ProcessingTask) -> Any:
        """Run task synchronously (in thread pool)."""
        # Simulate task processing based on type
        if task.task_type == ProcessingTaskType.VIDEO_PROCESSING:
            return self._process_video_task(task)
        elif task.task_type == ProcessingTaskType.IMAGE_ANALYSIS:
            return self._process_image_analysis_task(task)
        elif task.task_type == ProcessingTaskType.DATA_ANALYTICS:
            return self._process_data_analytics_task(task)
        elif task.task_type == ProcessingTaskType.ML_INFERENCE:
            return self._process_ml_inference_task(task)
        else:
            return self._process_generic_task(task)
    
    def _process_video_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process video task."""
        # Simulate video processing
        time.sleep(min(task.estimated_duration, 10))  # Cap at 10 seconds for demo
        
        return {
            "processed_frames": 1000,
            "processing_time": task.estimated_duration,
            "quality_score": 0.95,
            "output_size": 1024 * 1024  # 1MB
        }
    
    def _process_image_analysis_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process image analysis task."""
        # Simulate image analysis
        time.sleep(min(task.estimated_duration, 5))  # Cap at 5 seconds for demo
        
        return {
            "objects_detected": 5,
            "confidence_scores": [0.95, 0.87, 0.92, 0.78, 0.89],
            "analysis_time": task.estimated_duration
        }
    
    def _process_data_analytics_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process data analytics task."""
        # Simulate data analytics
        time.sleep(min(task.estimated_duration, 3))  # Cap at 3 seconds for demo
        
        return {
            "data_points_processed": 10000,
            "insights": ["trend_up", "anomaly_detected", "correlation_found"],
            "processing_time": task.estimated_duration
        }
    
    def _process_ml_inference_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process ML inference task."""
        # Simulate ML inference
        time.sleep(min(task.estimated_duration, 2))  # Cap at 2 seconds for demo
        
        return {
            "prediction": 0.85,
            "confidence": 0.92,
            "inference_time": task.estimated_duration,
            "model_version": "1.0.0"
        }
    
    def _process_generic_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process generic task."""
        # Simulate generic processing
        time.sleep(min(task.estimated_duration, 1))  # Cap at 1 second for demo
        
        return {
            "status": "completed",
            "processing_time": task.estimated_duration,
            "result": "Task completed successfully"
        }
    
    def _update_average_processing_time(self, processing_time: int) -> None:
        """Update average processing time."""
        completed_tasks = self.stats['completed_tasks']
        current_avg = self.stats['average_processing_time']
        
        if completed_tasks > 0:
            self.stats['average_processing_time'] = (
                (current_avg * (completed_tasks - 1) + processing_time) / completed_tasks
            )
        else:
            self.stats['average_processing_time'] = processing_time
        
        self.stats['total_processing_time'] += processing_time
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to check node status."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for node_id, node in list(self.edge_nodes.items()):
                    time_since_heartbeat = (current_time - node.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > 300:  # 5 minutes timeout
                        if node.status == EdgeNodeStatus.ONLINE:
                            node.status = EdgeNodeStatus.OFFLINE
                            self.stats['online_nodes'] -= 1
                            
                            logger.warning(
                                "Node marked as offline",
                                node_id=node_id,
                                time_since_heartbeat=time_since_heartbeat
                            )
                
                await asyncio.sleep(60)  # Check every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def _sync_loop(self) -> None:
        """Synchronization loop for edge-to-cloud sync."""
        while self.is_running:
            try:
                # Sync completed tasks to cloud
                await self._sync_completed_tasks()
                
                # Sync node status to cloud
                await self._sync_node_status()
                
                await asyncio.sleep(300)  # Sync every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sync loop error", error=str(e))
                await asyncio.sleep(300)
    
    async def _sync_completed_tasks(self) -> None:
        """Sync completed tasks to cloud."""
        # In a real implementation, you'd sync with cloud storage
        completed_tasks = [
            task for task in self.processing_tasks.values()
            if task.status == "completed"
        ]
        
        if completed_tasks:
            logger.info(f"Syncing {len(completed_tasks)} completed tasks to cloud")
    
    async def _sync_node_status(self) -> None:
        """Sync node status to cloud."""
        # In a real implementation, you'd sync node status with cloud
        online_nodes = [
            node for node in self.edge_nodes.values()
            if node.status == EdgeNodeStatus.ONLINE
        ]
        
        if online_nodes:
            logger.info(f"Syncing status of {len(online_nodes)} online nodes to cloud")
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task status."""
        return self.processing_tasks.get(task_id)
    
    def get_node_stats(self, node_id: str) -> Dict[str, Any]:
        """Get node statistics."""
        if node_id not in self.edge_nodes:
            return {}
        
        node = self.edge_nodes[node_id]
        
        # Count tasks assigned to this node
        assigned_tasks = len([
            task for task in self.processing_tasks.values()
            if task.assigned_node == node_id
        ])
        
        return {
            'node_id': node_id,
            'name': node.name,
            'type': node.node_type.value,
            'status': node.status.value,
            'capabilities': node.capabilities,
            'resources': node.resources,
            'assigned_tasks': assigned_tasks,
            'last_heartbeat': node.last_heartbeat.isoformat(),
            'uptime': (datetime.utcnow() - node.created_at).total_seconds()
        }
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            'nodes': {
                node_id: self.get_node_stats(node_id)
                for node_id in self.edge_nodes
            },
            'task_queue_size': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks_percentage': (
                self.stats['completed_tasks'] / max(1, self.stats['total_tasks']) * 100
            )
        }

# =============================================================================
# GLOBAL EDGE COMPUTING INSTANCES
# =============================================================================

# Global edge computing manager
edge_computing_manager = EdgeComputingManager()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'EdgeNodeType',
    'EdgeNodeStatus',
    'ProcessingTaskType',
    'TaskPriority',
    'EdgeNode',
    'ProcessingTask',
    'EdgeStorage',
    'EdgeComputingManager',
    'edge_computing_manager'
]





























