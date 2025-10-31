"""
Edge Computing and IoT Integration for Ultimate Opus Clip

Advanced edge computing capabilities for distributed video processing,
IoT device integration, and real-time edge analytics.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import asyncio
import time
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
import numpy as np
import cv2
import threading
from datetime import datetime, timedelta
import socket
import ssl
import websockets
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil

logger = structlog.get_logger("edge_computing")

class EdgeNodeType(Enum):
    """Types of edge nodes."""
    PROCESSING_NODE = "processing_node"
    STORAGE_NODE = "storage_node"
    ANALYTICS_NODE = "analytics_node"
    IOT_GATEWAY = "iot_gateway"
    CDN_NODE = "cdn_node"
    AI_INFERENCE_NODE = "ai_inference_node"

class IoTDeviceType(Enum):
    """Types of IoT devices."""
    CAMERA = "camera"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    DISPLAY = "display"
    AUDIO_DEVICE = "audio_device"

class ProcessingPriority(Enum):
    """Processing priorities."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BATCH = "batch"

@dataclass
class EdgeNode:
    """Edge computing node."""
    node_id: str
    node_type: EdgeNodeType
    location: Tuple[float, float]  # lat, lon
    capabilities: List[str]
    resources: Dict[str, Any]
    status: str
    last_heartbeat: float
    load_factor: float
    network_latency: float
    created_at: float = 0.0

@dataclass
class IoTDevice:
    """IoT device representation."""
    device_id: str
    device_type: IoTDeviceType
    node_id: str
    capabilities: List[str]
    status: str
    data_format: str
    sampling_rate: float
    last_data: Dict[str, Any]
    created_at: float = 0.0

@dataclass
class EdgeTask:
    """Edge computing task."""
    task_id: str
    task_type: str
    priority: ProcessingPriority
    input_data: Dict[str, Any]
    output_format: str
    requirements: Dict[str, Any]
    assigned_node: Optional[str] = None
    status: str = "pending"
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class EdgeMetrics:
    """Edge computing metrics."""
    node_id: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_usage: float
    storage_usage: float
    active_tasks: int
    queue_length: int
    timestamp: float

class EdgeNodeManager:
    """Edge node management system."""
    
    def __init__(self):
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.node_metrics: Dict[str, List[EdgeMetrics]] = {}
        self.task_queue: List[EdgeTask] = []
        self.active_tasks: Dict[str, EdgeTask] = {}
        self.completed_tasks: List[EdgeTask] = []
        
        logger.info("Edge Node Manager initialized")
    
    def register_node(self, node_id: str, node_type: EdgeNodeType, 
                     location: Tuple[float, float], capabilities: List[str]) -> bool:
        """Register edge node."""
        try:
            node = EdgeNode(
                node_id=node_id,
                node_type=node_type,
                location=location,
                capabilities=capabilities,
                resources=self._get_node_resources(),
                status="active",
                last_heartbeat=time.time(),
                load_factor=0.0,
                network_latency=0.0,
                created_at=time.time()
            )
            
            self.edge_nodes[node_id] = node
            self.node_metrics[node_id] = []
            
            logger.info(f"Edge node registered: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering edge node: {e}")
            return False
    
    def _get_node_resources(self) -> Dict[str, Any]:
        """Get node resource information."""
        try:
            return {
                "cpu_cores": psutil.cpu_count(),
                "total_memory": psutil.virtual_memory().total,
                "available_memory": psutil.virtual_memory().available,
                "gpu_count": len(GPUtil.getGPUs()) if GPUtil.getGPUs() else 0,
                "disk_space": psutil.disk_usage('/').free,
                "network_interfaces": len(psutil.net_if_addrs())
            }
        except Exception as e:
            logger.error(f"Error getting node resources: {e}")
            return {
                "cpu_cores": 4,
                "total_memory": 8 * 1024 * 1024 * 1024,  # 8GB
                "available_memory": 4 * 1024 * 1024 * 1024,  # 4GB
                "gpu_count": 0,
                "disk_space": 100 * 1024 * 1024 * 1024,  # 100GB
                "network_interfaces": 1
            }
    
    def update_node_metrics(self, node_id: str, metrics: EdgeMetrics) -> bool:
        """Update node metrics."""
        try:
            if node_id not in self.edge_nodes:
                return False
            
            # Update node status
            self.edge_nodes[node_id].last_heartbeat = time.time()
            self.edge_nodes[node_id].load_factor = metrics.cpu_usage
            
            # Store metrics
            self.node_metrics[node_id].append(metrics)
            
            # Keep only recent metrics (last 1000)
            if len(self.node_metrics[node_id]) > 1000:
                self.node_metrics[node_id] = self.node_metrics[node_id][-1000:]
            
            logger.info(f"Node metrics updated: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating node metrics: {e}")
            return False
    
    def get_best_node(self, task_requirements: Dict[str, Any]) -> Optional[str]:
        """Get best node for task based on requirements."""
        try:
            suitable_nodes = []
            
            for node_id, node in self.edge_nodes.items():
                if node.status != "active":
                    continue
                
                # Check if node has required capabilities
                required_capabilities = task_requirements.get("capabilities", [])
                if not all(cap in node.capabilities for cap in required_capabilities):
                    continue
                
                # Check resource requirements
                if not self._check_resource_requirements(node, task_requirements):
                    continue
                
                # Calculate node score
                score = self._calculate_node_score(node, task_requirements)
                suitable_nodes.append((node_id, score))
            
            if not suitable_nodes:
                return None
            
            # Return node with highest score
            suitable_nodes.sort(key=lambda x: x[1], reverse=True)
            return suitable_nodes[0][0]
            
        except Exception as e:
            logger.error(f"Error finding best node: {e}")
            return None
    
    def _check_resource_requirements(self, node: EdgeNode, requirements: Dict[str, Any]) -> bool:
        """Check if node meets resource requirements."""
        try:
            # Check CPU requirements
            required_cpu = requirements.get("cpu_cores", 1)
            if node.resources.get("cpu_cores", 0) < required_cpu:
                return False
            
            # Check memory requirements
            required_memory = requirements.get("memory_gb", 1) * 1024 * 1024 * 1024
            if node.resources.get("available_memory", 0) < required_memory:
                return False
            
            # Check GPU requirements
            required_gpu = requirements.get("gpu_count", 0)
            if node.resources.get("gpu_count", 0) < required_gpu:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking resource requirements: {e}")
            return False
    
    def _calculate_node_score(self, node: EdgeNode, requirements: Dict[str, Any]) -> float:
        """Calculate node suitability score."""
        try:
            score = 0.0
            
            # Base score
            score += 100.0
            
            # Deduct based on load factor
            score -= node.load_factor * 50.0
            
            # Deduct based on network latency
            score -= node.network_latency * 10.0
            
            # Bonus for matching capabilities
            required_capabilities = requirements.get("capabilities", [])
            matching_capabilities = sum(1 for cap in required_capabilities if cap in node.capabilities)
            score += matching_capabilities * 10.0
            
            # Bonus for resource availability
            cpu_utilization = 1.0 - (node.load_factor / 100.0)
            score += cpu_utilization * 20.0
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating node score: {e}")
            return 0.0
    
    async def submit_task(self, task_type: str, input_data: Dict[str, Any],
                         priority: ProcessingPriority = ProcessingPriority.MEDIUM,
                         requirements: Dict[str, Any] = None) -> str:
        """Submit task to edge computing system."""
        try:
            task = EdgeTask(
                task_id=str(uuid.uuid4()),
                task_type=task_type,
                priority=priority,
                input_data=input_data,
                output_format="json",
                requirements=requirements or {},
                created_at=time.time()
            )
            
            self.task_queue.append(task)
            
            # Sort queue by priority
            priority_order = {
                ProcessingPriority.CRITICAL: 0,
                ProcessingPriority.HIGH: 1,
                ProcessingPriority.MEDIUM: 2,
                ProcessingPriority.LOW: 3,
                ProcessingPriority.BATCH: 4
            }
            self.task_queue.sort(key=lambda x: priority_order.get(x.priority, 5))
            
            # Try to assign task immediately
            await self._assign_pending_tasks()
            
            logger.info(f"Task submitted: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            raise
    
    async def _assign_pending_tasks(self):
        """Assign pending tasks to available nodes."""
        try:
            for task in self.task_queue[:]:
                if task.assigned_node:
                    continue
                
                # Find best node for task
                best_node = self.get_best_node(task.requirements)
                if best_node:
                    task.assigned_node = best_node
                    task.status = "assigned"
                    self.active_tasks[task.task_id] = task
                    self.task_queue.remove(task)
                    
                    # Start task execution
                    asyncio.create_task(self._execute_task(task))
                    
        except Exception as e:
            logger.error(f"Error assigning tasks: {e}")
    
    async def _execute_task(self, task: EdgeTask):
        """Execute task on assigned node."""
        try:
            task.status = "running"
            task.started_at = time.time()
            
            # Simulate task execution based on type
            if task.task_type == "video_processing":
                result = await self._execute_video_processing_task(task)
            elif task.task_type == "ai_inference":
                result = await self._execute_ai_inference_task(task)
            elif task.task_type == "data_analysis":
                result = await self._execute_data_analysis_task(task)
            else:
                result = await self._execute_generic_task(task)
            
            task.result = result
            task.status = "completed"
            task.completed_at = time.time()
            
            # Move to completed tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)
            
            logger.info(f"Task completed: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            task.status = "failed"
            task.result = {"error": str(e)}
    
    async def _execute_video_processing_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute video processing task."""
        try:
            # Simulate video processing
            await asyncio.sleep(2.0)
            
            return {
                "task_type": "video_processing",
                "status": "completed",
                "processing_time": 2.0,
                "output_path": f"/edge/processed/{task.task_id}.mp4",
                "quality_score": 0.95
            }
            
        except Exception as e:
            logger.error(f"Error executing video processing task: {e}")
            return {"error": str(e)}
    
    async def _execute_ai_inference_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute AI inference task."""
        try:
            # Simulate AI inference
            await asyncio.sleep(1.5)
            
            return {
                "task_type": "ai_inference",
                "status": "completed",
                "inference_time": 1.5,
                "confidence": 0.92,
                "predictions": ["object_detected", "person_identified"]
            }
            
        except Exception as e:
            logger.error(f"Error executing AI inference task: {e}")
            return {"error": str(e)}
    
    async def _execute_data_analysis_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute data analysis task."""
        try:
            # Simulate data analysis
            await asyncio.sleep(1.0)
            
            return {
                "task_type": "data_analysis",
                "status": "completed",
                "analysis_time": 1.0,
                "insights": ["trend_detected", "anomaly_found"],
                "metrics": {"accuracy": 0.88, "precision": 0.91}
            }
            
        except Exception as e:
            logger.error(f"Error executing data analysis task: {e}")
            return {"error": str(e)}
    
    async def _execute_generic_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute generic task."""
        try:
            # Simulate generic processing
            await asyncio.sleep(0.5)
            
            return {
                "task_type": "generic",
                "status": "completed",
                "processing_time": 0.5,
                "result": "Task completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error executing generic task: {e}")
            return {"error": str(e)}
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status,
                "assigned_node": task.assigned_node,
                "created_at": task.created_at,
                "started_at": task.started_at
            }
        
        # Check completed tasks
        completed_task = next((t for t in self.completed_tasks if t.task_id == task_id), None)
        if completed_task:
            return {
                "task_id": task_id,
                "status": completed_task.status,
                "assigned_node": completed_task.assigned_node,
                "created_at": completed_task.created_at,
                "started_at": completed_task.started_at,
                "completed_at": completed_task.completed_at,
                "result": completed_task.result
            }
        
        # Check queued tasks
        queued_task = next((t for t in self.task_queue if t.task_id == task_id), None)
        if queued_task:
            return {
                "task_id": task_id,
                "status": "queued",
                "priority": queued_task.priority.value,
                "created_at": queued_task.created_at
            }
        
        return None

class IoTDeviceManager:
    """IoT device management system."""
    
    def __init__(self, edge_node_manager: EdgeNodeManager):
        self.edge_node_manager = edge_node_manager
        self.iot_devices: Dict[str, IoTDevice] = {}
        self.device_data: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("IoT Device Manager initialized")
    
    def register_device(self, device_id: str, device_type: IoTDeviceType,
                       node_id: str, capabilities: List[str]) -> bool:
        """Register IoT device."""
        try:
            device = IoTDevice(
                device_id=device_id,
                device_type=device_type,
                node_id=node_id,
                capabilities=capabilities,
                status="active",
                data_format="json",
                sampling_rate=1.0,
                last_data={},
                created_at=time.time()
            )
            
            self.iot_devices[device_id] = device
            self.device_data[device_id] = []
            
            logger.info(f"IoT device registered: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering IoT device: {e}")
            return False
    
    async def send_device_data(self, device_id: str, data: Dict[str, Any]) -> bool:
        """Send data from IoT device."""
        try:
            if device_id not in self.iot_devices:
                return False
            
            device = self.iot_devices[device_id]
            
            # Add timestamp and device info
            data_with_metadata = {
                **data,
                "device_id": device_id,
                "device_type": device.device_type.value,
                "node_id": device.node_id,
                "timestamp": time.time()
            }
            
            # Store data
            self.device_data[device_id].append(data_with_metadata)
            device.last_data = data_with_metadata
            
            # Keep only recent data (last 1000 points)
            if len(self.device_data[device_id]) > 1000:
                self.device_data[device_id] = self.device_data[device_id][-1000:]
            
            # Process data if needed
            await self._process_device_data(device_id, data_with_metadata)
            
            logger.info(f"Device data received: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending device data: {e}")
            return False
    
    async def _process_device_data(self, device_id: str, data: Dict[str, Any]):
        """Process device data."""
        try:
            device = self.iot_devices[device_id]
            
            # Submit processing task based on device type
            if device.device_type == IoTDeviceType.CAMERA:
                await self.edge_node_manager.submit_task(
                    "video_processing",
                    {"device_id": device_id, "data": data},
                    ProcessingPriority.HIGH,
                    {"capabilities": ["video_processing"], "cpu_cores": 2}
                )
            elif device.device_type == IoTDeviceType.SENSOR:
                await self.edge_node_manager.submit_task(
                    "data_analysis",
                    {"device_id": device_id, "data": data},
                    ProcessingPriority.MEDIUM,
                    {"capabilities": ["data_analysis"], "cpu_cores": 1}
                )
            
        except Exception as e:
            logger.error(f"Error processing device data: {e}")
    
    def get_device_data(self, device_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get device data."""
        return self.device_data.get(device_id, [])[-limit:] if device_id in self.device_data else []
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device status."""
        if device_id in self.iot_devices:
            device = self.iot_devices[device_id]
            return {
                "device_id": device_id,
                "device_type": device.device_type.value,
                "node_id": device.node_id,
                "status": device.status,
                "last_data_time": device.last_data.get("timestamp", 0),
                "data_points": len(self.device_data.get(device_id, []))
            }
        return None

class EdgeComputingSystem:
    """Main edge computing system."""
    
    def __init__(self):
        self.edge_node_manager = EdgeNodeManager()
        self.iot_device_manager = IoTDeviceManager(self.edge_node_manager)
        self.system_metrics: List[Dict[str, Any]] = []
        
        logger.info("Edge Computing System initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get edge computing system status."""
        return {
            "total_nodes": len(self.edge_node_manager.edge_nodes),
            "active_nodes": len([n for n in self.edge_node_manager.edge_nodes.values() if n.status == "active"]),
            "total_devices": len(self.iot_device_manager.iot_devices),
            "active_devices": len([d for d in self.iot_device_manager.iot_devices.values() if d.status == "active"]),
            "queued_tasks": len(self.edge_node_manager.task_queue),
            "active_tasks": len(self.edge_node_manager.active_tasks),
            "completed_tasks": len(self.edge_node_manager.completed_tasks)
        }
    
    async def start_system_monitoring(self):
        """Start system monitoring."""
        try:
            while True:
                # Collect system metrics
                metrics = {
                    "timestamp": time.time(),
                    "system_status": self.get_system_status(),
                    "node_metrics": {
                        node_id: metrics_list[-1] if metrics_list else None
                        for node_id, metrics_list in self.edge_node_manager.node_metrics.items()
                    }
                }
                
                self.system_metrics.append(metrics)
                
                # Keep only recent metrics
                if len(self.system_metrics) > 1000:
                    self.system_metrics = self.system_metrics[-1000:]
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")

# Global edge computing system instance
_global_edge_system: Optional[EdgeComputingSystem] = None

def get_edge_system() -> EdgeComputingSystem:
    """Get the global edge computing system instance."""
    global _global_edge_system
    if _global_edge_system is None:
        _global_edge_system = EdgeComputingSystem()
    return _global_edge_system

async def submit_edge_task(task_type: str, input_data: Dict[str, Any],
                          priority: ProcessingPriority = ProcessingPriority.MEDIUM) -> str:
    """Submit task to edge computing system."""
    edge_system = get_edge_system()
    return await edge_system.edge_node_manager.submit_task(task_type, input_data, priority)

def register_edge_node(node_id: str, node_type: EdgeNodeType, 
                      location: Tuple[float, float], capabilities: List[str]) -> bool:
    """Register edge node."""
    edge_system = get_edge_system()
    return edge_system.edge_node_manager.register_node(node_id, node_type, location, capabilities)

def register_iot_device(device_id: str, device_type: IoTDeviceType,
                       node_id: str, capabilities: List[str]) -> bool:
    """Register IoT device."""
    edge_system = get_edge_system()
    return edge_system.iot_device_manager.register_device(device_id, device_type, node_id, capabilities)


