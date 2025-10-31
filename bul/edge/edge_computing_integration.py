"""
Ultimate BUL System - Edge Computing & IoT Integration
Advanced edge computing capabilities for distributed document generation and real-time processing
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import redis
from prometheus_client import Counter, Histogram, Gauge
import time
import uuid

logger = logging.getLogger(__name__)

class EdgeNodeType(str, Enum):
    """Edge node types"""
    GATEWAY = "gateway"
    PROCESSOR = "processor"
    STORAGE = "storage"
    AI_INFERENCE = "ai_inference"
    ANALYTICS = "analytics"
    SECURITY = "security"

class IoTDeviceType(str, Enum):
    """IoT device types"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    DISPLAY = "display"
    PRINTER = "printer"
    SCANNER = "scanner"

class EdgeStatus(str, Enum):
    """Edge node status"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OVERLOADED = "overloaded"

@dataclass
class EdgeNode:
    """Edge computing node"""
    id: str
    name: str
    node_type: EdgeNodeType
    status: EdgeStatus
    location: Dict[str, float]  # lat, lng
    capabilities: List[str]
    resources: Dict[str, Any]
    last_heartbeat: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class IoTDevice:
    """IoT device"""
    id: str
    name: str
    device_type: IoTDeviceType
    edge_node_id: str
    status: EdgeStatus
    capabilities: List[str]
    data_format: str
    last_data: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class EdgeTask:
    """Edge computing task"""
    id: str
    task_type: str
    edge_node_id: str
    priority: int
    data: Dict[str, Any]
    status: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None

class EdgeComputingIntegration:
    """Edge computing and IoT integration system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.edge_nodes = {}
        self.iot_devices = {}
        self.edge_tasks = {}
        self.data_streams = {}
        
        # Redis for edge data caching
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 4)
        )
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Monitoring active
        self.monitoring_active = False
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""
        return {
            "edge_tasks": Counter(
                "bul_edge_tasks_total",
                "Total edge computing tasks",
                ["node_type", "task_type", "status"]
            ),
            "edge_task_duration": Histogram(
                "bul_edge_task_duration_seconds",
                "Edge task duration in seconds",
                ["node_type", "task_type"]
            ),
            "edge_node_resources": Gauge(
                "bul_edge_node_resources",
                "Edge node resource usage",
                ["node_id", "resource_type"]
            ),
            "iot_device_data": Counter(
                "bul_iot_device_data_total",
                "Total IoT device data points",
                ["device_type", "edge_node"]
            ),
            "edge_latency": Histogram(
                "bul_edge_latency_seconds",
                "Edge processing latency in seconds",
                ["node_type", "task_type"]
            ),
            "active_edge_nodes": Gauge(
                "bul_active_edge_nodes",
                "Number of active edge nodes"
            ),
            "active_iot_devices": Gauge(
                "bul_active_iot_devices",
                "Number of active IoT devices"
            )
        }
    
    async def start_monitoring(self):
        """Start edge computing monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting edge computing monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_edge_nodes())
        asyncio.create_task(self._monitor_iot_devices())
        asyncio.create_task(self._process_edge_tasks())
        asyncio.create_task(self._update_metrics())
    
    async def stop_monitoring(self):
        """Stop edge computing monitoring"""
        self.monitoring_active = False
        logger.info("Stopping edge computing monitoring")
    
    async def _monitor_edge_nodes(self):
        """Monitor edge nodes health"""
        while self.monitoring_active:
            try:
                current_time = datetime.utcnow()
                
                for node_id, node in self.edge_nodes.items():
                    # Check if node is responsive
                    time_since_heartbeat = (current_time - node.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > 300:  # 5 minutes
                        node.status = EdgeStatus.OFFLINE
                    elif time_since_heartbeat > 60:  # 1 minute
                        node.status = EdgeStatus.ERROR
                    else:
                        node.status = EdgeStatus.ONLINE
                    
                    # Check resource usage
                    if node.resources.get("cpu_usage", 0) > 90:
                        node.status = EdgeStatus.OVERLOADED
                
                # Update active nodes count
                active_nodes = len([n for n in self.edge_nodes.values() if n.status == EdgeStatus.ONLINE])
                self.prometheus_metrics["active_edge_nodes"].set(active_nodes)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring edge nodes: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_iot_devices(self):
        """Monitor IoT devices"""
        while self.monitoring_active:
            try:
                current_time = datetime.utcnow()
                
                for device_id, device in self.iot_devices.items():
                    # Check if device is responsive
                    time_since_data = (current_time - device.created_at).total_seconds()
                    
                    if time_since_data > 600:  # 10 minutes
                        device.status = EdgeStatus.OFFLINE
                    else:
                        device.status = EdgeStatus.ONLINE
                
                # Update active devices count
                active_devices = len([d for d in self.iot_devices.values() if d.status == EdgeStatus.ONLINE])
                self.prometheus_metrics["active_iot_devices"].set(active_devices)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring IoT devices: {e}")
                await asyncio.sleep(60)
    
    async def _process_edge_tasks(self):
        """Process edge computing tasks"""
        while self.monitoring_active:
            try:
                # Get pending tasks
                pending_tasks = [
                    task for task in self.edge_tasks.values()
                    if task.status == "pending"
                ]
                
                for task in pending_tasks:
                    # Find suitable edge node
                    suitable_node = self._find_suitable_edge_node(task)
                    
                    if suitable_node:
                        # Assign task to node
                        task.edge_node_id = suitable_node.id
                        task.status = "processing"
                        
                        # Process task
                        await self._execute_edge_task(task)
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Error processing edge tasks: {e}")
                await asyncio.sleep(30)
    
    async def _update_metrics(self):
        """Update Prometheus metrics"""
        while self.monitoring_active:
            try:
                # Update resource metrics
                for node_id, node in self.edge_nodes.items():
                    if node.resources:
                        for resource_type, value in node.resources.items():
                            self.prometheus_metrics["edge_node_resources"].labels(
                                node_id=node_id,
                                resource_type=resource_type
                            ).set(value)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    def _find_suitable_edge_node(self, task: EdgeTask) -> Optional[EdgeNode]:
        """Find suitable edge node for task"""
        suitable_nodes = [
            node for node in self.edge_nodes.values()
            if (node.status == EdgeStatus.ONLINE and
                task.task_type in node.capabilities and
                node.resources.get("cpu_usage", 0) < 80)
        ]
        
        if not suitable_nodes:
            return None
        
        # Return node with lowest CPU usage
        return min(suitable_nodes, key=lambda n: n.resources.get("cpu_usage", 0))
    
    async def _execute_edge_task(self, task: EdgeTask):
        """Execute edge computing task"""
        try:
            start_time = time.time()
            
            # Simulate task execution based on task type
            if task.task_type == "document_generation":
                result = await self._execute_document_generation_task(task)
            elif task.task_type == "ai_inference":
                result = await self._execute_ai_inference_task(task)
            elif task.task_type == "data_processing":
                result = await self._execute_data_processing_task(task)
            elif task.task_type == "analytics":
                result = await self._execute_analytics_task(task)
            else:
                result = {"status": "completed", "message": "Task executed successfully"}
            
            # Update task
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.result = result
            
            # Update metrics
            duration = time.time() - start_time
            self.prometheus_metrics["edge_task_duration"].labels(
                node_type=task.edge_node_id,
                task_type=task.task_type
            ).observe(duration)
            
            self.prometheus_metrics["edge_tasks"].labels(
                node_type=task.edge_node_id,
                task_type=task.task_type,
                status="completed"
            ).inc()
            
            logger.info(f"Edge task {task.id} completed on node {task.edge_node_id}")
            
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            
            self.prometheus_metrics["edge_tasks"].labels(
                node_type=task.edge_node_id,
                task_type=task.task_type,
                status="failed"
            ).inc()
            
            logger.error(f"Error executing edge task {task.id}: {e}")
    
    async def _execute_document_generation_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute document generation task on edge"""
        # Simulate document generation
        await asyncio.sleep(1)
        
        return {
            "document_id": f"doc_{uuid.uuid4().hex[:8]}",
            "content": task.data.get("content", ""),
            "format": task.data.get("format", "pdf"),
            "generated_at": datetime.utcnow().isoformat(),
            "edge_processed": True
        }
    
    async def _execute_ai_inference_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute AI inference task on edge"""
        # Simulate AI inference
        await asyncio.sleep(2)
        
        return {
            "inference_result": "AI inference completed on edge",
            "confidence": 0.95,
            "processing_time": 2.0,
            "edge_processed": True
        }
    
    async def _execute_data_processing_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute data processing task on edge"""
        # Simulate data processing
        await asyncio.sleep(1.5)
        
        return {
            "processed_data": "Data processed on edge",
            "records_processed": len(task.data.get("records", [])),
            "processing_time": 1.5,
            "edge_processed": True
        }
    
    async def _execute_analytics_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute analytics task on edge"""
        # Simulate analytics processing
        await asyncio.sleep(3)
        
        return {
            "analytics_result": "Analytics completed on edge",
            "insights": ["Edge processing advantage", "Low latency"],
            "processing_time": 3.0,
            "edge_processed": True
        }
    
    def register_edge_node(self, node: EdgeNode):
        """Register edge node"""
        self.edge_nodes[node.id] = node
        logger.info(f"Registered edge node: {node.id}")
    
    def unregister_edge_node(self, node_id: str):
        """Unregister edge node"""
        if node_id in self.edge_nodes:
            del self.edge_nodes[node_id]
            logger.info(f"Unregistered edge node: {node_id}")
    
    def register_iot_device(self, device: IoTDevice):
        """Register IoT device"""
        self.iot_devices[device.id] = device
        logger.info(f"Registered IoT device: {device.id}")
    
    def unregister_iot_device(self, device_id: str):
        """Unregister IoT device"""
        if device_id in self.iot_devices:
            del self.iot_devices[device_id]
            logger.info(f"Unregistered IoT device: {device_id}")
    
    async def submit_edge_task(self, task_type: str, data: Dict[str, Any], 
                             priority: int = 1) -> str:
        """Submit task to edge computing system"""
        try:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            task = EdgeTask(
                id=task_id,
                task_type=task_type,
                edge_node_id="",  # Will be assigned by scheduler
                priority=priority,
                data=data,
                status="pending"
            )
            
            self.edge_tasks[task_id] = task
            
            # Update metrics
            self.prometheus_metrics["edge_tasks"].labels(
                node_type="unknown",
                task_type=task_type,
                status="pending"
            ).inc()
            
            logger.info(f"Submitted edge task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error submitting edge task: {e}")
            raise
    
    async def get_edge_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get edge task result"""
        task = self.edge_tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.id,
            "status": task.status,
            "result": task.result,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        }
    
    async def process_iot_data(self, device_id: str, data: Dict[str, Any]):
        """Process IoT device data"""
        try:
            device = self.iot_devices.get(device_id)
            if not device:
                raise ValueError(f"Device {device_id} not found")
            
            # Update device data
            device.last_data = data
            device.status = EdgeStatus.ONLINE
            
            # Process data based on device type
            if device.device_type == IoTDeviceType.SENSOR:
                await self._process_sensor_data(device, data)
            elif device.device_type == IoTDeviceType.CAMERA:
                await self._process_camera_data(device, data)
            elif device.device_type == IoTDeviceType.MICROPHONE:
                await self._process_audio_data(device, data)
            elif device.device_type == IoTDeviceType.PRINTER:
                await self._process_printer_data(device, data)
            elif device.device_type == IoTDeviceType.SCANNER:
                await self._process_scanner_data(device, data)
            
            # Update metrics
            self.prometheus_metrics["iot_device_data"].labels(
                device_type=device.device_type.value,
                edge_node=device.edge_node_id
            ).inc()
            
            logger.info(f"Processed IoT data from device {device_id}")
            
        except Exception as e:
            logger.error(f"Error processing IoT data: {e}")
            raise
    
    async def _process_sensor_data(self, device: IoTDevice, data: Dict[str, Any]):
        """Process sensor data"""
        # Simulate sensor data processing
        await asyncio.sleep(0.1)
        
        # Store processed data
        processed_data = {
            "device_id": device.id,
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_type": data.get("type", "unknown"),
            "value": data.get("value", 0),
            "unit": data.get("unit", ""),
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Store in Redis
        await self.redis_client.setex(
            f"sensor_data:{device.id}",
            3600,  # 1 hour TTL
            json.dumps(processed_data)
        )
    
    async def _process_camera_data(self, device: IoTDevice, data: Dict[str, Any]):
        """Process camera data"""
        # Simulate camera data processing
        await asyncio.sleep(0.5)
        
        # Store processed data
        processed_data = {
            "device_id": device.id,
            "timestamp": datetime.utcnow().isoformat(),
            "image_url": data.get("image_url", ""),
            "metadata": data.get("metadata", {}),
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Store in Redis
        await self.redis_client.setex(
            f"camera_data:{device.id}",
            3600,  # 1 hour TTL
            json.dumps(processed_data)
        )
    
    async def _process_audio_data(self, device: IoTDevice, data: Dict[str, Any]):
        """Process audio data"""
        # Simulate audio data processing
        await asyncio.sleep(0.3)
        
        # Store processed data
        processed_data = {
            "device_id": device.id,
            "timestamp": datetime.utcnow().isoformat(),
            "audio_url": data.get("audio_url", ""),
            "duration": data.get("duration", 0),
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Store in Redis
        await self.redis_client.setex(
            f"audio_data:{device.id}",
            3600,  # 1 hour TTL
            json.dumps(processed_data)
        )
    
    async def _process_printer_data(self, device: IoTDevice, data: Dict[str, Any]):
        """Process printer data"""
        # Simulate printer data processing
        await asyncio.sleep(0.2)
        
        # Store processed data
        processed_data = {
            "device_id": device.id,
            "timestamp": datetime.utcnow().isoformat(),
            "print_job_id": data.get("job_id", ""),
            "status": data.get("status", "unknown"),
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Store in Redis
        await self.redis_client.setex(
            f"printer_data:{device.id}",
            3600,  # 1 hour TTL
            json.dumps(processed_data)
        )
    
    async def _process_scanner_data(self, device: IoTDevice, data: Dict[str, Any]):
        """Process scanner data"""
        # Simulate scanner data processing
        await asyncio.sleep(0.4)
        
        # Store processed data
        processed_data = {
            "device_id": device.id,
            "timestamp": datetime.utcnow().isoformat(),
            "scan_job_id": data.get("job_id", ""),
            "document_url": data.get("document_url", ""),
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Store in Redis
        await self.redis_client.setex(
            f"scanner_data:{device.id}",
            3600,  # 1 hour TTL
            json.dumps(processed_data)
        )
    
    def get_edge_node(self, node_id: str) -> Optional[EdgeNode]:
        """Get edge node by ID"""
        return self.edge_nodes.get(node_id)
    
    def list_edge_nodes(self, node_type: Optional[EdgeNodeType] = None) -> List[EdgeNode]:
        """List edge nodes"""
        nodes = list(self.edge_nodes.values())
        
        if node_type:
            nodes = [n for n in nodes if n.node_type == node_type]
        
        return nodes
    
    def get_iot_device(self, device_id: str) -> Optional[IoTDevice]:
        """Get IoT device by ID"""
        return self.iot_devices.get(device_id)
    
    def list_iot_devices(self, device_type: Optional[IoTDeviceType] = None) -> List[IoTDevice]:
        """List IoT devices"""
        devices = list(self.iot_devices.values())
        
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        
        return devices
    
    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get edge computing statistics"""
        total_nodes = len(self.edge_nodes)
        online_nodes = len([n for n in self.edge_nodes.values() if n.status == EdgeStatus.ONLINE])
        offline_nodes = total_nodes - online_nodes
        
        total_devices = len(self.iot_devices)
        online_devices = len([d for d in self.iot_devices.values() if d.status == EdgeStatus.ONLINE])
        offline_devices = total_devices - online_devices
        
        total_tasks = len(self.edge_tasks)
        completed_tasks = len([t for t in self.edge_tasks.values() if t.status == "completed"])
        pending_tasks = len([t for t in self.edge_tasks.values() if t.status == "pending"])
        failed_tasks = len([t for t in self.edge_tasks.values() if t.status == "failed"])
        
        # Count by node type
        node_type_counts = {}
        for node in self.edge_nodes.values():
            node_type = node.node_type.value
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        # Count by device type
        device_type_counts = {}
        for device in self.iot_devices.values():
            device_type = device.device_type.value
            device_type_counts[device_type] = device_type_counts.get(device_type, 0) + 1
        
        return {
            "total_nodes": total_nodes,
            "online_nodes": online_nodes,
            "offline_nodes": offline_nodes,
            "node_availability": (online_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            "total_devices": total_devices,
            "online_devices": online_devices,
            "offline_devices": offline_devices,
            "device_availability": (online_devices / total_devices * 100) if total_devices > 0 else 0,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "failed_tasks": failed_tasks,
            "task_success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "node_type_counts": node_type_counts,
            "device_type_counts": device_type_counts
        }
    
    def export_edge_data(self) -> Dict[str, Any]:
        """Export edge computing data for analysis"""
        return {
            "edge_nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "node_type": node.node_type.value,
                    "status": node.status.value,
                    "location": node.location,
                    "capabilities": node.capabilities,
                    "resources": node.resources,
                    "last_heartbeat": node.last_heartbeat.isoformat(),
                    "created_at": node.created_at.isoformat()
                }
                for node in self.edge_nodes.values()
            ],
            "iot_devices": [
                {
                    "id": device.id,
                    "name": device.name,
                    "device_type": device.device_type.value,
                    "edge_node_id": device.edge_node_id,
                    "status": device.status.value,
                    "capabilities": device.capabilities,
                    "data_format": device.data_format,
                    "last_data": device.last_data,
                    "created_at": device.created_at.isoformat()
                }
                for device in self.iot_devices.values()
            ],
            "edge_tasks": [
                {
                    "id": task.id,
                    "task_type": task.task_type,
                    "edge_node_id": task.edge_node_id,
                    "priority": task.priority,
                    "status": task.status,
                    "result": task.result,
                    "created_at": task.created_at.isoformat(),
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }
                for task in self.edge_tasks.values()
            ],
            "statistics": self.get_edge_statistics(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global edge computing integration instance
edge_integration = None

def get_edge_integration() -> EdgeComputingIntegration:
    """Get the global edge computing integration instance"""
    global edge_integration
    if edge_integration is None:
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 4
        }
        edge_integration = EdgeComputingIntegration(config)
    return edge_integration

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 4
        }
        
        edge = EdgeComputingIntegration(config)
        
        # Register edge node
        node = EdgeNode(
            id="edge_node_1",
            name="Gateway Node 1",
            node_type=EdgeNodeType.GATEWAY,
            status=EdgeStatus.ONLINE,
            location={"lat": 40.7128, "lng": -74.0060},
            capabilities=["document_generation", "ai_inference"],
            resources={"cpu_usage": 45, "memory_usage": 60},
            last_heartbeat=datetime.utcnow()
        )
        edge.register_edge_node(node)
        
        # Register IoT device
        device = IoTDevice(
            id="sensor_1",
            name="Temperature Sensor",
            device_type=IoTDeviceType.SENSOR,
            edge_node_id="edge_node_1",
            status=EdgeStatus.ONLINE,
            capabilities=["temperature_reading"],
            data_format="json"
        )
        edge.register_iot_device(device)
        
        # Submit edge task
        task_id = await edge.submit_edge_task(
            task_type="document_generation",
            data={"content": "Sample content", "format": "pdf"},
            priority=1
        )
        print(f"Submitted edge task: {task_id}")
        
        # Process IoT data
        await edge.process_iot_data("sensor_1", {
            "type": "temperature",
            "value": 23.5,
            "unit": "celsius"
        })
        
        # Get statistics
        stats = edge.get_edge_statistics()
        print("Edge Statistics:")
        print(json.dumps(stats, indent=2))
        
        await edge.stop_monitoring()
    
    asyncio.run(main())













