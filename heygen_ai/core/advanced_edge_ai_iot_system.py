import logging
import time
import json
import os
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from pathlib import Path
import hashlib
import yaml
import threading
import queue
import uuid

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Enums
class DeviceType(Enum):
    """IoT device types."""
    SENSOR = "sensor"
    CAMERA = "camera"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    EDGE_SERVER = "edge_server"
    MOBILE_DEVICE = "mobile_device"
    EMBEDDED = "embedded"

class EdgeTier(Enum):
    """Edge computing tiers."""
    DEVICE = "device"
    GATEWAY = "gateway"
    EDGE = "edge"
    FOG = "fog"
    CLOUD = "cloud"

class InferenceMode(Enum):
    """Inference execution modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"

class DeviceStatus(Enum):
    """Device operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SLEEP = "sleep"

# Configurations
@dataclass
class EdgeAIConfig:
    """Configuration for Advanced Edge AI & IoT System."""
    # Core Settings
    enable_edge_computing: bool = True
    enable_iot_management: bool = True
    enable_real_time_inference: bool = True
    enable_edge_cloud_orchestration: bool = True
    
    # Edge Computing
    edge_tier: EdgeTier = EdgeTier.EDGE
    max_edge_devices: int = 100
    enable_edge_optimization: bool = True
    enable_edge_caching: bool = True
    
    # IoT Management
    enable_device_discovery: bool = True
    enable_auto_provisioning: bool = True
    enable_device_monitoring: bool = True
    device_heartbeat_interval: int = 30
    
    # Real-Time Inference
    inference_latency_target_ms: int = 100
    enable_model_compression: bool = True
    enable_quantization: bool = True
    enable_pruning: bool = True
    
    # Edge-Cloud Orchestration
    enable_load_balancing: bool = True
    enable_failover: bool = True
    enable_data_sync: bool = True
    sync_interval_seconds: int = 60
    
    # Security
    enable_edge_security: bool = True
    enable_device_authentication: bool = True
    enable_data_encryption: bool = True
    
    # Logging
    log_level: str = "INFO"
    enable_edge_logging: bool = True

# Core Classes
class IoTDevice:
    """Represents an IoT device in the edge network."""
    
    def __init__(self, device_id: str, device_type: DeviceType, 
                 capabilities: Dict[str, Any], location: Optional[Dict[str, float]] = None):
        self.device_id = device_id
        self.device_type = device_type
        self.capabilities = capabilities
        self.location = location or {}
        self.status = DeviceStatus.OFFLINE
        self.last_heartbeat = None
        self.resources = {
            "cpu": 0.0,
            "memory": 0.0,
            "storage": 0.0,
            "network": 0.0
        }
        self.models = {}
        self.data_queue = queue.Queue()
        self.logger = logging.getLogger(f"{__name__}.device.{device_id}")
    
    def update_heartbeat(self):
        """Update device heartbeat."""
        self.last_heartbeat = datetime.now()
        self.status = DeviceStatus.ONLINE
    
    def update_resources(self, cpu: float, memory: float, storage: float, network: float):
        """Update device resource usage."""
        self.resources = {
            "cpu": cpu,
            "memory": memory,
            "storage": storage,
            "network": network
        }
    
    def add_model(self, model_id: str, model_info: Dict[str, Any]):
        """Add a model to the device."""
        self.models[model_id] = model_info
    
    def remove_model(self, model_id: str):
        """Remove a model from the device."""
        if model_id in self.models:
            del self.models[model_id]
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "capabilities": self.capabilities,
            "location": self.location,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "resources": self.resources,
            "models": list(self.models.keys())
        }

class EdgeComputingOptimizer:
    """Optimizes edge computing operations and resource allocation."""
    
    def __init__(self, config: EdgeAIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.optimizer")
        self.optimization_history = []
        self.resource_pools = {}
    
    def optimize_edge_deployment(self, devices: List[IoTDevice], 
                                models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize model deployment across edge devices."""
        self.logger.info(f"Optimizing deployment for {len(models)} models across {len(devices)} devices")
        
        # Simple optimization: distribute models based on device capabilities
        deployment_plan = {}
        device_loads = {device.device_id: 0 for device in devices}
        
        for model in models:
            best_device = self._find_best_device(devices, model, device_loads)
            if best_device:
                deployment_plan[model["id"]] = best_device.device_id
                device_loads[best_device.device_id] += 1
        
        optimization_result = {
            "deployment_plan": deployment_plan,
            "device_loads": device_loads,
            "optimization_score": self._calculate_optimization_score(deployment_plan, devices),
            "timestamp": datetime.now().isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def _find_best_device(self, devices: List[IoTDevice], model: Dict[str, Any], 
                          device_loads: Dict[str, int]) -> Optional[IoTDevice]:
        """Find the best device for a model based on capabilities and current load."""
        best_device = None
        best_score = -1
        
        for device in devices:
            if device.status != DeviceStatus.ONLINE:
                continue
            
            score = self._calculate_device_score(device, model, device_loads[device.device_id])
            if score > best_score:
                best_score = score
                best_device = device
        
        return best_device
    
    def _calculate_device_score(self, device: IoTDevice, model: Dict[str, Any], 
                               current_load: int) -> float:
        """Calculate device suitability score for a model."""
        # Base score from device capabilities
        capability_score = 0.0
        if "gpu" in device.capabilities and model.get("requires_gpu", False):
            capability_score += 2.0
        if "memory" in device.capabilities and device.capabilities["memory"] >= model.get("memory_required", 0):
            capability_score += 1.0
        
        # Load balancing score (prefer less loaded devices)
        load_score = max(0, 10 - current_load)
        
        # Resource availability score
        resource_score = (1.0 - device.resources["cpu"]) * 2.0 + \
                        (1.0 - device.resources["memory"]) * 2.0
        
        return capability_score + load_score + resource_score
    
    def _calculate_optimization_score(self, deployment_plan: Dict[str, str], 
                                    devices: List[IoTDevice]) -> float:
        """Calculate overall optimization score."""
        if not deployment_plan:
            return 0.0
        
        # Calculate load distribution
        device_loads = {}
        for device in devices:
            device_loads[device.device_id] = 0
        
        for model_id, device_id in deployment_plan.items():
            if device_id in device_loads:
                device_loads[device_id] += 1
        
        # Score based on load balance (lower variance is better)
        loads = list(device_loads.values())
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        
        # Normalize score (0-100)
        balance_score = max(0, 100 - variance * 10)
        
        return balance_score

class RealTimeInferenceEngine:
    """Handles real-time inference execution on edge devices."""
    
    def __init__(self, config: EdgeAIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.inference")
        self.active_inferences = {}
        self.inference_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency_ms": 0.0
        }
    
    async def execute_inference(self, device: IoTDevice, model_id: str, 
                               input_data: Any, mode: InferenceMode = InferenceMode.REAL_TIME) -> Dict[str, Any]:
        """Execute inference on a device."""
        inference_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Starting inference {inference_id} on device {device.device_id}")
        
        try:
            # Check if model is available on device
            if model_id not in device.models:
                raise ValueError(f"Model {model_id} not available on device {device.device_id}")
            
            # Execute inference based on mode
            if mode == InferenceMode.REAL_TIME:
                result = await self._execute_real_time_inference(device, model_id, input_data)
            elif mode == InferenceMode.BATCH:
                result = await self._execute_batch_inference(device, model_id, input_data)
            elif mode == InferenceMode.STREAMING:
                result = await self._execute_streaming_inference(device, model_id, input_data)
            else:
                result = await self._execute_hybrid_inference(device, model_id, input_data)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_inference_stats(True, latency_ms)
            
            inference_result = {
                "inference_id": inference_id,
                "device_id": device.device_id,
                "model_id": model_id,
                "result": result,
                "latency_ms": latency_ms,
                "mode": mode.value,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.active_inferences[inference_id] = inference_result
            return inference_result
            
        except Exception as e:
            self.logger.error(f"Inference {inference_id} failed: {e}")
            self._update_inference_stats(False, 0)
            
            return {
                "inference_id": inference_id,
                "device_id": device.device_id,
                "model_id": model_id,
                "error": str(e),
                "latency_ms": 0,
                "mode": mode.value,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    async def _execute_real_time_inference(self, device: IoTDevice, model_id: str, 
                                         input_data: Any) -> Any:
        """Execute real-time inference with minimal latency."""
        # Simulate real-time inference
        await asyncio.sleep(0.01)  # 10ms simulation
        
        # Return mock result
        return {
            "prediction": "real_time_result",
            "confidence": 0.95,
            "processing_time_ms": 10
        }
    
    async def _execute_batch_inference(self, device: IoTDevice, model_id: str, 
                                     input_data: Any) -> Any:
        """Execute batch inference for multiple inputs."""
        # Simulate batch processing
        await asyncio.sleep(0.1)  # 100ms simulation
        
        return {
            "predictions": ["batch_result_1", "batch_result_2"],
            "batch_size": len(input_data) if hasattr(input_data, '__len__') else 1,
            "processing_time_ms": 100
        }
    
    async def _execute_streaming_inference(self, device: IoTDevice, model_id: str, 
                                         input_data: Any) -> Any:
        """Execute streaming inference for continuous data."""
        # Simulate streaming processing
        await asyncio.sleep(0.05)  # 50ms simulation
        
        return {
            "stream_id": str(uuid.uuid4()),
            "prediction": "streaming_result",
            "processing_time_ms": 50
        }
    
    async def _execute_hybrid_inference(self, device: IoTDevice, model_id: str, 
                                      input_data: Any) -> Any:
        """Execute hybrid inference combining multiple modes."""
        # Simulate hybrid processing
        await asyncio.sleep(0.03)  # 30ms simulation
        
        return {
            "prediction": "hybrid_result",
            "mode_used": "hybrid",
            "processing_time_ms": 30
        }
    
    def _update_inference_stats(self, success: bool, latency_ms: float):
        """Update inference statistics."""
        self.inference_stats["total_requests"] += 1
        
        if success:
            self.inference_stats["successful_requests"] += 1
            # Update average latency
            current_avg = self.inference_stats["average_latency_ms"]
            total_successful = self.inference_stats["successful_requests"]
            self.inference_stats["average_latency_ms"] = \
                (current_avg * (total_successful - 1) + latency_ms) / total_successful
        else:
            self.inference_stats["failed_requests"] += 1
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get current inference statistics."""
        return self.inference_stats.copy()

class IoTDeviceManager:
    """Manages IoT devices in the edge network."""
    
    def __init__(self, config: EdgeAIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.device_manager")
        self.devices = {}
        self.device_groups = {}
        self.discovery_thread = None
        self.monitoring_thread = None
        self.running = False
    
    def start_device_management(self):
        """Start device management services."""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting IoT device management services")
        
        # Start device discovery
        if self.config.enable_device_discovery:
            self.discovery_thread = threading.Thread(target=self._device_discovery_worker, daemon=True)
            self.discovery_thread.start()
        
        # Start device monitoring
        if self.config.enable_device_monitoring:
            self.monitoring_thread = threading.Thread(target=self._device_monitoring_worker, daemon=True)
            self.monitoring_thread.start()
    
    def stop_device_management(self):
        """Stop device management services."""
        self.running = False
        self.logger.info("Stopping IoT device management services")
    
    def register_device(self, device: IoTDevice) -> bool:
        """Register a new IoT device."""
        try:
            if device.device_id in self.devices:
                self.logger.warning(f"Device {device.device_id} already registered")
                return False
            
            self.devices[device.device_id] = device
            device.update_heartbeat()
            
            # Auto-provision if enabled
            if self.config.enable_auto_provisioning:
                self._auto_provision_device(device)
            
            self.logger.info(f"Device {device.device_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register device {device.device_id}: {e}")
            return False
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device."""
        if device_id in self.devices:
            del self.devices[device_id]
            self.logger.info(f"Device {device_id} unregistered")
            return True
        return False
    
    def get_device(self, device_id: str) -> Optional[IoTDevice]:
        """Get a device by ID."""
        return self.devices.get(device_id)
    
    def get_all_devices(self) -> List[IoTDevice]:
        """Get all registered devices."""
        return list(self.devices.values())
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[IoTDevice]:
        """Get devices by type."""
        return [device for device in self.devices.values() if device.device_type == device_type]
    
    def get_devices_by_status(self, status: DeviceStatus) -> List[IoTDevice]:
        """Get devices by status."""
        return [device for device in self.devices.values() if device.status == status]
    
    def _auto_provision_device(self, device: IoTDevice):
        """Automatically provision a newly registered device."""
        self.logger.info(f"Auto-provisioning device {device.device_id}")
        
        # Set default models based on device type
        if device.device_type == DeviceType.CAMERA:
            device.add_model("image_classification", {
                "name": "Image Classification Model",
                "type": "cnn",
                "version": "1.0"
            })
        elif device.device_type == DeviceType.SENSOR:
            device.add_model("sensor_analysis", {
                "name": "Sensor Data Analysis",
                "type": "regression",
                "version": "1.0"
            })
    
    def _device_discovery_worker(self):
        """Worker thread for device discovery."""
        while self.running:
            try:
                # Simulate device discovery
                time.sleep(self.config.device_heartbeat_interval)
                
                # Check for new devices (in real implementation, this would scan network)
                self.logger.debug("Performing device discovery scan")
                
            except Exception as e:
                self.logger.error(f"Device discovery error: {e}")
    
    def _device_monitoring_worker(self):
        """Worker thread for device monitoring."""
        while self.running:
            try:
                # Check device heartbeats
                current_time = datetime.now()
                for device in self.devices.values():
                    if device.last_heartbeat:
                        time_since_heartbeat = (current_time - device.last_heartbeat).total_seconds()
                        if time_since_heartbeat > self.config.device_heartbeat_interval * 2:
                            device.status = DeviceStatus.OFFLINE
                            self.logger.warning(f"Device {device.device_id} appears offline")
                
                time.sleep(self.config.device_heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Device monitoring error: {e}")

class EdgeCloudOrchestrator:
    """Orchestrates operations between edge devices and cloud services."""
    
    def __init__(self, config: EdgeAIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.orchestrator")
        self.edge_nodes = {}
        self.cloud_services = {}
        self.sync_queue = queue.Queue()
        self.orchestration_rules = {}
    
    def register_edge_node(self, node_id: str, node_info: Dict[str, Any]):
        """Register an edge node."""
        self.edge_nodes[node_id] = {
            "info": node_info,
            "status": "online",
            "last_sync": datetime.now(),
            "sync_status": "pending"
        }
        self.logger.info(f"Edge node {node_id} registered")
    
    def register_cloud_service(self, service_id: str, service_info: Dict[str, Any]):
        """Register a cloud service."""
        self.cloud_services[service_id] = {
            "info": service_info,
            "status": "online",
            "last_health_check": datetime.now()
        }
        self.logger.info(f"Cloud service {service_id} registered")
    
    def orchestrate_workload(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate workload distribution between edge and cloud."""
        self.logger.info(f"Orchestrating workload: {workload.get('id', 'unknown')}")
        
        # Determine optimal placement based on workload characteristics
        placement = self._determine_workload_placement(workload)
        
        # Execute workload placement
        result = self._execute_workload_placement(workload, placement)
        
        return {
            "workload_id": workload.get("id"),
            "placement": placement,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _determine_workload_placement(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal placement for a workload."""
        workload_type = workload.get("type", "unknown")
        latency_requirement = workload.get("latency_requirement_ms", 1000)
        data_size = workload.get("data_size_mb", 0)
        
        # Simple placement logic
        if latency_requirement < 100:  # Ultra-low latency
            placement = "edge"
        elif latency_requirement < 500:  # Low latency
            placement = "edge_or_fog"
        else:  # Standard latency
            placement = "cloud"
        
        # Adjust based on data size
        if data_size > 100:  # Large data
            placement = "cloud"  # Prefer cloud for large data
        
        return {
            "target_tier": placement,
            "reasoning": f"Latency: {latency_requirement}ms, Data: {data_size}MB",
            "estimated_latency_ms": self._estimate_latency(placement, workload)
        }
    
    def _estimate_latency(self, placement: str, workload: Dict[str, Any]) -> float:
        """Estimate latency for workload placement."""
        base_latencies = {
            "edge": 10,
            "edge_or_fog": 50,
            "fog": 100,
            "cloud": 200
        }
        
        base_latency = base_latencies.get(placement, 200)
        
        # Adjust for data size
        data_size = workload.get("data_size_mb", 0)
        if data_size > 50:
            base_latency += data_size * 0.5
        
        return base_latency
    
    def _execute_workload_placement(self, workload: Dict[str, Any], 
                                  placement: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workload placement."""
        target_tier = placement["target_tier"]
        
        if target_tier == "edge":
            return self._execute_edge_workload(workload)
        elif target_tier == "cloud":
            return self._execute_cloud_workload(workload)
        else:
            return self._execute_hybrid_workload(workload)
    
    def _execute_edge_workload(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workload on edge devices."""
        return {
            "execution_location": "edge",
            "status": "executed",
            "execution_time_ms": 50,
            "device_id": "edge_device_1"
        }
    
    def _execute_cloud_workload(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workload on cloud services."""
        return {
            "execution_location": "cloud",
            "status": "executed",
            "execution_time_ms": 200,
            "service_id": "cloud_service_1"
        }
    
    def _execute_hybrid_workload(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workload using hybrid edge-cloud approach."""
        return {
            "execution_location": "hybrid",
            "status": "executed",
            "edge_execution_time_ms": 30,
            "cloud_execution_time_ms": 150,
            "total_time_ms": 180
        }
    
    def sync_edge_cloud_data(self, data: Dict[str, Any]) -> bool:
        """Synchronize data between edge and cloud."""
        try:
            # Add to sync queue
            self.sync_queue.put({
                "data": data,
                "timestamp": datetime.now(),
                "sync_id": str(uuid.uuid4())
            })
            
            self.logger.info(f"Data queued for edge-cloud sync: {data.get('id', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue data for sync: {e}")
            return False

class AdvancedEdgeAIIoTSystem:
    """Main system for Advanced Edge AI & IoT capabilities."""
    
    def __init__(self, config: EdgeAIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.main_system")
        self.initialized = False
        
        # Initialize components
        self.edge_optimizer = EdgeComputingOptimizer(config)
        self.inference_engine = RealTimeInferenceEngine(config)
        self.device_manager = IoTDeviceManager(config)
        self.orchestrator = EdgeCloudOrchestrator(config)
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the Edge AI & IoT system."""
        try:
            # Start device management
            self.device_manager.start_device_management()
            
            # Initialize edge-cloud orchestration
            self._initialize_edge_cloud_services()
            
            self.initialized = True
            self.logger.info("Advanced Edge AI & IoT System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Edge AI & IoT system: {e}")
            raise
    
    def _initialize_edge_cloud_services(self):
        """Initialize edge-cloud orchestration services."""
        # Register default cloud services
        self.orchestrator.register_cloud_service("ml_training", {
            "name": "ML Training Service",
            "type": "training",
            "capabilities": ["model_training", "hyperparameter_optimization"]
        })
        
        self.orchestrator.register_cloud_service("data_storage", {
            "name": "Data Storage Service",
            "type": "storage",
            "capabilities": ["data_ingestion", "data_processing", "data_analytics"]
        })
    
    def register_device(self, device: IoTDevice) -> bool:
        """Register a new IoT device."""
        if not self.initialized:
            raise RuntimeError("Edge AI & IoT system not initialized")
        
        return self.device_manager.register_device(device)
    
    def execute_inference(self, device_id: str, model_id: str, 
                         input_data: Any, mode: InferenceMode = InferenceMode.REAL_TIME) -> Dict[str, Any]:
        """Execute inference on a device."""
        if not self.initialized:
            raise RuntimeError("Edge AI & IoT system not initialized")
        
        device = self.device_manager.get_device(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        # Execute inference asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.inference_engine.execute_inference(device, model_id, input_data, mode)
            )
            return result
        finally:
            loop.close()
    
    def optimize_edge_deployment(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize model deployment across edge devices."""
        if not self.initialized:
            raise RuntimeError("Edge AI & IoT system not initialized")
        
        devices = self.device_manager.get_all_devices()
        return self.edge_optimizer.optimize_edge_deployment(devices, models)
    
    def orchestrate_workload(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate workload between edge and cloud."""
        if not self.initialized:
            raise RuntimeError("Edge AI & IoT system not initialized")
        
        return self.orchestrator.orchestrate_workload(workload)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "initialized": self.initialized,
            "total_devices": len(self.device_manager.devices),
            "online_devices": len(self.device_manager.get_devices_by_status(DeviceStatus.ONLINE)),
            "edge_nodes": len(self.orchestrator.edge_nodes),
            "cloud_services": len(self.orchestrator.cloud_services),
            "inference_stats": self.inference_engine.get_inference_stats(),
            "config": {
                "enable_edge_computing": self.config.enable_edge_computing,
                "enable_iot_management": self.config.enable_iot_management,
                "enable_real_time_inference": self.config.enable_real_time_inference,
                "enable_edge_cloud_orchestration": self.config.enable_edge_cloud_orchestration
            }
        }
    
    def shutdown(self):
        """Shutdown the system gracefully."""
        if self.initialized:
            self.device_manager.stop_device_management()
            self.initialized = False
            self.logger.info("Advanced Edge AI & IoT System shutdown completed")

# Factory functions
def create_edge_ai_config(
    enable_edge_computing: bool = True,
    enable_iot_management: bool = True,
    enable_real_time_inference: bool = True,
    enable_edge_cloud_orchestration: bool = True
) -> EdgeAIConfig:
    """Create Edge AI configuration."""
    return EdgeAIConfig(
        enable_edge_computing=enable_edge_computing,
        enable_iot_management=enable_iot_management,
        enable_real_time_inference=enable_real_time_inference,
        enable_edge_cloud_orchestration=enable_edge_cloud_orchestration
    )

def create_advanced_edge_ai_iot_system(config: Optional[EdgeAIConfig] = None) -> AdvancedEdgeAIIoTSystem:
    """Create Advanced Edge AI & IoT System instance."""
    if config is None:
        config = create_edge_ai_config()
    return AdvancedEdgeAIIoTSystem(config)

def create_minimal_edge_ai_config() -> EdgeAIConfig:
    """Create minimal Edge AI configuration."""
    return EdgeAIConfig(
        enable_edge_computing=True,
        enable_iot_management=True,
        enable_real_time_inference=False,
        enable_edge_cloud_orchestration=False
    )

def create_maximum_edge_ai_config() -> EdgeAIConfig:
    """Create maximum Edge AI configuration."""
    return EdgeAIConfig(
        enable_edge_computing=True,
        enable_iot_management=True,
        enable_real_time_inference=True,
        enable_edge_cloud_orchestration=True,
        enable_edge_optimization=True,
        enable_edge_caching=True,
        enable_device_discovery=True,
        enable_auto_provisioning=True,
        enable_device_monitoring=True,
        enable_model_compression=True,
        enable_quantization=True,
        enable_pruning=True,
        enable_load_balancing=True,
        enable_failover=True,
        enable_data_sync=True,
        enable_edge_security=True,
        enable_device_authentication=True,
        enable_data_encryption=True
    )
