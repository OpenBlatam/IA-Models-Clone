"""
Advanced Edge Computing and IoT Integration for Microservices
Features: Edge device management, IoT data processing, edge AI inference, fog computing, edge orchestration
"""

import asyncio
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
from abc import ABC, abstractmethod
import numpy as np

# Edge computing imports
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    import asyncio_mqtt
    ASYNCIO_MQTT_AVAILABLE = True
except ImportError:
    ASYNCIO_MQTT_AVAILABLE = False

try:
    import redis
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import tensorflow as tf
    import tensorflow_lite as tflite
    TENSORFLOW_LITE_AVAILABLE = True
except ImportError:
    TENSORFLOW_LITE_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """IoT device types"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    MICROCONTROLLER = "microcontroller"
    EMBEDDED_SYSTEM = "embedded_system"
    EDGE_SERVER = "edge_server"

class DataType(Enum):
    """Data types for edge processing"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"

class ProcessingMode(Enum):
    """Edge processing modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"

class DeviceStatus(Enum):
    """Device status"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SLEEPING = "sleeping"

@dataclass
class EdgeDevice:
    """Edge device definition"""
    device_id: str
    device_type: DeviceType
    name: str
    location: Dict[str, float]  # lat, lon, alt
    capabilities: List[str]
    status: DeviceStatus = DeviceStatus.OFFLINE
    last_seen: float = 0.0
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeData:
    """Edge data structure"""
    data_id: str
    device_id: str
    data_type: DataType
    payload: Any
    timestamp: float
    location: Optional[Dict[str, float]] = None
    quality_score: float = 1.0
    processing_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeTask:
    """Edge processing task"""
    task_id: str
    device_id: str
    task_type: str
    priority: int = 1
    deadline: Optional[float] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    data: Any = None
    result: Any = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)

class EdgeDeviceManager:
    """
    Edge device management system
    """
    
    def __init__(self):
        self.devices: Dict[str, EdgeDevice] = {}
        self.device_groups: Dict[str, List[str]] = defaultdict(list)
        self.device_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.heartbeat_interval = 30.0
        self.offline_threshold = 120.0
    
    def register_device(self, device: EdgeDevice) -> bool:
        """Register edge device"""
        try:
            self.devices[device.device_id] = device
            device.status = DeviceStatus.ONLINE
            device.last_seen = time.time()
            
            # Add to default group
            self.device_groups["all"].append(device.device_id)
            
            logger.info(f"Registered edge device: {device.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            return False
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister edge device"""
        try:
            if device_id in self.devices:
                del self.devices[device_id]
                
                # Remove from all groups
                for group_devices in self.device_groups.values():
                    if device_id in group_devices:
                        group_devices.remove(device_id)
                
                logger.info(f"Unregistered edge device: {device_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Device unregistration failed: {e}")
            return False
    
    def update_device_status(self, device_id: str, status: DeviceStatus, metrics: Dict[str, Any] = None):
        """Update device status and metrics"""
        try:
            if device_id in self.devices:
                device = self.devices[device_id]
                device.status = status
                device.last_seen = time.time()
                
                if metrics:
                    device.battery_level = metrics.get("battery_level")
                    device.signal_strength = metrics.get("signal_strength")
                    device.metadata.update(metrics)
                
                # Store metrics
                self.device_metrics[device_id].append({
                    "timestamp": time.time(),
                    "status": status.value,
                    "metrics": metrics or {}
                })
                
        except Exception as e:
            logger.error(f"Device status update failed: {e}")
    
    def get_online_devices(self) -> List[EdgeDevice]:
        """Get all online devices"""
        current_time = time.time()
        online_devices = []
        
        for device in self.devices.values():
            if (device.status == DeviceStatus.ONLINE and 
                current_time - device.last_seen < self.offline_threshold):
                online_devices.append(device)
        
        return online_devices
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[EdgeDevice]:
        """Get devices by type"""
        return [device for device in self.devices.values() if device.device_type == device_type]
    
    def get_devices_by_location(self, center: Dict[str, float], radius: float) -> List[EdgeDevice]:
        """Get devices within radius of center location"""
        devices_in_range = []
        
        for device in self.devices.values():
            if device.location:
                distance = self._calculate_distance(center, device.location)
                if distance <= radius:
                    devices_in_range.append(device)
        
        return devices_in_range
    
    def _calculate_distance(self, point1: Dict[str, float], point2: Dict[str, float]) -> float:
        """Calculate distance between two points"""
        import math
        
        lat1, lon1 = point1["lat"], point1["lon"]
        lat2, lon2 = point2["lat"], point2["lon"]
        
        # Haversine formula
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def get_device_stats(self) -> Dict[str, Any]:
        """Get device statistics"""
        total_devices = len(self.devices)
        online_devices = len(self.get_online_devices())
        
        device_types = defaultdict(int)
        for device in self.devices.values():
            device_types[device.device_type.value] += 1
        
        return {
            "total_devices": total_devices,
            "online_devices": online_devices,
            "offline_devices": total_devices - online_devices,
            "device_types": dict(device_types),
            "device_groups": {group: len(devices) for group, devices in self.device_groups.items()}
        }

class EdgeDataProcessor:
    """
    Edge data processing system
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.data_queue: asyncio.Queue = asyncio.Queue()
        self.processors: Dict[DataType, Callable] = {}
        self.processing_active = False
        self.worker_tasks: List[asyncio.Task] = []
        self.processed_data: deque = deque(maxlen=10000)
        
        # Initialize default processors
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize data processors"""
        self.processors[DataType.TEMPERATURE] = self._process_temperature
        self.processors[DataType.HUMIDITY] = self._process_humidity
        self.processors[DataType.PRESSURE] = self._process_pressure
        self.processors[DataType.IMAGE] = self._process_image
        self.processors[DataType.VIDEO] = self._process_video
        self.processors[DataType.AUDIO] = self._process_audio
        self.processors[DataType.JSON] = self._process_json
    
    async def start_processing(self, num_workers: int = 4):
        """Start data processing workers"""
        if self.processing_active:
            return
        
        self.processing_active = True
        
        # Start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._data_worker(f"worker-{i}"))
            self.worker_tasks.append(worker)
        
        logger.info(f"Started edge data processing with {num_workers} workers")
    
    async def stop_processing(self):
        """Stop data processing"""
        self.processing_active = False
        
        # Cancel worker tasks
        for worker in self.worker_tasks:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("Stopped edge data processing")
    
    async def process_data(self, data: EdgeData) -> EdgeData:
        """Process edge data"""
        try:
            # Get processor for data type
            processor = self.processors.get(data.data_type)
            if not processor:
                logger.warning(f"No processor for data type: {data.data_type}")
                return data
            
            # Process data
            processed_payload = await processor(data.payload)
            
            # Create processed data
            processed_data = EdgeData(
                data_id=f"processed_{data.data_id}",
                device_id=data.device_id,
                data_type=data.data_type,
                payload=processed_payload,
                timestamp=time.time(),
                location=data.location,
                quality_score=data.quality_score,
                processing_required=False,
                metadata={
                    **data.metadata,
                    "original_data_id": data.data_id,
                    "processing_timestamp": time.time()
                }
            )
            
            # Store processed data
            self.processed_data.append(processed_data)
            
            # Store in Redis if available
            if self.redis:
                await self._store_processed_data(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return data
    
    async def _data_worker(self, worker_id: str):
        """Data processing worker"""
        while self.processing_active:
            try:
                # Get data from queue
                data = await asyncio.wait_for(self.data_queue.get(), timeout=1.0)
                
                # Process data
                processed_data = await self.process_data(data)
                
                logger.debug(f"Worker {worker_id} processed data: {data.data_id}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def _store_processed_data(self, data: EdgeData):
        """Store processed data in Redis"""
        try:
            data_key = f"edge_data:{data.data_id}"
            data_value = {
                "device_id": data.device_id,
                "data_type": data.data_type.value,
                "payload": json.dumps(data.payload, default=str),
                "timestamp": data.timestamp,
                "quality_score": data.quality_score,
                "metadata": json.dumps(data.metadata)
            }
            
            await self.redis.hset(data_key, mapping=data_value)
            await self.redis.expire(data_key, 86400)  # 24 hours TTL
            
        except Exception as e:
            logger.error(f"Failed to store processed data: {e}")
    
    # Data processors
    async def _process_temperature(self, payload: Any) -> Any:
        """Process temperature data"""
        try:
            temp = float(payload)
            # Apply calibration, filtering, etc.
            processed_temp = temp + 0.1  # Example calibration
            return {
                "value": processed_temp,
                "unit": "celsius",
                "quality": "good" if 0 <= processed_temp <= 100 else "warning"
            }
        except Exception as e:
            logger.error(f"Temperature processing failed: {e}")
            return payload
    
    async def _process_humidity(self, payload: Any) -> Any:
        """Process humidity data"""
        try:
            humidity = float(payload)
            return {
                "value": humidity,
                "unit": "percent",
                "quality": "good" if 0 <= humidity <= 100 else "warning"
            }
        except Exception as e:
            logger.error(f"Humidity processing failed: {e}")
            return payload
    
    async def _process_pressure(self, payload: Any) -> Any:
        """Process pressure data"""
        try:
            pressure = float(payload)
            return {
                "value": pressure,
                "unit": "hPa",
                "quality": "good" if 800 <= pressure <= 1200 else "warning"
            }
        except Exception as e:
            logger.error(f"Pressure processing failed: {e}")
            return payload
    
    async def _process_image(self, payload: Any) -> Any:
        """Process image data"""
        try:
            # This would implement actual image processing
            # For demo, just return metadata
            return {
                "size": len(str(payload)),
                "format": "jpeg",
                "processed": True,
                "features_extracted": True
            }
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return payload
    
    async def _process_video(self, payload: Any) -> Any:
        """Process video data"""
        try:
            return {
                "size": len(str(payload)),
                "format": "mp4",
                "processed": True,
                "frames_analyzed": 30
            }
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return payload
    
    async def _process_audio(self, payload: Any) -> Any:
        """Process audio data"""
        try:
            return {
                "size": len(str(payload)),
                "format": "wav",
                "processed": True,
                "transcription": "sample text"
            }
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return payload
    
    async def _process_json(self, payload: Any) -> Any:
        """Process JSON data"""
        try:
            if isinstance(payload, str):
                data = json.loads(payload)
            else:
                data = payload
            
            # Add processing metadata
            data["processed_at"] = time.time()
            data["processing_node"] = "edge"
            
            return data
        except Exception as e:
            logger.error(f"JSON processing failed: {e}")
            return payload

class EdgeAIInference:
    """
    Edge AI inference system
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.inference_cache: Dict[str, Any] = {}
        self.model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def load_model(self, model_id: str, model_path: str, model_type: str = "tflite") -> bool:
        """Load AI model for edge inference"""
        try:
            if model_type == "tflite" and TENSORFLOW_LITE_AVAILABLE:
                interpreter = tflite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                self.models[model_id] = {
                    "interpreter": interpreter,
                    "type": "tflite",
                    "path": model_path
                }
            
            elif model_type == "onnx" and ONNX_AVAILABLE:
                session = ort.InferenceSession(model_path)
                self.models[model_id] = {
                    "session": session,
                    "type": "onnx",
                    "path": model_path
                }
            
            else:
                logger.warning(f"Model type {model_type} not supported or library not available")
                return False
            
            logger.info(f"Loaded edge AI model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    async def run_inference(self, model_id: str, input_data: Any) -> Any:
        """Run AI inference on edge device"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            start_time = time.time()
            
            if model["type"] == "tflite":
                result = await self._run_tflite_inference(model["interpreter"], input_data)
            elif model["type"] == "onnx":
                result = await self._run_onnx_inference(model["session"], input_data)
            else:
                raise ValueError(f"Unsupported model type: {model['type']}")
            
            inference_time = time.time() - start_time
            
            # Store metrics
            self.model_metrics[model_id].append({
                "timestamp": time.time(),
                "inference_time": inference_time,
                "input_size": len(str(input_data)),
                "success": True
            })
            
            # Cache result
            cache_key = f"{model_id}:{hash(str(input_data))}"
            self.inference_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            
            # Store error metrics
            self.model_metrics[model_id].append({
                "timestamp": time.time(),
                "inference_time": 0.0,
                "error": str(e),
                "success": False
            })
            
            raise
    
    async def _run_tflite_inference(self, interpreter: Any, input_data: Any) -> Any:
        """Run TensorFlow Lite inference"""
        try:
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare input data
            input_shape = input_details[0]['shape']
            input_data_processed = np.array(input_data, dtype=np.float32).reshape(input_shape)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data_processed)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            return output_data.tolist()
            
        except Exception as e:
            logger.error(f"TensorFlow Lite inference failed: {e}")
            raise
    
    async def _run_onnx_inference(self, session: Any, input_data: Any) -> Any:
        """Run ONNX inference"""
        try:
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Prepare input data
            input_data_processed = np.array(input_data, dtype=np.float32)
            
            # Run inference
            result = session.run(None, {input_name: input_data_processed})
            
            return result[0].tolist()
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get AI model statistics"""
        stats = {}
        
        for model_id, metrics in self.model_metrics.items():
            if metrics:
                recent_metrics = list(metrics)[-10:]
                success_rate = sum(1 for m in recent_metrics if m.get("success", False)) / len(recent_metrics)
                avg_inference_time = statistics.mean([m.get("inference_time", 0) for m in recent_metrics])
                
                stats[model_id] = {
                    "success_rate": success_rate,
                    "avg_inference_time": avg_inference_time,
                    "total_inferences": len(metrics),
                    "model_type": self.models[model_id]["type"] if model_id in self.models else "unknown"
                }
        
        return stats

class EdgeOrchestrator:
    """
    Edge computing orchestration system
    """
    
    def __init__(self, device_manager: EdgeDeviceManager, data_processor: EdgeDataProcessor, ai_inference: EdgeAIInference):
        self.device_manager = device_manager
        self.data_processor = data_processor
        self.ai_inference = ai_inference
        self.tasks: Dict[str, EdgeTask] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.orchestration_active = False
        self.worker_tasks: List[asyncio.Task] = []
        self.resource_monitor: Dict[str, Dict[str, float]] = {}
    
    async def start_orchestration(self, num_workers: int = 2):
        """Start edge orchestration"""
        if self.orchestration_active:
            return
        
        self.orchestration_active = True
        
        # Start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._orchestration_worker(f"worker-{i}"))
            self.worker_tasks.append(worker)
        
        logger.info(f"Started edge orchestration with {num_workers} workers")
    
    async def stop_orchestration(self):
        """Stop edge orchestration"""
        self.orchestration_active = False
        
        # Cancel worker tasks
        for worker in self.worker_tasks:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("Stopped edge orchestration")
    
    async def submit_task(self, task: EdgeTask) -> str:
        """Submit edge task for execution"""
        try:
            # Add to task queue with priority
            await self.task_queue.put((task.priority, task.task_id, task))
            self.tasks[task.task_id] = task
            
            logger.info(f"Submitted edge task: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            raise
    
    async def _orchestration_worker(self, worker_id: str):
        """Edge orchestration worker"""
        while self.orchestration_active:
            try:
                # Get task from queue
                priority, task_id, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Execute task
                await self._execute_task(task)
                
                logger.debug(f"Worker {worker_id} executed task: {task_id}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def _execute_task(self, task: EdgeTask):
        """Execute edge task"""
        try:
            task.status = "running"
            
            # Check if device is available
            if task.device_id not in self.device_manager.devices:
                task.status = "failed"
                task.result = {"error": "Device not found"}
                return
            
            device = self.device_manager.devices[task.device_id]
            if device.status != DeviceStatus.ONLINE:
                task.status = "failed"
                task.result = {"error": "Device offline"}
                return
            
            # Execute based on task type
            if task.task_type == "data_processing":
                result = await self._execute_data_processing_task(task)
            elif task.task_type == "ai_inference":
                result = await self._execute_ai_inference_task(task)
            elif task.task_type == "device_control":
                result = await self._execute_device_control_task(task)
            else:
                result = {"error": f"Unknown task type: {task.task_type}"}
            
            task.result = result
            task.status = "completed"
            
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            logger.error(f"Task execution failed: {e}")
    
    async def _execute_data_processing_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute data processing task"""
        try:
            # Create edge data from task
            edge_data = EdgeData(
                data_id=task.task_id,
                device_id=task.device_id,
                data_type=DataType.JSON,  # Default type
                payload=task.data,
                timestamp=time.time()
            )
            
            # Process data
            processed_data = await self.data_processor.process_data(edge_data)
            
            return {
                "processed_data": processed_data.payload,
                "processing_time": time.time() - task.created_at,
                "quality_score": processed_data.quality_score
            }
            
        except Exception as e:
            logger.error(f"Data processing task failed: {e}")
            return {"error": str(e)}
    
    async def _execute_ai_inference_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute AI inference task"""
        try:
            model_id = task.metadata.get("model_id", "default_model")
            
            # Run inference
            result = await self.ai_inference.run_inference(model_id, task.data)
            
            return {
                "inference_result": result,
                "model_id": model_id,
                "inference_time": time.time() - task.created_at
            }
            
        except Exception as e:
            logger.error(f"AI inference task failed: {e}")
            return {"error": str(e)}
    
    async def _execute_device_control_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Execute device control task"""
        try:
            # This would implement actual device control
            # For demo, just return success
            return {
                "control_command": task.data,
                "device_id": task.device_id,
                "execution_time": time.time() - task.created_at,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Device control task failed: {e}")
            return {"error": str(e)}
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        failed_tasks = len([t for t in self.tasks.values() if t.status == "failed"])
        running_tasks = len([t for t in self.tasks.values() if t.status == "running"])
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "queue_size": self.task_queue.qsize(),
            "orchestration_active": self.orchestration_active,
            "device_stats": self.device_manager.get_device_stats(),
            "ai_model_stats": self.ai_inference.get_model_stats()
        }

class EdgeComputingManager:
    """
    Main edge computing management system
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.device_manager = EdgeDeviceManager()
        self.data_processor = EdgeDataProcessor(redis_client)
        self.ai_inference = EdgeAIInference()
        self.orchestrator = EdgeOrchestrator(
            self.device_manager, 
            self.data_processor, 
            self.ai_inference
        )
        self.edge_active = False
    
    async def start_edge_computing(self):
        """Start edge computing system"""
        if self.edge_active:
            return
        
        try:
            # Start data processing
            await self.data_processor.start_processing()
            
            # Start orchestration
            await self.orchestrator.start_orchestration()
            
            self.edge_active = True
            logger.info("Edge computing system started")
            
        except Exception as e:
            logger.error(f"Failed to start edge computing: {e}")
            raise
    
    async def stop_edge_computing(self):
        """Stop edge computing system"""
        if not self.edge_active:
            return
        
        try:
            # Stop orchestration
            await self.orchestrator.stop_orchestration()
            
            # Stop data processing
            await self.data_processor.stop_processing()
            
            self.edge_active = False
            logger.info("Edge computing system stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop edge computing: {e}")
    
    def get_edge_stats(self) -> Dict[str, Any]:
        """Get edge computing statistics"""
        return {
            "edge_active": self.edge_active,
            "device_stats": self.device_manager.get_device_stats(),
            "orchestration_stats": self.orchestrator.get_orchestration_stats(),
            "ai_model_stats": self.ai_inference.get_model_stats(),
            "processed_data_count": len(self.data_processor.processed_data)
        }

# Global edge computing manager
edge_computing_manager: Optional[EdgeComputingManager] = None

def initialize_edge_computing(redis_client: Optional[aioredis.Redis] = None):
    """Initialize edge computing manager"""
    global edge_computing_manager
    
    edge_computing_manager = EdgeComputingManager(redis_client)
    logger.info("Edge computing manager initialized")

# Decorator for edge computing operations
def edge_operation(device_type: DeviceType = None):
    """Decorator for edge computing operations"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            if not edge_computing_manager:
                initialize_edge_computing()
            
            return await func(*args, **kwargs)
        
        return async_wrapper
    return decorator

# Initialize edge computing on import
initialize_edge_computing()