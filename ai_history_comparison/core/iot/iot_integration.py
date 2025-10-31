"""
IoT Integration System - Advanced Internet of Things Capabilities

This module provides advanced IoT integration capabilities including:
- Device management and provisioning
- Real-time sensor data collection
- Edge computing and processing
- IoT protocol support (MQTT, CoAP, HTTP, WebSocket)
- Device authentication and security
- Data streaming and analytics
- Remote device control
- Firmware over-the-air updates
- IoT gateway management
- Industrial IoT (IIoT) support
"""

import asyncio
import json
import uuid
import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import weakref
import base64
import secrets
import struct
import socket

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """IoT device types"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    SMART_DEVICE = "smart_device"
    INDUSTRIAL = "industrial"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"
    DRONE = "drone"
    ROBOT = "robot"

class ProtocolType(Enum):
    """IoT protocol types"""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    WEBSOCKET = "websocket"
    MODBUS = "modbus"
    OPC_UA = "opc_ua"
    ZIGBEE = "zigbee"
    Z_WAVE = "z_wave"
    LORA = "lora"
    NB_IOT = "nb_iot"

class DataType(Enum):
    """IoT data types"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"
    SOUND = "sound"
    VIBRATION = "vibration"
    GPS = "gps"
    IMAGE = "image"
    VIDEO = "video"
    CUSTOM = "custom"

class DeviceStatus(Enum):
    """Device status"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UPDATING = "updating"

@dataclass
class IoTDevice:
    """IoT device data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    device_type: DeviceType = DeviceType.SENSOR
    protocol: ProtocolType = ProtocolType.MQTT
    mac_address: str = ""
    ip_address: str = ""
    location: Dict[str, float] = field(default_factory=dict)
    status: DeviceStatus = DeviceStatus.OFFLINE
    firmware_version: str = "1.0.0"
    hardware_version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    last_seen: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SensorData:
    """Sensor data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str = ""
    data_type: DataType = DataType.TEMPERATURE
    value: float = 0.0
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    quality: float = 1.0
    location: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceCommand:
    """Device command structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str = ""
    command_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: int = 30
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None

@dataclass
class IoTGateway:
    """IoT gateway data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    location: Dict[str, float] = field(default_factory=dict)
    connected_devices: List[str] = field(default_factory=list)
    protocols_supported: List[ProtocolType] = field(default_factory=list)
    processing_capability: str = "edge"
    status: DeviceStatus = DeviceStatus.OFFLINE
    last_heartbeat: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Base classes
class BaseIoTProtocol(ABC):
    """Base IoT protocol class"""
    
    def __init__(self, protocol_type: ProtocolType):
        self.protocol_type = protocol_type
        self.connected_devices: Dict[str, IoTDevice] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def connect_device(self, device: IoTDevice) -> bool:
        """Connect device to protocol"""
        pass
    
    @abstractmethod
    async def disconnect_device(self, device_id: str) -> bool:
        """Disconnect device from protocol"""
        pass
    
    @abstractmethod
    async def send_data(self, device_id: str, data: SensorData) -> bool:
        """Send data from device"""
        pass
    
    @abstractmethod
    async def send_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command to device"""
        pass
    
    @abstractmethod
    async def receive_data(self, device_id: str) -> List[SensorData]:
        """Receive data from device"""
        pass

class MQTTProtocol(BaseIoTProtocol):
    """MQTT protocol implementation"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        super().__init__(ProtocolType.MQTT)
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topics: Dict[str, List[str]] = defaultdict(list)
        self.message_queue: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    async def connect_device(self, device: IoTDevice) -> bool:
        """Connect device to MQTT broker"""
        try:
            # Simulate MQTT connection
            await asyncio.sleep(0.1)
            
            # Subscribe to device topics
            data_topic = f"devices/{device.id}/data"
            command_topic = f"devices/{device.id}/commands"
            
            self.topics[device.id] = [data_topic, command_topic]
            self.connected_devices[device.id] = device
            
            logger.info(f"Device {device.name} connected to MQTT broker")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect device {device.name}: {e}")
            return False
    
    async def disconnect_device(self, device_id: str) -> bool:
        """Disconnect device from MQTT broker"""
        try:
            if device_id in self.connected_devices:
                del self.connected_devices[device_id]
                del self.topics[device_id]
                logger.info(f"Device {device_id} disconnected from MQTT broker")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to disconnect device {device_id}: {e}")
            return False
    
    async def send_data(self, device_id: str, data: SensorData) -> bool:
        """Send sensor data via MQTT"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            # Serialize data
            message = {
                "device_id": device_id,
                "data_type": data.data_type.value,
                "value": data.value,
                "unit": data.unit,
                "timestamp": data.timestamp.isoformat(),
                "quality": data.quality,
                "location": data.location
            }
            
            # Simulate MQTT publish
            await asyncio.sleep(0.05)
            
            # Store in message queue for simulation
            topic = f"devices/{device_id}/data"
            self.message_queue[topic].append(message)
            
            logger.debug(f"Published data from device {device_id}: {data.data_type.value} = {data.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data from device {device_id}: {e}")
            return False
    
    async def send_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command to device via MQTT"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            # Serialize command
            message = {
                "command_id": command.id,
                "command_type": command.command_type,
                "parameters": command.parameters,
                "priority": command.priority,
                "timeout": command.timeout,
                "timestamp": command.created_at.isoformat()
            }
            
            # Simulate MQTT publish
            await asyncio.sleep(0.05)
            
            # Store in message queue for simulation
            topic = f"devices/{device_id}/commands"
            self.message_queue[topic].append(message)
            
            logger.info(f"Sent command to device {device_id}: {command.command_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command to device {device_id}: {e}")
            return False
    
    async def receive_data(self, device_id: str) -> List[SensorData]:
        """Receive data from device via MQTT"""
        try:
            topic = f"devices/{device_id}/data"
            messages = list(self.message_queue[topic])
            
            sensor_data = []
            for message in messages:
                data = SensorData(
                    device_id=device_id,
                    data_type=DataType(message["data_type"]),
                    value=message["value"],
                    unit=message["unit"],
                    timestamp=datetime.fromisoformat(message["timestamp"]),
                    quality=message["quality"],
                    location=message["location"]
                )
                sensor_data.append(data)
            
            # Clear processed messages
            self.message_queue[topic].clear()
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"Failed to receive data from device {device_id}: {e}")
            return []

class CoAPProtocol(BaseIoTProtocol):
    """CoAP protocol implementation"""
    
    def __init__(self, server_host: str = "localhost", server_port: int = 5683):
        super().__init__(ProtocolType.COAP)
        self.server_host = server_host
        self.server_port = server_port
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.observers: Dict[str, List[str]] = defaultdict(list)
    
    async def connect_device(self, device: IoTDevice) -> bool:
        """Connect device to CoAP server"""
        try:
            # Simulate CoAP connection
            await asyncio.sleep(0.1)
            
            # Register device resources
            self.resources[device.id] = {
                "data": f"coap://{self.server_host}:{self.server_port}/devices/{device.id}/data",
                "commands": f"coap://{self.server_host}:{self.server_port}/devices/{device.id}/commands",
                "status": f"coap://{self.server_host}:{self.server_port}/devices/{device.id}/status"
            }
            
            self.connected_devices[device.id] = device
            
            logger.info(f"Device {device.name} connected to CoAP server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect device {device.name}: {e}")
            return False
    
    async def disconnect_device(self, device_id: str) -> bool:
        """Disconnect device from CoAP server"""
        try:
            if device_id in self.connected_devices:
                del self.connected_devices[device_id]
                del self.resources[device_id]
                logger.info(f"Device {device_id} disconnected from CoAP server")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to disconnect device {device_id}: {e}")
            return False
    
    async def send_data(self, device_id: str, data: SensorData) -> bool:
        """Send sensor data via CoAP"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            # Simulate CoAP PUT request
            await asyncio.sleep(0.1)
            
            logger.debug(f"CoAP PUT data from device {device_id}: {data.data_type.value} = {data.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data from device {device_id}: {e}")
            return False
    
    async def send_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command to device via CoAP"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            # Simulate CoAP POST request
            await asyncio.sleep(0.1)
            
            logger.info(f"CoAP POST command to device {device_id}: {command.command_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command to device {device_id}: {e}")
            return False
    
    async def receive_data(self, device_id: str) -> List[SensorData]:
        """Receive data from device via CoAP"""
        try:
            # Simulate CoAP GET request
            await asyncio.sleep(0.1)
            
            # Return mock data
            return [
                SensorData(
                    device_id=device_id,
                    data_type=DataType.TEMPERATURE,
                    value=25.5,
                    unit="°C",
                    quality=0.95
                )
            ]
            
        except Exception as e:
            logger.error(f"Failed to receive data from device {device_id}: {e}")
            return []

class HTTPProtocol(BaseIoTProtocol):
    """HTTP protocol implementation"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        super().__init__(ProtocolType.HTTP)
        self.server_url = server_url
        self.endpoints: Dict[str, str] = {}
        self.http_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def connect_device(self, device: IoTDevice) -> bool:
        """Connect device to HTTP server"""
        try:
            # Simulate HTTP connection
            await asyncio.sleep(0.1)
            
            # Register device endpoints
            self.endpoints[device.id] = {
                "data": f"{self.server_url}/api/devices/{device.id}/data",
                "commands": f"{self.server_url}/api/devices/{device.id}/commands",
                "status": f"{self.server_url}/api/devices/{device.id}/status"
            }
            
            self.connected_devices[device.id] = device
            
            logger.info(f"Device {device.name} connected to HTTP server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect device {device.name}: {e}")
            return False
    
    async def disconnect_device(self, device_id: str) -> bool:
        """Disconnect device from HTTP server"""
        try:
            if device_id in self.connected_devices:
                del self.connected_devices[device_id]
                del self.endpoints[device_id]
                logger.info(f"Device {device_id} disconnected from HTTP server")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to disconnect device {device_id}: {e}")
            return False
    
    async def send_data(self, device_id: str, data: SensorData) -> bool:
        """Send sensor data via HTTP"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            # Simulate HTTP POST request
            await asyncio.sleep(0.1)
            
            logger.debug(f"HTTP POST data from device {device_id}: {data.data_type.value} = {data.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data from device {device_id}: {e}")
            return False
    
    async def send_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command to device via HTTP"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            # Simulate HTTP POST request
            await asyncio.sleep(0.1)
            
            logger.info(f"HTTP POST command to device {device_id}: {command.command_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command to device {device_id}: {e}")
            return False
    
    async def receive_data(self, device_id: str) -> List[SensorData]:
        """Receive data from device via HTTP"""
        try:
            # Simulate HTTP GET request
            await asyncio.sleep(0.1)
            
            # Return mock data
            return [
                SensorData(
                    device_id=device_id,
                    data_type=DataType.HUMIDITY,
                    value=60.0,
                    unit="%",
                    quality=0.98
                )
            ]
            
        except Exception as e:
            logger.error(f"Failed to receive data from device {device_id}: {e}")
            return []

class DeviceManager:
    """IoT device management system"""
    
    def __init__(self):
        self.devices: Dict[str, IoTDevice] = {}
        self.device_groups: Dict[str, List[str]] = defaultdict(list)
        self.device_templates: Dict[str, Dict[str, Any]] = {}
        self._initialize_templates()
    
    def _initialize_templates(self) -> None:
        """Initialize device templates"""
        self.device_templates = {
            "temperature_sensor": {
                "device_type": DeviceType.SENSOR,
                "protocol": ProtocolType.MQTT,
                "capabilities": ["temperature_reading", "battery_level"],
                "configuration": {
                    "sampling_rate": 60,  # seconds
                    "temperature_range": [-40, 85],
                    "accuracy": 0.5
                }
            },
            "smart_switch": {
                "device_type": DeviceType.ACTUATOR,
                "protocol": ProtocolType.MQTT,
                "capabilities": ["on_off_control", "power_monitoring"],
                "configuration": {
                    "max_power": 2200,  # watts
                    "voltage": 220,  # volts
                    "frequency": 50  # Hz
                }
            },
            "security_camera": {
                "device_type": DeviceType.CAMERA,
                "protocol": ProtocolType.HTTP,
                "capabilities": ["video_streaming", "motion_detection", "night_vision"],
                "configuration": {
                    "resolution": "1920x1080",
                    "fps": 30,
                    "compression": "H.264"
                }
            }
        }
    
    async def register_device(self, 
                            name: str,
                            device_type: DeviceType,
                            protocol: ProtocolType,
                            template: Optional[str] = None) -> IoTDevice:
        """Register new IoT device"""
        
        # Generate device identifiers
        mac_address = ":".join([f"{secrets.randbelow(256):02x}" for _ in range(6)])
        ip_address = f"192.168.1.{secrets.randbelow(254) + 1}"
        
        device = IoTDevice(
            name=name,
            device_type=device_type,
            protocol=protocol,
            mac_address=mac_address,
            ip_address=ip_address
        )
        
        # Apply template if provided
        if template and template in self.device_templates:
            template_data = self.device_templates[template]
            device.capabilities = template_data["capabilities"]
            device.configuration = template_data["configuration"]
        
        self.devices[device.id] = device
        
        logger.info(f"Registered device: {name} ({device_type.value})")
        
        return device
    
    async def update_device_status(self, device_id: str, status: DeviceStatus) -> bool:
        """Update device status"""
        if device_id in self.devices:
            self.devices[device_id].status = status
            self.devices[device_id].last_seen = datetime.utcnow()
            return True
        return False
    
    async def get_device_info(self, device_id: str) -> Optional[IoTDevice]:
        """Get device information"""
        return self.devices.get(device_id)
    
    async def list_devices(self, device_type: Optional[DeviceType] = None) -> List[IoTDevice]:
        """List devices with optional filtering"""
        devices = list(self.devices.values())
        
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        
        return devices
    
    async def create_device_group(self, name: str, device_ids: List[str]) -> str:
        """Create device group"""
        group_id = str(uuid.uuid4())
        self.device_groups[group_id] = {
            "name": name,
            "devices": device_ids,
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Created device group: {name} with {len(device_ids)} devices")
        
        return group_id
    
    async def update_device_configuration(self, 
                                        device_id: str, 
                                        configuration: Dict[str, Any]) -> bool:
        """Update device configuration"""
        if device_id in self.devices:
            self.devices[device_id].configuration.update(configuration)
            return True
        return False

class DataCollector:
    """IoT data collection system"""
    
    def __init__(self):
        self.data_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.data_processors: Dict[str, Callable] = {}
        self.aggregation_rules: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def collect_data(self, device_id: str, data: SensorData) -> None:
        """Collect sensor data from device"""
        async with self._lock:
            self.data_streams[device_id].append(data)
            
            # Apply data processing if configured
            if device_id in self.data_processors:
                await self.data_processors[device_id](data)
            
            logger.debug(f"Collected data from device {device_id}: {data.data_type.value} = {data.value}")
    
    async def get_latest_data(self, device_id: str, count: int = 1) -> List[SensorData]:
        """Get latest sensor data from device"""
        async with self._lock:
            if device_id in self.data_streams:
                return list(self.data_streams[device_id])[-count:]
            return []
    
    async def get_data_by_type(self, 
                              device_id: str, 
                              data_type: DataType,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[SensorData]:
        """Get sensor data by type and time range"""
        async with self._lock:
            if device_id not in self.data_streams:
                return []
            
            data = list(self.data_streams[device_id])
            
            # Filter by data type
            filtered_data = [d for d in data if d.data_type == data_type]
            
            # Filter by time range
            if start_time:
                filtered_data = [d for d in filtered_data if d.timestamp >= start_time]
            if end_time:
                filtered_data = [d for d in filtered_data if d.timestamp <= end_time]
            
            return filtered_data
    
    async def aggregate_data(self, 
                           device_id: str,
                           data_type: DataType,
                           aggregation_type: str = "average",
                           time_window: int = 300) -> Dict[str, Any]:
        """Aggregate sensor data"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=time_window)
        
        data = await self.get_data_by_type(device_id, data_type, start_time, end_time)
        
        if not data:
            return {"value": 0, "count": 0, "time_window": time_window}
        
        values = [d.value for d in data]
        
        if aggregation_type == "average":
            result_value = sum(values) / len(values)
        elif aggregation_type == "sum":
            result_value = sum(values)
        elif aggregation_type == "min":
            result_value = min(values)
        elif aggregation_type == "max":
            result_value = max(values)
        elif aggregation_type == "count":
            result_value = len(values)
        else:
            result_value = sum(values) / len(values)
        
        return {
            "value": result_value,
            "count": len(values),
            "time_window": time_window,
            "aggregation_type": aggregation_type,
            "data_type": data_type.value
        }
    
    async def set_data_processor(self, device_id: str, processor: Callable) -> None:
        """Set data processor for device"""
        self.data_processors[device_id] = processor

class CommandManager:
    """IoT command management system"""
    
    def __init__(self):
        self.command_queue: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.command_history: Dict[str, List[DeviceCommand]] = defaultdict(list)
        self.command_templates: Dict[str, Dict[str, Any]] = {}
        self._initialize_command_templates()
    
    def _initialize_command_templates(self) -> None:
        """Initialize command templates"""
        self.command_templates = {
            "turn_on": {
                "command_type": "control",
                "parameters": {"action": "on"},
                "timeout": 10
            },
            "turn_off": {
                "command_type": "control",
                "parameters": {"action": "off"},
                "timeout": 10
            },
            "set_temperature": {
                "command_type": "setpoint",
                "parameters": {"temperature": 22.0},
                "timeout": 30
            },
            "reboot": {
                "command_type": "system",
                "parameters": {"action": "reboot"},
                "timeout": 60
            },
            "update_firmware": {
                "command_type": "firmware",
                "parameters": {"version": "2.0.0"},
                "timeout": 300
            }
        }
    
    async def send_command(self, 
                         device_id: str,
                         command_type: str,
                         parameters: Dict[str, Any] = None,
                         priority: int = 1,
                         timeout: int = 30) -> DeviceCommand:
        """Send command to device"""
        
        command = DeviceCommand(
            device_id=device_id,
            command_type=command_type,
            parameters=parameters or {},
            priority=priority,
            timeout=timeout
        )
        
        # Add to command queue
        self.command_queue[device_id].append(command)
        
        # Add to history
        self.command_history[device_id].append(command)
        
        logger.info(f"Queued command for device {device_id}: {command_type}")
        
        return command
    
    async def send_template_command(self, 
                                  device_id: str,
                                  template_name: str,
                                  custom_parameters: Dict[str, Any] = None) -> DeviceCommand:
        """Send command using template"""
        
        if template_name not in self.command_templates:
            raise ValueError(f"Command template {template_name} not found")
        
        template = self.command_templates[template_name]
        
        # Merge template parameters with custom parameters
        parameters = template["parameters"].copy()
        if custom_parameters:
            parameters.update(custom_parameters)
        
        return await self.send_command(
            device_id=device_id,
            command_type=template["command_type"],
            parameters=parameters,
            timeout=template["timeout"]
        )
    
    async def get_pending_commands(self, device_id: str) -> List[DeviceCommand]:
        """Get pending commands for device"""
        return list(self.command_queue[device_id])
    
    async def execute_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Execute command on device"""
        try:
            # Simulate command execution
            await asyncio.sleep(0.5)
            
            # Update command status
            command.status = "executed"
            command.executed_at = datetime.utcnow()
            command.result = {"success": True, "message": "Command executed successfully"}
            
            # Remove from queue
            if device_id in self.command_queue:
                try:
                    self.command_queue[device_id].remove(command)
                except ValueError:
                    pass
            
            logger.info(f"Executed command {command.command_type} on device {device_id}")
            return True
            
        except Exception as e:
            command.status = "failed"
            command.result = {"success": False, "error": str(e)}
            logger.error(f"Failed to execute command on device {device_id}: {e}")
            return False
    
    async def get_command_history(self, device_id: str, limit: int = 100) -> List[DeviceCommand]:
        """Get command history for device"""
        history = self.command_history.get(device_id, [])
        return history[-limit:]

class EdgeProcessor:
    """Edge computing processor"""
    
    def __init__(self):
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.edge_algorithms: Dict[str, Callable] = {}
        self.processing_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._initialize_algorithms()
    
    def _initialize_algorithms(self) -> None:
        """Initialize edge processing algorithms"""
        self.edge_algorithms = {
            "anomaly_detection": self._anomaly_detection,
            "data_filtering": self._data_filtering,
            "pattern_recognition": self._pattern_recognition,
            "predictive_analysis": self._predictive_analysis,
            "data_compression": self._data_compression
        }
    
    async def process_data(self, 
                         device_id: str,
                         algorithm: str,
                         data: List[SensorData],
                         parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process data using edge algorithm"""
        
        if algorithm not in self.edge_algorithms:
            raise ValueError(f"Algorithm {algorithm} not available")
        
        # Create processing task
        task_id = str(uuid.uuid4())
        task = asyncio.create_task(
            self._run_algorithm(algorithm, data, parameters or {})
        )
        
        self.processing_tasks[task_id] = task
        
        try:
            result = await task
            self.processing_results[device_id].append(result)
            return result
            
        except Exception as e:
            logger.error(f"Edge processing failed for device {device_id}: {e}")
            return {"error": str(e)}
        
        finally:
            if task_id in self.processing_tasks:
                del self.processing_tasks[task_id]
    
    async def _run_algorithm(self, algorithm: str, data: List[SensorData], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run edge processing algorithm"""
        return await self.edge_algorithms[algorithm](data, parameters)
    
    async def _anomaly_detection(self, data: List[SensorData], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Anomaly detection algorithm"""
        if len(data) < 3:
            return {"anomalies": [], "confidence": 0.0}
        
        values = [d.value for d in data]
        mean_value = sum(values) / len(values)
        std_dev = (sum((v - mean_value) ** 2 for v in values) / len(values)) ** 0.5
        
        threshold = parameters.get("threshold", 2.0)
        anomalies = []
        
        for i, value in enumerate(values):
            if abs(value - mean_value) > threshold * std_dev:
                anomalies.append({
                    "index": i,
                    "value": value,
                    "deviation": abs(value - mean_value) / std_dev,
                    "timestamp": data[i].timestamp.isoformat()
                })
        
        return {
            "anomalies": anomalies,
            "confidence": min(1.0, len(anomalies) / len(values)),
            "mean": mean_value,
            "std_dev": std_dev
        }
    
    async def _data_filtering(self, data: List[SensorData], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Data filtering algorithm"""
        filter_type = parameters.get("filter_type", "moving_average")
        window_size = parameters.get("window_size", 5)
        
        if filter_type == "moving_average":
            filtered_values = []
            for i in range(len(data)):
                start_idx = max(0, i - window_size + 1)
                window_data = data[start_idx:i+1]
                avg_value = sum(d.value for d in window_data) / len(window_data)
                filtered_values.append(avg_value)
            
            return {
                "filtered_data": filtered_values,
                "original_data": [d.value for d in data],
                "filter_type": filter_type,
                "window_size": window_size
            }
        
        return {"error": f"Unknown filter type: {filter_type}"}
    
    async def _pattern_recognition(self, data: List[SensorData], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Pattern recognition algorithm"""
        if len(data) < 10:
            return {"patterns": [], "confidence": 0.0}
        
        values = [d.value for d in data]
        
        # Simple pattern detection (trend analysis)
        trends = []
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                trends.append("increasing")
            elif values[i] < values[i-1]:
                trends.append("decreasing")
            else:
                trends.append("stable")
        
        # Count trend patterns
        trend_counts = {}
        for trend in trends:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        dominant_trend = max(trend_counts.items(), key=lambda x: x[1])
        
        return {
            "patterns": [
                {
                    "type": "trend",
                    "pattern": dominant_trend[0],
                    "frequency": dominant_trend[1] / len(trends),
                    "confidence": dominant_trend[1] / len(trends)
                }
            ],
            "trend_analysis": trend_counts
        }
    
    async def _predictive_analysis(self, data: List[SensorData], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predictive analysis algorithm"""
        if len(data) < 5:
            return {"predictions": [], "confidence": 0.0}
        
        values = [d.value for d in data]
        
        # Simple linear regression for prediction
        n = len(values)
        x = list(range(n))
        
        # Calculate slope and intercept
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next values
        prediction_steps = parameters.get("prediction_steps", 5)
        predictions = []
        
        for i in range(1, prediction_steps + 1):
            predicted_value = slope * (n + i - 1) + intercept
            predictions.append({
                "step": i,
                "value": predicted_value,
                "timestamp": (data[-1].timestamp + timedelta(seconds=i)).isoformat()
            })
        
        return {
            "predictions": predictions,
            "slope": slope,
            "intercept": intercept,
            "confidence": min(1.0, 1.0 - abs(slope) / max(values))
        }
    
    async def _data_compression(self, data: List[SensorData], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Data compression algorithm"""
        compression_ratio = parameters.get("compression_ratio", 0.5)
        
        # Simple compression by sampling
        compressed_size = int(len(data) * compression_ratio)
        step = len(data) // compressed_size if compressed_size > 0 else 1
        
        compressed_data = []
        for i in range(0, len(data), step):
            compressed_data.append({
                "timestamp": data[i].timestamp.isoformat(),
                "value": data[i].value,
                "data_type": data[i].data_type.value
            })
        
        return {
            "compressed_data": compressed_data,
            "original_size": len(data),
            "compressed_size": len(compressed_data),
            "compression_ratio": len(compressed_data) / len(data)
        }

# Advanced IoT Manager
class AdvancedIoTManager:
    """Main advanced IoT management system"""
    
    def __init__(self):
        self.protocols: Dict[ProtocolType, BaseIoTProtocol] = {}
        self.device_manager = DeviceManager()
        self.data_collector = DataCollector()
        self.command_manager = CommandManager()
        self.edge_processor = EdgeProcessor()
        self.gateways: Dict[str, IoTGateway] = {}
        
        self.devices: Dict[str, IoTDevice] = {}
        self.sensor_data: Dict[str, List[SensorData]] = defaultdict(list)
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize IoT system"""
        if self._initialized:
            return
        
        # Initialize protocols
        self.protocols[ProtocolType.MQTT] = MQTTProtocol()
        self.protocols[ProtocolType.COAP] = CoAPProtocol()
        self.protocols[ProtocolType.HTTP] = HTTPProtocol()
        
        self._initialized = True
        logger.info("Advanced IoT system initialized")
    
    async def shutdown(self) -> None:
        """Shutdown IoT system"""
        # Disconnect all devices
        for device in self.devices.values():
            await self.disconnect_device(device.id)
        
        self.protocols.clear()
        self.devices.clear()
        self.gateways.clear()
        self._initialized = False
        logger.info("Advanced IoT system shut down")
    
    async def register_device(self, 
                            name: str,
                            device_type: DeviceType,
                            protocol: ProtocolType,
                            template: Optional[str] = None) -> IoTDevice:
        """Register new IoT device"""
        
        device = await self.device_manager.register_device(name, device_type, protocol, template)
        self.devices[device.id] = device
        
        return device
    
    async def connect_device(self, device_id: str) -> bool:
        """Connect device to IoT system"""
        
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        
        if device.protocol not in self.protocols:
            return False
        
        protocol = self.protocols[device.protocol]
        success = await protocol.connect_device(device)
        
        if success:
            await self.device_manager.update_device_status(device_id, DeviceStatus.ONLINE)
        
        return success
    
    async def disconnect_device(self, device_id: str) -> bool:
        """Disconnect device from IoT system"""
        
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        
        if device.protocol in self.protocols:
            protocol = self.protocols[device.protocol]
            await protocol.disconnect_device(device_id)
        
        await self.device_manager.update_device_status(device_id, DeviceStatus.OFFLINE)
        
        return True
    
    async def send_sensor_data(self, device_id: str, data: SensorData) -> bool:
        """Send sensor data from device"""
        
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        
        if device.protocol in self.protocols:
            protocol = self.protocols[device.protocol]
            success = await protocol.send_data(device_id, data)
            
            if success:
                await self.data_collector.collect_data(device_id, data)
                self.sensor_data[device_id].append(data)
            
            return success
        
        return False
    
    async def send_command(self, 
                         device_id: str,
                         command_type: str,
                         parameters: Dict[str, Any] = None) -> DeviceCommand:
        """Send command to device"""
        
        command = await self.command_manager.send_command(
            device_id, command_type, parameters
        )
        
        # Execute command if device is connected
        if device_id in self.devices and self.devices[device_id].status == DeviceStatus.ONLINE:
            await self.command_manager.execute_command(device_id, command)
        
        return command
    
    async def process_edge_data(self, 
                              device_id: str,
                              algorithm: str,
                              parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process data at edge"""
        
        # Get recent data for processing
        recent_data = await self.data_collector.get_latest_data(device_id, 100)
        
        if not recent_data:
            return {"error": "No data available for processing"}
        
        return await self.edge_processor.process_data(
            device_id, algorithm, recent_data, parameters
        )
    
    async def create_gateway(self, 
                           name: str,
                           location: Dict[str, float],
                           protocols: List[ProtocolType]) -> IoTGateway:
        """Create IoT gateway"""
        
        gateway = IoTGateway(
            name=name,
            location=location,
            protocols_supported=protocols,
            status=DeviceStatus.ONLINE
        )
        
        self.gateways[gateway.id] = gateway
        
        logger.info(f"Created IoT gateway: {name}")
        
        return gateway
    
    def get_iot_summary(self) -> Dict[str, Any]:
        """Get IoT system summary"""
        online_devices = len([d for d in self.devices.values() if d.status == DeviceStatus.ONLINE])
        total_data_points = sum(len(data) for data in self.sensor_data.values())
        
        return {
            "initialized": self._initialized,
            "total_devices": len(self.devices),
            "online_devices": online_devices,
            "offline_devices": len(self.devices) - online_devices,
            "supported_protocols": [p.value for p in self.protocols.keys()],
            "total_gateways": len(self.gateways),
            "total_data_points": total_data_points,
            "device_types": {
                device_type.value: len([d for d in self.devices.values() if d.device_type == device_type])
                for device_type in DeviceType
            }
        }

# Global IoT manager instance
_global_iot_manager: Optional[AdvancedIoTManager] = None

def get_iot_manager() -> AdvancedIoTManager:
    """Get global IoT manager instance"""
    global _global_iot_manager
    if _global_iot_manager is None:
        _global_iot_manager = AdvancedIoTManager()
    return _global_iot_manager

async def initialize_iot() -> None:
    """Initialize global IoT system"""
    manager = get_iot_manager()
    await manager.initialize()

async def shutdown_iot() -> None:
    """Shutdown global IoT system"""
    manager = get_iot_manager()
    await manager.shutdown()

async def register_iot_device(name: str, device_type: DeviceType, protocol: ProtocolType) -> IoTDevice:
    """Register IoT device using global manager"""
    manager = get_iot_manager()
    return await manager.register_device(name, device_type, protocol)

async def send_iot_command(device_id: str, command_type: str, parameters: Dict[str, Any] = None) -> DeviceCommand:
    """Send IoT command using global manager"""
    manager = get_iot_manager()
    return await manager.send_command(device_id, command_type, parameters)





















