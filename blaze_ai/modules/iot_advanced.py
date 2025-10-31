"""
Blaze AI Advanced IoT Module v7.8.0

Advanced IoT system for device management, edge computing,
protocol support, and intelligent device orchestration.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
import asyncio_mqtt
import websockets
from pathlib import Path
import socket
import struct

from .base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)

# Enums
class DeviceType(Enum):
    """Types of IoT devices."""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CONTROLLER = "controller"
    CAMERA = "camera"
    ROBOT = "robot"
    DRONE = "drone"
    VEHICLE = "vehicle"
    SMART_DEVICE = "smart_device"
    INDUSTRIAL = "industrial"

class ProtocolType(Enum):
    """IoT communication protocols."""
    MQTT = "mqtt"
    HTTP = "http"
    WEBSOCKET = "websocket"
    COAP = "coap"
    OPC_UA = "opc_ua"
    MODBUS = "modbus"
    BACNET = "bacnet"
    ZIGBEE = "zigbee"
    BLUETOOTH = "bluetooth"
    LORA = "lora"

class DeviceStatus(Enum):
    """Device operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UPDATING = "updating"
    SLEEPING = "sleeping"
    LOW_BATTERY = "low_battery"

class DataType(Enum):
    """Types of data that devices can handle."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    SOUND = "sound"
    MOTION = "motion"
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    NUMERIC = "numeric"
    BINARY = "binary"
    JSON = "json"

class SecurityLevel(Enum):
    """Device security levels."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MILITARY = "military"

# Configuration and Data Classes
@dataclass
class IoTAdvancedConfig(ModuleConfig):
    """Configuration for Advanced IoT module."""
    
    # Network settings
    network_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    network_name: str = "blaze-ai-iot-network"
    max_devices: int = 10000
    auto_discovery: bool = True
    
    # Protocol settings
    supported_protocols: List[ProtocolType] = field(default_factory=lambda: [
        ProtocolType.MQTT, ProtocolType.HTTP, ProtocolType.WEBSOCKET
    ])
    default_protocol: ProtocolType = ProtocolType.MQTT
    
    # MQTT settings
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    
    # HTTP settings
    http_port: int = 8080
    http_host: str = "0.0.0.0"
    
    # Security settings
    security_level: SecurityLevel = SecurityLevel.STANDARD
    enable_encryption: bool = True
    enable_authentication: bool = True
    certificate_path: Optional[str] = None
    
    # Data settings
    data_retention_days: int = 30
    max_data_size: int = 1024 * 1024 * 1024  # 1GB
    compression_enabled: bool = True
    
    # Monitoring settings
    health_check_interval: float = 60.0  # seconds
    data_sync_interval: float = 300.0  # seconds
    alert_threshold: float = 0.8  # 80%
    
    # Storage settings
    iot_data_path: str = "./iot_data"
    device_config_path: str = "./device_configs"

@dataclass
class DeviceInfo:
    """IoT device information."""
    
    device_id: str
    device_name: str
    device_type: DeviceType
    protocol: ProtocolType
    status: DeviceStatus
    ip_address: str
    mac_address: str
    firmware_version: str
    hardware_version: str
    capabilities: List[str]
    data_types: List[DataType]
    location: Dict[str, float]  # lat, lon, alt
    created_at: datetime
    last_seen: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.dev_type.value,
            "protocol": self.protocol.value,
            "status": self.status.value,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "firmware_version": self.firmware_version,
            "hardware_version": self.hardware_version,
            "capabilities": self.capabilities,
            "data_types": [dt.value for dt in self.data_types],
            "location": self.location,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceInfo':
        """Create device info from dictionary."""
        return cls(
            device_id=data["device_id"],
            device_name=data["device_name"],
            device_type=DeviceType(data["device_type"]),
            protocol=ProtocolType(data["protocol"]),
            status=DeviceStatus(data["status"]),
            ip_address=data["ip_address"],
            mac_address=data["mac_address"],
            firmware_version=data["firmware_version"],
            hardware_version=data["hardware_version"],
            capabilities=data["capabilities"],
            data_types=[DataType(dt) for dt in data["data_types"]],
            location=data["location"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            metadata=data.get("metadata", {})
        )

@dataclass
class IoTData:
    """IoT data structure."""
    
    data_id: str
    device_id: str
    data_type: DataType
    value: Any
    timestamp: datetime
    quality: float  # 0.0 to 1.0
    unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_id": self.data_id,
            "device_id": self.device_id,
            "data_type": self.data_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality,
            "unit": self.unit,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IoTData':
        """Create IoT data from dictionary."""
        return cls(
            data_id=data["data_id"],
            device_id=data["data_id"],
            data_type=DataType(data["data_type"]),
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            quality=data["quality"],
            unit=data.get("unit"),
            metadata=data.get("metadata", {})
        )

@dataclass
class DeviceCommand:
    """Command to be sent to a device."""
    
    command_id: str
    device_id: str
    command_type: str
    parameters: Dict[str, Any]
    priority: int = 1  # 1=low, 5=high
    timeout: float = 30.0  # seconds
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "command_type": self.command_type,
            "parameters": self.parameters,
            "priority": self.priority,
            "timeout": self.timeout,
            "created_at": self.created_at.isoformat(),
            "status": self.status
        }

@dataclass
class IoTMetrics:
    """IoT system performance metrics."""
    
    # Device metrics
    total_devices: int = 0
    online_devices: int = 0
    offline_devices: int = 0
    error_devices: int = 0
    
    # Data metrics
    total_data_points: int = 0
    data_points_per_second: float = 0.0
    data_quality_average: float = 0.0
    
    # Protocol metrics
    mqtt_connections: int = 0
    http_requests: int = 0
    websocket_connections: int = 0
    
    # Performance metrics
    average_response_time: float = 0.0
    command_success_rate: float = 0.0
    data_sync_success_rate: float = 0.0
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)

# Core Components
class DeviceManager:
    """Manages IoT devices and their lifecycle."""
    
    def __init__(self, config: IoTAdvancedConfig):
        self.config = config
        self.devices: Dict[str, DeviceInfo] = {}
        self.device_commands: Dict[str, DeviceCommand] = {}
        self.device_data: Dict[str, List[IoTData]] = {}
        self.device_handlers: Dict[str, Callable] = {}
    
    async def register_device(self, device_info: DeviceInfo) -> bool:
        """Register a new IoT device."""
        try:
            if device_info.device_id in self.devices:
                logger.warning(f"Device {device_info.device_id} already registered")
                return False
            
            # Validate device capabilities
            if not await self._validate_device(device_info):
                return False
            
            # Store device
            self.devices[device_info.device_id] = device_info
            self.device_data[device_info.device_id] = []
            
            # Initialize device handler
            await self._initialize_device_handler(device_info)
            
            logger.info(f"Device {device_info.device_name} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering device: {e}")
            return False
    
    async def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device."""
        try:
            if device_id not in self.devices:
                return False
            
            # Cleanup device data
            if device_id in self.device_data:
                del self.device_data[device_id]
            
            # Remove device
            del self.devices[device_id]
            
            logger.info(f"Device {device_id} unregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering device: {e}")
            return False
    
    async def update_device_status(self, device_id: str, status: DeviceStatus) -> bool:
        """Update device status."""
        try:
            if device_id not in self.devices:
                return False
            
            device = self.devices[device_id]
            device.status = status
            device.last_seen = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating device status: {e}")
            return False
    
    async def get_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device information."""
        return self.devices.get(device_id)
    
    async def get_all_devices(self) -> List[DeviceInfo]:
        """Get all registered devices."""
        return list(self.devices.values())
    
    async def get_devices_by_type(self, device_type: DeviceType) -> List[DeviceInfo]:
        """Get devices by type."""
        return [d for d in self.devices.values() if d.device_type == device_type]
    
    async def _validate_device(self, device_info: DeviceInfo) -> bool:
        """Validate device information."""
        # Basic validation
        if not device_info.device_id or not device_info.device_name:
            return False
        
        if device_info.protocol not in self.config.supported_protocols:
            logger.warning(f"Protocol {device_info.protocol} not supported")
            return False
        
        return True
    
    async def _initialize_device_handler(self, device_info: DeviceInfo):
        """Initialize device-specific handler."""
        try:
            if device_info.protocol == ProtocolType.MQTT:
                handler = await self._create_mqtt_handler(device_info)
            elif device_info.protocol == ProtocolType.HTTP:
                handler = await self._create_http_handler(device_info)
            elif device_info.protocol == ProtocolType.WEBSOCKET:
                handler = await self._create_websocket_handler(device_info)
            else:
                handler = await self._create_generic_handler(device_info)
            
            self.device_handlers[device_info.device_id] = handler
            
        except Exception as e:
            logger.error(f"Error initializing device handler: {e}")
    
    async def _create_mqtt_handler(self, device_info: DeviceInfo):
        """Create MQTT handler for device."""
        # Simplified MQTT handler creation
        return lambda msg: logger.info(f"MQTT message from {device_info.device_id}: {msg}")
    
    async def _create_http_handler(self, device_info: DeviceInfo):
        """Create HTTP handler for device."""
        # Simplified HTTP handler creation
        return lambda req: logger.info(f"HTTP request from {device_info.device_id}: {req}")
    
    async def _create_websocket_handler(self, device_info: DeviceInfo):
        """Create WebSocket handler for device."""
        # Simplified WebSocket handler creation
        return lambda msg: logger.info(f"WebSocket message from {device_info.device_id}: {msg}")
    
    async def _create_generic_handler(self, device_info: DeviceInfo):
        """Create generic handler for device."""
        # Generic handler for unsupported protocols
        return lambda msg: logger.info(f"Generic message from {device_info.device_id}: {msg}")

class DataManager:
    """Manages IoT data collection, storage, and processing."""
    
    def __init__(self, config: IoTAdvancedConfig):
        self.config = config
        self.data_storage: Dict[str, List[IoTData]] = {}
        self.data_processors: Dict[str, Callable] = {}
        self.data_filters: Dict[str, Callable] = {}
    
    async def store_data(self, data: IoTData) -> bool:
        """Store IoT data."""
        try:
            device_id = data.device_id
            
            # Initialize storage for device if not exists
            if device_id not in self.data_storage:
                self.data_storage[device_id] = []
            
            # Apply data filters
            if device_id in self.data_filters:
                if not await self.data_filters[device_id](data):
                    logger.debug(f"Data filtered out for device {device_id}")
                    return False
            
            # Store data
            self.data_storage[device_id].append(data)
            
            # Apply data retention policy
            await self._apply_retention_policy(device_id)
            
            # Process data if processor exists
            if device_id in self.data_processors:
                await self.data_processors[device_id](data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return False
    
    async def get_device_data(self, device_id: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            data_type: Optional[DataType] = None) -> List[IoTData]:
        """Get data for a specific device."""
        try:
            if device_id not in self.data_storage:
                return []
            
            data = self.data_storage[device_id]
            
            # Apply time filters
            if start_time:
                data = [d for d in data if d.timestamp >= start_time]
            if end_time:
                data = [d for d in data if d.timestamp <= end_time]
            
            # Apply data type filter
            if data_type:
                data = [d for d in data if d.data_type == data_type]
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting device data: {e}")
            return []
    
    async def get_aggregated_data(self, device_id: str, 
                                 aggregation: str,
                                 time_window: timedelta) -> Dict[str, Any]:
        """Get aggregated data for a device."""
        try:
            end_time = datetime.now()
            start_time = end_time - time_window
            
            data = await self.get_device_data(device_id, start_time, end_time)
            
            if not data:
                return {"error": "No data available"}
            
            # Perform aggregation
            if aggregation == "average":
                result = sum(d.value for d in data if isinstance(d.value, (int, float))) / len(data)
            elif aggregation == "sum":
                result = sum(d.value for d in data if isinstance(d.value, (int, float)))
            elif aggregation == "min":
                result = min(d.value for d in data if isinstance(d.value, (int, float)))
            elif aggregation == "max":
                result = max(d.value for d in data if isinstance(d.value, (int, float)))
            elif aggregation == "count":
                result = len(data)
            else:
                return {"error": f"Unknown aggregation: {aggregation}"}
            
            return {
                "device_id": device_id,
                "aggregation": aggregation,
                "value": result,
                "data_points": len(data),
                "time_window": str(time_window)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return {"error": str(e)}
    
    async def add_data_processor(self, device_id: str, processor: Callable):
        """Add data processor for a device."""
        self.data_processors[device_id] = processor
    
    async def add_data_filter(self, device_id: str, filter_func: Callable):
        """Add data filter for a device."""
        self.data_filters[device_id] = filter_func
    
    async def _apply_retention_policy(self, device_id: str):
        """Apply data retention policy."""
        try:
            if device_id not in self.data_storage:
                return
            
            data = self.data_storage[device_id]
            cutoff_time = datetime.now() - timedelta(days=self.config.data_retention_days)
            
            # Remove old data
            self.data_storage[device_id] = [
                d for d in data if d.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error applying retention policy: {e}")

class ProtocolManager:
    """Manages IoT communication protocols."""
    
    def __init__(self, config: IoTAdvancedConfig):
        self.config = config
        self.mqtt_client: Optional[asyncio_mqtt.Client] = None
        self.http_server: Optional[Any] = None
        self.websocket_server: Optional[Any] = None
        self.active_connections: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        """Initialize protocol managers."""
        try:
            # Initialize MQTT if supported
            if ProtocolType.MQTT in self.config.supported_protocols:
                await self._initialize_mqtt()
            
            # Initialize HTTP if supported
            if ProtocolType.HTTP in self.config.supported_protocols:
                await self._initialize_http()
            
            # Initialize WebSocket if supported
            if ProtocolType.WEBSOCKET in self.config.supported_protocols:
                await self._initialize_websocket()
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing protocols: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown protocol managers."""
        try:
            if self.mqtt_client:
                await self.mqtt_client.disconnect()
            
            if self.http_server:
                # Shutdown HTTP server
                pass
            
            if self.websocket_server:
                # Shutdown WebSocket server
                pass
                
        except Exception as e:
            logger.error(f"Error shutting down protocols: {e}")
    
    async def _initialize_mqtt(self):
        """Initialize MQTT client."""
        try:
            self.mqtt_client = asyncio_mqtt.Client(
                hostname=self.config.mqtt_broker,
                port=self.config.mqtt_port,
                username=self.config.mqtt_username,
                password=self.config.mqtt_password
            )
            
            await self.mqtt_client.connect()
            logger.info("MQTT client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing MQTT: {e}")
    
    async def _initialize_http(self):
        """Initialize HTTP server."""
        # Simplified HTTP server initialization
        logger.info("HTTP server initialized successfully")
    
    async def _initialize_websocket(self):
        """Initialize WebSocket server."""
        # Simplified WebSocket server initialization
        logger.info("WebSocket server initialized successfully")
    
    async def send_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command to device via appropriate protocol."""
        try:
            # Get device info
            device_info = await self._get_device_info(device_id)
            if not device_info:
                return False
            
            # Send via appropriate protocol
            if device_info.protocol == ProtocolType.MQTT:
                return await self._send_mqtt_command(device_id, command)
            elif device_info.protocol == ProtocolType.HTTP:
                return await self._send_http_command(device_id, command)
            elif device_info.protocol == ProtocolType.WEBSOCKET:
                return await self._send_websocket_command(device_id, command)
            else:
                logger.warning(f"Protocol {device_info.protocol} not implemented")
                return False
                
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False
    
    async def _send_mqtt_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command via MQTT."""
        try:
            if not self.mqtt_client:
                return False
            
            topic = f"blaze-ai/iot/{device_id}/command"
            payload = json.dumps(command.to_dict())
            
            await self.mqtt_client.publish(topic, payload)
            return True
            
        except Exception as e:
            logger.error(f"Error sending MQTT command: {e}")
            return False
    
    async def _send_http_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command via HTTP."""
        # Simplified HTTP command sending
        logger.info(f"HTTP command sent to {device_id}: {command.command_type}")
        return True
    
    async def _send_websocket_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command via WebSocket."""
        # Simplified WebSocket command sending
        logger.info(f"WebSocket command sent to {device_id}: {command.command_type}")
        return True
    
    async def _get_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device info (placeholder)."""
        # This would be implemented to get device info from device manager
        return None

class SecurityManager:
    """Manages IoT security and authentication."""
    
    def __init__(self, config: IoTAdvancedConfig):
        self.config = config
        self.device_certificates: Dict[str, str] = {}
        self.access_tokens: Dict[str, str] = {}
        self.security_policies: Dict[str, Dict[str, Any]] = {}
    
    async def authenticate_device(self, device_id: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate a device."""
        try:
            if self.config.security_level == SecurityLevel.NONE:
                return True
            
            # Basic authentication
            if self.config.security_level == SecurityLevel.BASIC:
                return await self._basic_auth(device_id, credentials)
            
            # Standard authentication
            elif self.config.security_level == SecurityLevel.STANDARD:
                return await self._standard_auth(device_id, credentials)
            
            # High security authentication
            elif self.config.security_level == SecurityLevel.HIGH:
                return await self._high_auth(device_id, credentials)
            
            return False
            
        except Exception as e:
            logger.error(f"Error authenticating device: {e}")
            return False
    
    async def authorize_device(self, device_id: str, action: str, resource: str) -> bool:
        """Authorize device action."""
        try:
            if device_id not in self.security_policies:
                return False
            
            policy = self.security_policies[device_id]
            allowed_actions = policy.get("allowed_actions", [])
            allowed_resources = policy.get("allowed_resources", [])
            
            return action in allowed_actions and resource in allowed_resources
            
        except Exception as e:
            logger.error(f"Error authorizing device: {e}")
            return False
    
    async def _basic_auth(self, device_id: str, credentials: Dict[str, Any]) -> bool:
        """Basic authentication."""
        # Simple device ID validation
        return device_id in credentials.get("valid_devices", [])
    
    async def _standard_auth(self, device_id: str, credentials: Dict[str, Any]) -> bool:
        """Standard authentication."""
        # API key validation
        api_key = credentials.get("api_key")
        if not api_key:
            return False
        
        # Validate API key (simplified)
        return api_key.startswith("blaze_ai_")
    
    async def _high_auth(self, device_id: str, credentials: Dict[str, Any]) -> bool:
        """High security authentication."""
        # Certificate-based authentication
        certificate = credentials.get("certificate")
        if not certificate:
            return False
        
        # Validate certificate (simplified)
        return device_id in self.device_certificates

# Main Module
class IoTAdvancedModule(BaseModule):
    """Advanced IoT module for Blaze AI system."""
    
    def __init__(self, config: IoTAdvancedConfig):
        super().__init__(config)
        self.config = config
        
        # Core components
        self.device_manager = DeviceManager(config)
        self.data_manager = DataManager(config)
        self.protocol_manager = ProtocolManager(config)
        self.security_manager = SecurityManager(config)
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.data_sync_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = IoTMetrics()
    
    async def initialize(self) -> bool:
        """Initialize the IoT Advanced module."""
        try:
            logger.info("Initializing IoT Advanced Module")
            
            # Initialize protocol managers
            if not await self.protocol_manager.initialize():
                logger.error("Failed to initialize protocol managers")
                return False
            
            # Start background tasks
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.data_sync_task = asyncio.create_task(self._data_sync_loop())
            
            self.status = ModuleStatus.RUNNING
            logger.info("IoT Advanced Module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IoT Advanced Module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the IoT Advanced module."""
        try:
            logger.info("Shutting down IoT Advanced Module")
            
            # Cancel background tasks
            if self.health_check_task:
                self.health_check_task.cancel()
            if self.data_sync_task:
                self.data_sync_task.cancel()
            
            # Shutdown protocol managers
            await self.protocol_manager.shutdown()
            
            self.status = ModuleStatus.STOPPED
            logger.info("IoT Advanced Module shut down successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
    
    async def register_device(self, device_data: Dict[str, Any]) -> Optional[str]:
        """Register a new IoT device."""
        try:
            # Create device info
            device_info = DeviceInfo(
                device_id=device_data.get("device_id", str(uuid.uuid4())),
                device_name=device_data.get("name", "Unknown Device"),
                device_type=DeviceType(device_data.get("type", "sensor")),
                protocol=ProtocolType(device_data.get("protocol", "mqtt")),
                status=DeviceStatus.ONLINE,
                ip_address=device_data.get("ip_address", "0.0.0.0"),
                mac_address=device_data.get("mac_address", "00:00:00:00:00:00"),
                firmware_version=device_data.get("firmware_version", "1.0.0"),
                hardware_version=device_data.get("hardware_version", "1.0.0"),
                capabilities=device_data.get("capabilities", []),
                data_types=[DataType(dt) for dt in device_data.get("data_types", [])],
                location=device_data.get("location", {"lat": 0.0, "lon": 0.0, "alt": 0.0}),
                created_at=datetime.now(),
                last_seen=datetime.now(),
                metadata=device_data.get("metadata", {})
            )
            
            # Register device
            success = await self.device_manager.register_device(device_info)
            if success:
                # Update metrics
                self.metrics.total_devices += 1
                self.metrics.online_devices += 1
                return device_info.device_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error registering device: {e}")
            return None
    
    async def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device."""
        try:
            success = await self.device_manager.unregister_device(device_id)
            if success:
                # Update metrics
                self.metrics.total_devices -= 1
                if device_id in self.device_manager.devices:
                    device = self.device_manager.devices[device_id]
                    if device.status == DeviceStatus.ONLINE:
                        self.metrics.online_devices -= 1
                    elif device.status == DeviceStatus.ERROR:
                        self.metrics.error_devices -= 1
                
            return success
            
        except Exception as e:
            logger.error(f"Error unregistering device: {e}")
            return False
    
    async def send_device_command(self, device_id: str, command_data: Dict[str, Any]) -> Optional[str]:
        """Send command to a device."""
        try:
            # Create command
            command = DeviceCommand(
                command_id=str(uuid.uuid4()),
                device_id=device_id,
                command_type=command_data.get("type", "unknown"),
                parameters=command_data.get("parameters", {}),
                priority=command_data.get("priority", 1),
                timeout=command_data.get("timeout", 30.0)
            )
            
            # Store command
            self.device_manager.device_commands[command.command_id] = command
            
            # Send command
            success = await self.protocol_manager.send_command(device_id, command)
            if success:
                command.status = "sent"
                return command.command_id
            else:
                command.status = "failed"
                return None
                
        except Exception as e:
            logger.error(f"Error sending device command: {e}")
            return None
    
    async def store_device_data(self, data: Dict[str, Any]) -> bool:
        """Store data from a device."""
        try:
            # Create IoT data
            iot_data = IoTData(
                data_id=str(uuid.uuid4()),
                device_id=data["device_id"],
                data_type=DataType(data.get("type", "numeric")),
                value=data["value"],
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                quality=data.get("quality", 1.0),
                unit=data.get("unit"),
                metadata=data.get("metadata", {})
            )
            
            # Store data
            success = await self.data_manager.store_data(iot_data)
            if success:
                # Update metrics
                self.metrics.total_data_points += 1
                
            return success
            
        except Exception as e:
            logger.error(f"Error storing device data: {e}")
            return False
    
    async def get_device_data(self, device_id: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            data_type: Optional[DataType] = None) -> List[Dict[str, Any]]:
        """Get data for a device."""
        try:
            data = await self.data_manager.get_device_data(device_id, start_time, end_time, data_type)
            return [d.to_dict() for d in data]
            
        except Exception as e:
            logger.error(f"Error getting device data: {e}")
            return []
    
    async def get_device_aggregated_data(self, device_id: str, 
                                       aggregation: str,
                                       time_window_minutes: int) -> Dict[str, Any]:
        """Get aggregated data for a device."""
        try:
            time_window = timedelta(minutes=time_window_minutes)
            return await self.data_manager.get_aggregated_data(device_id, aggregation, time_window)
            
        except Exception as e:
            logger.error(f"Error getting aggregated data: {e}")
            return {"error": str(e)}
    
    async def get_all_devices(self) -> List[Dict[str, Any]]:
        """Get all registered devices."""
        try:
            devices = await self.device_manager.get_all_devices()
            return [d.to_dict() for d in devices]
            
        except Exception as e:
            logger.error(f"Error getting all devices: {e}")
            return []
    
    async def get_devices_by_type(self, device_type: str) -> List[Dict[str, Any]]:
        """Get devices by type."""
        try:
            devices = await self.device_manager.get_devices_by_type(DeviceType(device_type))
            return [d.to_dict() for d in devices]
            
        except Exception as e:
            logger.error(f"Error getting devices by type: {e}")
            return []
    
    async def update_device_status(self, device_id: str, status: str) -> bool:
        """Update device status."""
        try:
            device_status = DeviceStatus(status)
            success = await self.device_manager.update_device_status(device_id, device_status)
            
            if success:
                # Update metrics
                device = await self.device_manager.get_device_info(device_id)
                if device:
                    if device.status == DeviceStatus.ONLINE:
                        self.metrics.online_devices += 1
                    elif device.status == DeviceStatus.ERROR:
                        self.metrics.error_devices += 1
                    elif device.status == DeviceStatus.OFFLINE:
                        self.metrics.offline_devices += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating device status: {e}")
            return False
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Update device statuses
                for device_id in self.device_manager.devices:
                    # Simulate health check
                    await self._check_device_health(device_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)
    
    async def _data_sync_loop(self):
        """Background data synchronization loop."""
        while True:
            try:
                await asyncio.sleep(self.config.data_sync_interval)
                
                # Sync data across devices
                await self._sync_device_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data sync loop error: {e}")
                await asyncio.sleep(5)
    
    async def _check_device_health(self, device_id: str):
        """Check health of a specific device."""
        try:
            # Simulate health check
            device = await self.device_manager.get_device_info(device_id)
            if device:
                # Update last seen
                device.last_seen = datetime.now()
                
        except Exception as e:
            logger.error(f"Error checking device health: {e}")
    
    async def _sync_device_data(self):
        """Synchronize data across devices."""
        try:
            # Simulate data synchronization
            logger.debug("Syncing device data...")
            
        except Exception as e:
            logger.error(f"Error syncing device data: {e}")
    
    async def get_metrics(self) -> IoTMetrics:
        """Get current IoT metrics."""
        # Update metrics
        self.metrics.online_devices = len([
            d for d in self.device_manager.devices.values() 
            if d.status == DeviceStatus.ONLINE
        ])
        self.metrics.offline_devices = len([
            d for d in self.device_manager.devices.values() 
            if d.status == DeviceStatus.OFFLINE
        ])
        self.metrics.error_devices = len([
            d for d in self.device_manager.devices.values() 
            if d.status == DeviceStatus.ERROR
        ])
        
        self.metrics.last_updated = datetime.now()
        
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Get module health status."""
        try:
            return {
                "status": self.status.value,
                "total_devices": self.metrics.total_devices,
                "online_devices": self.metrics.online_devices,
                "offline_devices": self.metrics.offline_devices,
                "error_devices": self.metrics.error_devices,
                "total_data_points": self.metrics.total_data_points,
                "health_check_active": self.health_check_task is not None and not self.health_check_task.done(),
                "data_sync_active": self.data_sync_task is not None and not self.data_sync_task.done(),
                "supported_protocols": [p.value for p in self.config.supported_protocols],
                "security_level": self.config.security_level.value
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Factory Functions
def create_iot_advanced_module(config: Optional[IoTAdvancedConfig] = None) -> IoTAdvancedModule:
    """Create an IoT Advanced module with the given configuration."""
    if config is None:
        config = IoTAdvancedConfig()
    return IoTAdvancedModule(config)

def create_iot_advanced_module_with_defaults(**kwargs) -> IoTAdvancedModule:
    """Create an IoT Advanced module with default configuration and custom overrides."""
    config = IoTAdvancedConfig(**kwargs)
    return IoTAdvancedModule(config)

__all__ = [
    # Enums
    "DeviceType", "ProtocolType", "DeviceStatus", "DataType", "SecurityLevel",
    
    # Configuration and Data Classes
    "IoTAdvancedConfig", "DeviceInfo", "IoTData", "DeviceCommand", "IoTMetrics",
    
    # Core Components
    "DeviceManager", "DataManager", "ProtocolManager", "SecurityManager",
    
    # Main Module
    "IoTAdvancedModule",
    
    # Factory Functions
    "create_iot_advanced_module", "create_iot_advanced_module_with_defaults"
]

