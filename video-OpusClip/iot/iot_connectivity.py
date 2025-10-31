#!/usr/bin/env python3
"""
IoT Connectivity System

Advanced IoT connectivity with:
- Device management and registration
- Real-time sensor data collection
- Edge computing integration
- Device communication protocols
- IoT analytics and monitoring
- Device security and authentication
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
import paho.mqtt.client as mqtt
import asyncio_mqtt
import websockets
from cryptography.fernet import Fernet
import hashlib

logger = structlog.get_logger("iot_connectivity")

# =============================================================================
# IOT MODELS
# =============================================================================

class DeviceType(Enum):
    """IoT device types."""
    CAMERA = "camera"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    EDGE_COMPUTER = "edge_computer"
    MOBILE_DEVICE = "mobile_device"
    WEARABLE = "wearable"
    SMART_SPEAKER = "smart_speaker"

class DeviceStatus(Enum):
    """Device status."""
    ONLINE = "online"
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SLEEPING = "sleeping"

class CommunicationProtocol(Enum):
    """Communication protocols."""
    MQTT = "mqtt"
    HTTP = "http"
    WEBSOCKET = "websocket"
    COAP = "coap"
    MODBUS = "modbus"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    LORA = "lora"

class DataType(Enum):
    """IoT data types."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"
    SOUND = "sound"
    VIDEO = "video"
    IMAGE = "image"
    GPS = "gps"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    CUSTOM = "custom"

@dataclass
class Device:
    """IoT device information."""
    device_id: str
    name: str
    device_type: DeviceType
    status: DeviceStatus
    protocol: CommunicationProtocol
    ip_address: Optional[str]
    mac_address: Optional[str]
    location: Dict[str, float]  # lat, lng, alt
    capabilities: List[str]
    firmware_version: str
    hardware_version: str
    last_seen: datetime
    created_at: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.device_id:
            self.device_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.last_seen:
            self.last_seen = datetime.utcnow()
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type.value,
            "status": self.status.value,
            "protocol": self.protocol.value,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "location": self.location,
            "capabilities": self.capabilities,
            "firmware_version": self.firmware_version,
            "hardware_version": self.hardware_version,
            "last_seen": self.last_seen.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class SensorData:
    """IoT sensor data."""
    data_id: str
    device_id: str
    data_type: DataType
    value: Any
    unit: str
    timestamp: datetime
    quality: float  # 0.0 to 1.0
    location: Optional[Dict[str, float]]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.data_id:
            self.data_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_id": self.data_id,
            "device_id": self.device_id,
            "data_type": self.data_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality,
            "location": self.location,
            "metadata": self.metadata
        }

@dataclass
class DeviceCommand:
    """Device command."""
    command_id: str
    device_id: str
    command_type: str
    parameters: Dict[str, Any]
    priority: int
    timeout: int
    created_at: datetime
    executed_at: Optional[datetime]
    status: str
    
    def __post_init__(self):
        if not self.command_id:
            self.command_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "command_type": self.command_type,
            "parameters": self.parameters,
            "priority": self.priority,
            "timeout": self.timeout,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "status": self.status
        }

@dataclass
class IoTConfig:
    """IoT system configuration."""
    mqtt_broker: str
    mqtt_port: int
    mqtt_username: Optional[str]
    mqtt_password: Optional[str]
    websocket_port: int
    device_timeout: int
    data_retention_days: int
    encryption_key: Optional[str]
    enable_ssl: bool
    max_devices: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mqtt_broker": self.mqtt_broker,
            "mqtt_port": self.mqtt_port,
            "mqtt_username": self.mqtt_username,
            "websocket_port": self.websocket_port,
            "device_timeout": self.device_timeout,
            "data_retention_days": self.data_retention_days,
            "enable_ssl": self.enable_ssl,
            "max_devices": self.max_devices
        }

# =============================================================================
# IOT DEVICE MANAGER
# =============================================================================

class IoTDeviceManager:
    """IoT device management system."""
    
    def __init__(self, config: IoTConfig):
        self.config = config
        self.devices: Dict[str, Device] = {}
        self.device_connections: Dict[str, Any] = {}
        self.sensor_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.device_commands: Dict[str, List[DeviceCommand]] = defaultdict(list)
        
        # MQTT client
        self.mqtt_client: Optional[asyncio_mqtt.Client] = None
        
        # WebSocket server
        self.websocket_server: Optional[Any] = None
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Encryption
        self.encryption_key = config.encryption_key
        self.cipher_suite = Fernet(config.encryption_key.encode()) if config.encryption_key else None
        
        # Statistics
        self.stats = {
            'total_devices': 0,
            'online_devices': 0,
            'offline_devices': 0,
            'total_data_points': 0,
            'total_commands': 0,
            'successful_commands': 0,
            'failed_commands': 0
        }
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start the IoT device manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start MQTT client
        await self._start_mqtt_client()
        
        # Start WebSocket server
        await self._start_websocket_server()
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("IoT Device Manager started")
    
    async def stop(self) -> None:
        """Stop the IoT device manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop MQTT client
        if self.mqtt_client:
            await self.mqtt_client.disconnect()
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        logger.info("IoT Device Manager stopped")
    
    async def _start_mqtt_client(self) -> None:
        """Start MQTT client."""
        try:
            self.mqtt_client = asyncio_mqtt.Client(
                hostname=self.config.mqtt_broker,
                port=self.config.mqtt_port,
                username=self.config.mqtt_username,
                password=self.config.mqtt_password
            )
            
            await self.mqtt_client.connect()
            
            # Subscribe to device topics
            await self.mqtt_client.subscribe("devices/+/data")
            await self.mqtt_client.subscribe("devices/+/status")
            await self.mqtt_client.subscribe("devices/+/response")
            
            # Start message handling
            asyncio.create_task(self._handle_mqtt_messages())
            
            logger.info("MQTT client started", broker=self.config.mqtt_broker)
        
        except Exception as e:
            logger.error("Failed to start MQTT client", error=str(e))
            raise
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server."""
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                "0.0.0.0",
                self.config.websocket_port
            )
            
            logger.info("WebSocket server started", port=self.config.websocket_port)
        
        except Exception as e:
            logger.error("Failed to start WebSocket server", error=str(e))
            raise
    
    async def _handle_mqtt_messages(self) -> None:
        """Handle MQTT messages."""
        try:
            async for message in self.mqtt_client.messages:
                topic_parts = message.topic.value.split('/')
                
                if len(topic_parts) >= 3:
                    device_id = topic_parts[1]
                    message_type = topic_parts[2]
                    
                    # Decrypt message if encryption is enabled
                    payload = message.payload
                    if self.cipher_suite:
                        try:
                            payload = self.cipher_suite.decrypt(payload)
                        except Exception:
                            logger.warning("Failed to decrypt MQTT message", device_id=device_id)
                            continue
                    
                    data = json.loads(payload.decode())
                    
                    if message_type == "data":
                        await self._handle_sensor_data(device_id, data)
                    elif message_type == "status":
                        await self._handle_device_status(device_id, data)
                    elif message_type == "response":
                        await self._handle_command_response(device_id, data)
        
        except Exception as e:
            logger.error("MQTT message handling error", error=str(e))
    
    async def _handle_websocket_connection(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """Handle WebSocket connection."""
        try:
            # Authenticate device
            device_id = await self._authenticate_device(websocket)
            if not device_id:
                await websocket.close(code=4001, reason="Authentication failed")
                return
            
            # Register connection
            self.websocket_connections[device_id] = websocket
            
            # Update device status
            if device_id in self.devices:
                self.devices[device_id].status = DeviceStatus.ONLINE
                self.devices[device_id].last_seen = datetime.utcnow()
            
            logger.info("Device connected via WebSocket", device_id=device_id)
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(device_id, data)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in WebSocket message", device_id=device_id)
                except Exception as e:
                    logger.error("WebSocket message handling error", device_id=device_id, error=str(e))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error("WebSocket connection error", error=str(e))
        finally:
            # Clean up connection
            if device_id in self.websocket_connections:
                del self.websocket_connections[device_id]
            
            # Update device status
            if device_id in self.devices:
                self.devices[device_id].status = DeviceStatus.OFFLINE
            
            logger.info("Device disconnected from WebSocket", device_id=device_id)
    
    async def _authenticate_device(self, websocket: websockets.WebSocketServerProtocol) -> Optional[str]:
        """Authenticate device connection."""
        try:
            # Wait for authentication message
            message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_data = json.loads(message)
            
            device_id = auth_data.get("device_id")
            auth_token = auth_data.get("auth_token")
            
            if not device_id or not auth_token:
                return None
            
            # Verify device exists and token is valid
            if device_id in self.devices:
                # In a real implementation, you'd verify the auth_token
                return device_id
            
            return None
        
        except Exception as e:
            logger.error("Device authentication error", error=str(e))
            return None
    
    async def _handle_websocket_message(self, device_id: str, data: Dict[str, Any]) -> None:
        """Handle WebSocket message from device."""
        message_type = data.get("type")
        
        if message_type == "sensor_data":
            await self._handle_sensor_data(device_id, data.get("data", {}))
        elif message_type == "status":
            await self._handle_device_status(device_id, data.get("data", {}))
        elif message_type == "command_response":
            await self._handle_command_response(device_id, data.get("data", {}))
    
    async def _handle_sensor_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """Handle sensor data from device."""
        try:
            sensor_data = SensorData(
                device_id=device_id,
                data_type=DataType(data.get("data_type", "custom")),
                value=data.get("value"),
                unit=data.get("unit", ""),
                quality=data.get("quality", 1.0),
                location=data.get("location"),
                metadata=data.get("metadata", {})
            )
            
            # Store sensor data
            self.sensor_data[device_id].append(sensor_data)
            
            # Update statistics
            self.stats['total_data_points'] += 1
            
            logger.debug(
                "Sensor data received",
                device_id=device_id,
                data_type=sensor_data.data_type.value,
                value=sensor_data.value
            )
        
        except Exception as e:
            logger.error("Sensor data handling error", device_id=device_id, error=str(e))
    
    async def _handle_device_status(self, device_id: str, data: Dict[str, Any]) -> None:
        """Handle device status update."""
        try:
            if device_id in self.devices:
                status = DeviceStatus(data.get("status", "online"))
                self.devices[device_id].status = status
                self.devices[device_id].last_seen = datetime.utcnow()
                
                # Update statistics
                if status == DeviceStatus.ONLINE:
                    self.stats['online_devices'] += 1
                    self.stats['offline_devices'] = max(0, self.stats['offline_devices'] - 1)
                elif status == DeviceStatus.OFFLINE:
                    self.stats['offline_devices'] += 1
                    self.stats['online_devices'] = max(0, self.stats['online_devices'] - 1)
                
                logger.info("Device status updated", device_id=device_id, status=status.value)
        
        except Exception as e:
            logger.error("Device status handling error", device_id=device_id, error=str(e))
    
    async def _handle_command_response(self, device_id: str, data: Dict[str, Any]) -> None:
        """Handle command response from device."""
        try:
            command_id = data.get("command_id")
            success = data.get("success", False)
            response_data = data.get("response")
            
            # Find and update command
            if device_id in self.device_commands:
                for command in self.device_commands[device_id]:
                    if command.command_id == command_id:
                        command.status = "completed" if success else "failed"
                        command.executed_at = datetime.utcnow()
                        
                        # Update statistics
                        if success:
                            self.stats['successful_commands'] += 1
                        else:
                            self.stats['failed_commands'] += 1
                        
                        logger.info(
                            "Command response received",
                            device_id=device_id,
                            command_id=command_id,
                            success=success
                        )
                        break
        
        except Exception as e:
            logger.error("Command response handling error", device_id=device_id, error=str(e))
    
    def register_device(self, device: Device) -> str:
        """Register a new IoT device."""
        self.devices[device.device_id] = device
        self.stats['total_devices'] += 1
        
        if device.status == DeviceStatus.ONLINE:
            self.stats['online_devices'] += 1
        else:
            self.stats['offline_devices'] += 1
        
        logger.info(
            "Device registered",
            device_id=device.device_id,
            name=device.name,
            type=device.device_type.value
        )
        
        return device.device_id
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device."""
        if device_id in self.devices:
            device = self.devices[device_id]
            
            # Update statistics
            self.stats['total_devices'] -= 1
            if device.status == DeviceStatus.ONLINE:
                self.stats['online_devices'] -= 1
            else:
                self.stats['offline_devices'] -= 1
            
            # Clean up
            del self.devices[device_id]
            if device_id in self.sensor_data:
                del self.sensor_data[device_id]
            if device_id in self.device_commands:
                del self.device_commands[device_id]
            if device_id in self.websocket_connections:
                del self.websocket_connections[device_id]
            
            logger.info("Device unregistered", device_id=device_id)
            return True
        
        return False
    
    async def send_command(self, device_id: str, command: DeviceCommand) -> bool:
        """Send command to device."""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        
        # Add command to queue
        self.device_commands[device_id].append(command)
        self.stats['total_commands'] += 1
        
        # Send command based on protocol
        try:
            if device.protocol == CommunicationProtocol.MQTT:
                await self._send_mqtt_command(device_id, command)
            elif device.protocol == CommunicationProtocol.WEBSOCKET:
                await self._send_websocket_command(device_id, command)
            else:
                logger.warning("Unsupported protocol for command", device_id=device_id, protocol=device.protocol.value)
                return False
            
            logger.info(
                "Command sent",
                device_id=device_id,
                command_id=command.command_id,
                command_type=command.command_type
            )
            
            return True
        
        except Exception as e:
            logger.error("Command sending failed", device_id=device_id, command_id=command.command_id, error=str(e))
            return False
    
    async def _send_mqtt_command(self, device_id: str, command: DeviceCommand) -> None:
        """Send command via MQTT."""
        if not self.mqtt_client:
            raise RuntimeError("MQTT client not initialized")
        
        topic = f"devices/{device_id}/command"
        payload = json.dumps(command.to_dict())
        
        # Encrypt payload if encryption is enabled
        if self.cipher_suite:
            payload = self.cipher_suite.encrypt(payload.encode())
        else:
            payload = payload.encode()
        
        await self.mqtt_client.publish(topic, payload)
    
    async def _send_websocket_command(self, device_id: str, command: DeviceCommand) -> None:
        """Send command via WebSocket."""
        if device_id not in self.websocket_connections:
            raise RuntimeError("Device not connected via WebSocket")
        
        websocket = self.websocket_connections[device_id]
        message = json.dumps({
            "type": "command",
            "data": command.to_dict()
        })
        
        await websocket.send(message)
    
    def get_device_data(self, device_id: str, limit: int = 100) -> List[SensorData]:
        """Get sensor data for device."""
        if device_id not in self.sensor_data:
            return []
        
        data_list = list(self.sensor_data[device_id])
        return data_list[-limit:] if limit > 0 else data_list
    
    def get_device_stats(self, device_id: str) -> Dict[str, Any]:
        """Get device statistics."""
        if device_id not in self.devices:
            return {}
        
        device = self.devices[device_id]
        data_count = len(self.sensor_data.get(device_id, []))
        command_count = len(self.device_commands.get(device_id, []))
        
        return {
            'device_id': device_id,
            'name': device.name,
            'type': device.device_type.value,
            'status': device.status.value,
            'protocol': device.protocol.value,
            'data_points': data_count,
            'commands': command_count,
            'last_seen': device.last_seen.isoformat(),
            'uptime': (datetime.utcnow() - device.created_at).total_seconds()
        }
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to check device status."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for device_id, device in list(self.devices.items()):
                    time_since_last_seen = (current_time - device.last_seen).total_seconds()
                    
                    if time_since_last_seen > self.config.device_timeout:
                        if device.status == DeviceStatus.ONLINE:
                            device.status = DeviceStatus.OFFLINE
                            self.stats['online_devices'] -= 1
                            self.stats['offline_devices'] += 1
                            
                            logger.warning(
                                "Device marked as offline",
                                device_id=device_id,
                                time_since_last_seen=time_since_last_seen
                            )
                
                await asyncio.sleep(60)  # Check every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for old data."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(days=self.config.data_retention_days)
                
                # Clean up old sensor data
                for device_id, data_queue in self.sensor_data.items():
                    # Remove old data points
                    while data_queue and data_queue[0].timestamp < cutoff_time:
                        data_queue.popleft()
                
                # Clean up old commands
                for device_id, commands in self.device_commands.items():
                    self.device_commands[device_id] = [
                        cmd for cmd in commands
                        if cmd.created_at > cutoff_time
                    ]
                
                await asyncio.sleep(3600)  # Clean up every hour
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(3600)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            'devices': {
                device_id: self.get_device_stats(device_id)
                for device_id in self.devices
            },
            'websocket_connections': len(self.websocket_connections),
            'mqtt_connected': self.mqtt_client is not None
        }

# =============================================================================
# GLOBAL IOT INSTANCES
# =============================================================================

# Default IoT configuration
default_iot_config = IoTConfig(
    mqtt_broker="localhost",
    mqtt_port=1883,
    mqtt_username=None,
    mqtt_password=None,
    websocket_port=8765,
    device_timeout=300,
    data_retention_days=30,
    encryption_key=None,
    enable_ssl=False,
    max_devices=1000
)

# Global IoT device manager
iot_device_manager = IoTDeviceManager(default_iot_config)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DeviceType',
    'DeviceStatus',
    'CommunicationProtocol',
    'DataType',
    'Device',
    'SensorData',
    'DeviceCommand',
    'IoTConfig',
    'IoTDeviceManager',
    'default_iot_config',
    'iot_device_manager'
]





























