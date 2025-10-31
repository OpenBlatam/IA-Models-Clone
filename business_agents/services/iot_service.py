"""
IoT Service
===========

Advanced IoT integration service for connecting and managing
Internet of Things devices in business workflows.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import asyncio_mqtt
import paho.mqtt.client as mqtt
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib
import hmac
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Types of IoT devices."""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CAMERA = "camera"
    GATEWAY = "gateway"
    SMART_DEVICE = "smart_device"
    INDUSTRIAL = "industrial"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"

class DeviceStatus(Enum):
    """Device status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UNKNOWN = "unknown"

class DataType(Enum):
    """Types of IoT data."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    MOTION = "motion"
    LIGHT = "light"
    SOUND = "sound"
    LOCATION = "location"
    BATTERY = "battery"
    CUSTOM = "custom"

class Protocol(Enum):
    """Communication protocols."""
    MQTT = "mqtt"
    HTTP = "http"
    WEBSOCKET = "websocket"
    COAP = "coap"
    MODBUS = "modbus"
    OPCUA = "opcua"

@dataclass
class IoTDevice:
    """IoT device definition."""
    device_id: str
    name: str
    device_type: DeviceType
    protocol: Protocol
    status: DeviceStatus
    location: Dict[str, float]
    capabilities: List[str]
    configuration: Dict[str, Any]
    last_seen: datetime
    metadata: Dict[str, Any]
    encryption_key: Optional[str] = None

@dataclass
class IoTData:
    """IoT data point."""
    data_id: str
    device_id: str
    data_type: DataType
    value: Union[float, int, str, bool, Dict[str, Any]]
    unit: str
    timestamp: datetime
    quality: float
    metadata: Dict[str, Any]

@dataclass
class IoTCommand:
    """IoT command."""
    command_id: str
    device_id: str
    command_type: str
    parameters: Dict[str, Any]
    timestamp: datetime
    status: str
    response: Optional[Dict[str, Any]] = None

@dataclass
class IoTAlert:
    """IoT alert."""
    alert_id: str
    device_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    acknowledged: bool
    resolved: bool
    metadata: Dict[str, Any]

class IoTService:
    """
    Advanced IoT integration service.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.devices = {}
        self.data_streams = {}
        self.commands = {}
        self.alerts = {}
        self.mqtt_client = None
        self.websocket_connections = {}
        self.data_handlers = {}
        self.alert_handlers = {}
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # IoT configurations
        self.mqtt_config = config.get("mqtt", {
            "broker": "localhost",
            "port": 1883,
            "username": None,
            "password": None,
            "keepalive": 60
        })
        
        self.websocket_config = config.get("websocket", {
            "host": "localhost",
            "port": 8080,
            "path": "/ws"
        })
        
    async def initialize(self):
        """Initialize the IoT service."""
        try:
            await self._initialize_mqtt()
            await self._initialize_websocket()
            await self._load_default_devices()
            await self._start_data_collection()
            logger.info("IoT Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IoT Service: {str(e)}")
            raise
            
    async def _initialize_mqtt(self):
        """Initialize MQTT client."""
        try:
            self.mqtt_client = mqtt.Client()
            
            if self.mqtt_config.get("username"):
                self.mqtt_client.username_pw_set(
                    self.mqtt_config["username"],
                    self.mqtt_config["password"]
                )
                
            # Set callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Connect to broker
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.mqtt_client.connect,
                self.mqtt_config["broker"],
                self.mqtt_config["port"],
                self.mqtt_config["keepalive"]
            )
            
            self.mqtt_client.loop_start()
            logger.info("MQTT client initialized and connected")
            
        except Exception as e:
            logger.error(f"Failed to initialize MQTT: {str(e)}")
            
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info("MQTT client connected successfully")
            # Subscribe to device topics
            client.subscribe("devices/+/data")
            client.subscribe("devices/+/status")
            client.subscribe("devices/+/alerts")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Extract device ID from topic
            device_id = topic.split('/')[1]
            
            # Handle different message types
            if topic.endswith('/data'):
                asyncio.create_task(self._handle_device_data(device_id, payload))
            elif topic.endswith('/status'):
                asyncio.create_task(self._handle_device_status(device_id, payload))
            elif topic.endswith('/alerts'):
                asyncio.create_task(self._handle_device_alert(device_id, payload))
                
        except Exception as e:
            logger.error(f"Failed to handle MQTT message: {str(e)}")
            
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        logger.warning(f"MQTT client disconnected with code {rc}")
        
    async def _handle_device_data(self, device_id: str, data: Dict[str, Any]):
        """Handle device data from MQTT."""
        try:
            # Create IoT data point
            iot_data = IoTData(
                data_id=f"data_{device_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                device_id=device_id,
                data_type=DataType(data.get("type", "custom")),
                value=data.get("value"),
                unit=data.get("unit", ""),
                timestamp=datetime.utcnow(),
                quality=data.get("quality", 1.0),
                metadata=data.get("metadata", {})
            )
            
            # Store data
            if device_id not in self.data_streams:
                self.data_streams[device_id] = []
            self.data_streams[device_id].append(iot_data)
            
            # Keep only last 1000 data points per device
            if len(self.data_streams[device_id]) > 1000:
                self.data_streams[device_id] = self.data_streams[device_id][-1000:]
                
            # Call data handlers
            if device_id in self.data_handlers:
                for handler in self.data_handlers[device_id]:
                    try:
                        await handler(iot_data)
                    except Exception as e:
                        logger.error(f"Data handler error: {str(e)}")
                        
            logger.debug(f"Processed data from device {device_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle device data: {str(e)}")
            
    async def _handle_device_status(self, device_id: str, status_data: Dict[str, Any]):
        """Handle device status from MQTT."""
        try:
            if device_id in self.devices:
                self.devices[device_id].status = DeviceStatus(status_data.get("status", "unknown"))
                self.devices[device_id].last_seen = datetime.utcnow()
                
            logger.debug(f"Updated status for device {device_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle device status: {str(e)}")
            
    async def _handle_device_alert(self, device_id: str, alert_data: Dict[str, Any]):
        """Handle device alert from MQTT."""
        try:
            # Create IoT alert
            alert = IoTAlert(
                alert_id=f"alert_{device_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                device_id=device_id,
                alert_type=alert_data.get("type", "general"),
                severity=alert_data.get("severity", "medium"),
                message=alert_data.get("message", "Device alert"),
                timestamp=datetime.utcnow(),
                acknowledged=False,
                resolved=False,
                metadata=alert_data.get("metadata", {})
            )
            
            # Store alert
            self.alerts[alert.alert_id] = alert
            
            # Call alert handlers
            if device_id in self.alert_handlers:
                for handler in self.alert_handlers[device_id]:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {str(e)}")
                        
            logger.warning(f"Received alert from device {device_id}: {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to handle device alert: {str(e)}")
            
    async def _initialize_websocket(self):
        """Initialize WebSocket server."""
        try:
            # This would initialize a WebSocket server
            # For now, just log the configuration
            logger.info(f"WebSocket server configured: {self.websocket_config}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {str(e)}")
            
    async def _load_default_devices(self):
        """Load default IoT devices."""
        try:
            # Create sample devices
            devices = [
                IoTDevice(
                    device_id="sensor_temp_001",
                    name="Temperature Sensor 001",
                    device_type=DeviceType.SENSOR,
                    protocol=Protocol.MQTT,
                    status=DeviceStatus.ONLINE,
                    location={"lat": 40.7128, "lon": -74.0060},
                    capabilities=["temperature_reading", "humidity_reading"],
                    configuration={"sampling_rate": 30, "threshold": 25.0},
                    last_seen=datetime.utcnow(),
                    metadata={"manufacturer": "IoT Corp", "model": "TempSensor v2.0"}
                ),
                IoTDevice(
                    device_id="actuator_light_001",
                    name="Smart Light 001",
                    device_type=DeviceType.ACTUATOR,
                    protocol=Protocol.MQTT,
                    status=DeviceStatus.ONLINE,
                    location={"lat": 40.7128, "lon": -74.0060},
                    capabilities=["light_control", "brightness_control", "color_control"],
                    configuration={"brightness": 80, "color": "#FFFFFF"},
                    last_seen=datetime.utcnow(),
                    metadata={"manufacturer": "SmartHome Inc", "model": "SmartLight Pro"}
                ),
                IoTDevice(
                    device_id="camera_security_001",
                    name="Security Camera 001",
                    device_type=DeviceType.CAMERA,
                    protocol=Protocol.HTTP,
                    status=DeviceStatus.ONLINE,
                    location={"lat": 40.7128, "lon": -74.0060},
                    capabilities=["video_streaming", "motion_detection", "night_vision"],
                    configuration={"resolution": "1080p", "fps": 30, "night_vision": True},
                    last_seen=datetime.utcnow(),
                    metadata={"manufacturer": "SecurityTech", "model": "CamPro 4K"}
                )
            ]
            
            for device in devices:
                self.devices[device.device_id] = device
                
            logger.info(f"Loaded {len(devices)} default IoT devices")
            
        except Exception as e:
            logger.error(f"Failed to load default devices: {str(e)}")
            
    async def _start_data_collection(self):
        """Start data collection from devices."""
        try:
            # Start background task for data collection
            asyncio.create_task(self._collect_device_data())
            logger.info("Started IoT data collection")
            
        except Exception as e:
            logger.error(f"Failed to start data collection: {str(e)}")
            
    async def _collect_device_data(self):
        """Collect data from IoT devices."""
        while True:
            try:
                # Simulate data collection from devices
                for device_id, device in self.devices.items():
                    if device.status == DeviceStatus.ONLINE:
                        # Generate sample data based on device type
                        if device.device_type == DeviceType.SENSOR:
                            await self._generate_sensor_data(device)
                        elif device.device_type == DeviceType.CAMERA:
                            await self._generate_camera_data(device)
                            
                # Wait before next collection cycle
                await asyncio.sleep(30)  # Collect data every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in data collection: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _generate_sensor_data(self, device: IoTDevice):
        """Generate sample sensor data."""
        try:
            import random
            
            # Generate temperature data
            if "temperature_reading" in device.capabilities:
                temperature = 20 + random.uniform(-5, 15)  # 15-35°C
                data = IoTData(
                    data_id=f"data_{device.device_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    device_id=device.device_id,
                    data_type=DataType.TEMPERATURE,
                    value=temperature,
                    unit="°C",
                    timestamp=datetime.utcnow(),
                    quality=0.95 + random.uniform(0, 0.05),
                    metadata={"sensor_type": "temperature"}
                )
                
                # Store data
                if device.device_id not in self.data_streams:
                    self.data_streams[device.device_id] = []
                self.data_streams[device.device_id].append(data)
                
            # Generate humidity data
            if "humidity_reading" in device.capabilities:
                humidity = 40 + random.uniform(-10, 30)  # 30-70%
                data = IoTData(
                    data_id=f"data_{device.device_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    device_id=device.device_id,
                    data_type=DataType.HUMIDITY,
                    value=humidity,
                    unit="%",
                    timestamp=datetime.utcnow(),
                    quality=0.95 + random.uniform(0, 0.05),
                    metadata={"sensor_type": "humidity"}
                )
                
                # Store data
                if device.device_id not in self.data_streams:
                    self.data_streams[device.device_id] = []
                self.data_streams[device.device_id].append(data)
                
        except Exception as e:
            logger.error(f"Failed to generate sensor data: {str(e)}")
            
    async def _generate_camera_data(self, device: IoTDevice):
        """Generate sample camera data."""
        try:
            import random
            
            # Generate motion detection data
            if "motion_detection" in device.capabilities:
                motion_detected = random.random() < 0.1  # 10% chance of motion
                if motion_detected:
                    data = IoTData(
                        data_id=f"data_{device.device_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        device_id=device.device_id,
                        data_type=DataType.MOTION,
                        value=True,
                        unit="boolean",
                        timestamp=datetime.utcnow(),
                        quality=0.98,
                        metadata={"motion_confidence": 0.85 + random.uniform(0, 0.15)}
                    )
                    
                    # Store data
                    if device.device_id not in self.data_streams:
                        self.data_streams[device.device_id] = []
                    self.data_streams[device.device_id].append(data)
                    
        except Exception as e:
            logger.error(f"Failed to generate camera data: {str(e)}")
            
    async def register_device(self, device: IoTDevice) -> str:
        """Register a new IoT device."""
        try:
            # Generate device ID if not provided
            if not device.device_id:
                device.device_id = f"device_{uuid.uuid4().hex[:8]}"
                
            # Set encryption key
            device.encryption_key = base64.b64encode(self.encryption_key).decode()
            
            # Register device
            self.devices[device.device_id] = device
            
            # Initialize data stream
            self.data_streams[device.device_id] = []
            
            logger.info(f"Registered IoT device: {device.device_id}")
            
            return device.device_id
            
        except Exception as e:
            logger.error(f"Failed to register device: {str(e)}")
            raise
            
    async def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device."""
        try:
            if device_id in self.devices:
                del self.devices[device_id]
                
            if device_id in self.data_streams:
                del self.data_streams[device_id]
                
            logger.info(f"Unregistered IoT device: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister device: {str(e)}")
            return False
            
    async def get_device(self, device_id: str) -> Optional[IoTDevice]:
        """Get IoT device by ID."""
        return self.devices.get(device_id)
        
    async def get_devices(self, device_type: Optional[DeviceType] = None) -> List[IoTDevice]:
        """Get IoT devices."""
        devices = list(self.devices.values())
        
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
            
        return devices
        
    async def get_device_data(
        self, 
        device_id: str, 
        data_type: Optional[DataType] = None,
        limit: int = 100
    ) -> List[IoTData]:
        """Get device data."""
        if device_id not in self.data_streams:
            return []
            
        data = self.data_streams[device_id]
        
        if data_type:
            data = [d for d in data if d.data_type == data_type]
            
        return data[-limit:] if limit else data
        
    async def send_command(self, device_id: str, command: IoTCommand) -> bool:
        """Send command to IoT device."""
        try:
            if device_id not in self.devices:
                raise ValueError(f"Device {device_id} not found")
                
            device = self.devices[device_id]
            
            # Store command
            self.commands[command.command_id] = command
            
            # Send command based on protocol
            if device.protocol == Protocol.MQTT:
                await self._send_mqtt_command(device_id, command)
            elif device.protocol == Protocol.HTTP:
                await self._send_http_command(device_id, command)
            elif device.protocol == Protocol.WEBSOCKET:
                await self._send_websocket_command(device_id, command)
                
            logger.info(f"Sent command {command.command_id} to device {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command: {str(e)}")
            return False
            
    async def _send_mqtt_command(self, device_id: str, command: IoTCommand):
        """Send command via MQTT."""
        try:
            topic = f"devices/{device_id}/commands"
            payload = json.dumps({
                "command_id": command.command_id,
                "command_type": command.command_type,
                "parameters": command.parameters,
                "timestamp": command.timestamp.isoformat()
            })
            
            self.mqtt_client.publish(topic, payload)
            
        except Exception as e:
            logger.error(f"Failed to send MQTT command: {str(e)}")
            
    async def _send_http_command(self, device_id: str, command: IoTCommand):
        """Send command via HTTP."""
        try:
            device = self.devices[device_id]
            url = f"http://{device.configuration.get('ip_address', 'localhost')}/api/commands"
            
            payload = {
                "command_id": command.command_id,
                "command_type": command.command_type,
                "parameters": command.parameters,
                "timestamp": command.timestamp.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        command.status = "sent"
                        command.response = await response.json()
                    else:
                        command.status = "failed"
                        
        except Exception as e:
            logger.error(f"Failed to send HTTP command: {str(e)}")
            command.status = "failed"
            
    async def _send_websocket_command(self, device_id: str, command: IoTCommand):
        """Send command via WebSocket."""
        try:
            if device_id in self.websocket_connections:
                ws = self.websocket_connections[device_id]
                payload = {
                    "command_id": command.command_id,
                    "command_type": command.command_type,
                    "parameters": command.parameters,
                    "timestamp": command.timestamp.isoformat()
                }
                await ws.send_str(json.dumps(payload))
                command.status = "sent"
            else:
                command.status = "failed"
                
        except Exception as e:
            logger.error(f"Failed to send WebSocket command: {str(e)}")
            command.status = "failed"
            
    async def add_data_handler(self, device_id: str, handler: Callable):
        """Add data handler for device."""
        if device_id not in self.data_handlers:
            self.data_handlers[device_id] = []
        self.data_handlers[device_id].append(handler)
        
    async def add_alert_handler(self, device_id: str, handler: Callable):
        """Add alert handler for device."""
        if device_id not in self.alert_handlers:
            self.alert_handlers[device_id] = []
        self.alert_handlers[device_id].append(handler)
        
    async def get_alerts(
        self, 
        device_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[IoTAlert]:
        """Get IoT alerts."""
        alerts = list(self.alerts.values())
        
        if device_id:
            alerts = [a for a in alerts if a.device_id == device_id]
            
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
            
        return alerts[-limit:] if limit else alerts
        
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.alerts:
                self.alerts[alert_id].acknowledged = True
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {str(e)}")
            return False
            
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {str(e)}")
            return False
            
    async def get_service_status(self) -> Dict[str, Any]:
        """Get IoT service status."""
        try:
            online_devices = len([d for d in self.devices.values() if d.status == DeviceStatus.ONLINE])
            total_data_points = sum(len(stream) for stream in self.data_streams.values())
            active_alerts = len([a for a in self.alerts.values() if not a.resolved])
            
            return {
                "service_status": "active",
                "total_devices": len(self.devices),
                "online_devices": online_devices,
                "offline_devices": len(self.devices) - online_devices,
                "total_data_points": total_data_points,
                "active_alerts": active_alerts,
                "mqtt_connected": self.mqtt_client.is_connected() if self.mqtt_client else False,
                "websocket_connections": len(self.websocket_connections),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get service status: {str(e)}")
            return {"service_status": "error", "error": str(e)}
            
    async def encrypt_data(self, data: str) -> str:
        """Encrypt IoT data."""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {str(e)}")
            return data
            
    async def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt IoT data."""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {str(e)}")
            return encrypted_data




























