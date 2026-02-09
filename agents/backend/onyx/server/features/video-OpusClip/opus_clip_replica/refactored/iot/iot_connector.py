"""
IoT Integration for Opus Clip

Advanced IoT capabilities with:
- Device management and discovery
- Real-time data processing
- Multi-protocol support (MQTT, CoAP, HTTP)
- Edge device integration
- Sensor data processing
- Automated content generation
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Tuple
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import aiohttp
import paho.mqtt.client as mqtt
import asyncio_mqtt
import aiocoap
from aiocoap import Context, Message, Code
import numpy as np
from pathlib import Path
import cv2
import base64
import hashlib

logger = structlog.get_logger("iot_connector")

class DeviceType(Enum):
    """IoT device type enumeration."""
    CAMERA = "camera"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    DISPLAY = "display"
    AUDIO_DEVICE = "audio_device"
    HAPTIC_DEVICE = "haptic_device"
    GATEWAY = "gateway"

class Protocol(Enum):
    """Communication protocol enumeration."""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    WEBSOCKET = "websocket"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"

class DataType(Enum):
    """Data type enumeration."""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    SENSOR_DATA = "sensor_data"
    CONTROL_COMMAND = "control_command"
    STATUS_UPDATE = "status_update"

@dataclass
class IoTDevice:
    """IoT device information."""
    device_id: str
    name: str
    device_type: DeviceType
    protocol: Protocol
    endpoint: str
    capabilities: List[str]
    status: str = "offline"
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SensorData:
    """Sensor data structure."""
    device_id: str
    data_type: DataType
    timestamp: datetime
    value: Any
    unit: Optional[str] = None
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IoTMessage:
    """IoT message structure."""
    message_id: str
    device_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    qos: int = 0
    retain: bool = False

class IoTDeviceManager:
    """IoT device management system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("iot_device_manager")
        self.devices: Dict[str, IoTDevice] = {}
        self.device_topics: Dict[str, str] = {}
        self.message_handlers: Dict[str, callable] = {}
        
    async def register_device(self, device: IoTDevice) -> bool:
        """Register an IoT device."""
        try:
            self.devices[device.device_id] = device
            device.status = "online"
            device.last_seen = datetime.now()
            
            # Create device topic
            topic = f"devices/{device.device_id}"
            self.device_topics[device.device_id] = topic
            
            self.logger.info(f"Registered IoT device: {device.name} ({device.device_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register device {device.device_id}: {e}")
            return False
    
    async def discover_devices(self, protocol: Protocol = None) -> List[IoTDevice]:
        """Discover IoT devices."""
        try:
            discovered_devices = []
            
            if protocol is None or protocol == Protocol.MQTT:
                mqtt_devices = await self._discover_mqtt_devices()
                discovered_devices.extend(mqtt_devices)
            
            if protocol is None or protocol == Protocol.COAP:
                coap_devices = await self._discover_coap_devices()
                discovered_devices.extend(coap_devices)
            
            if protocol is None or protocol == Protocol.HTTP:
                http_devices = await self._discover_http_devices()
                discovered_devices.extend(http_devices)
            
            # Register discovered devices
            for device in discovered_devices:
                await self.register_device(device)
            
            return discovered_devices
            
        except Exception as e:
            self.logger.error(f"Device discovery failed: {e}")
            return []
    
    async def _discover_mqtt_devices(self) -> List[IoTDevice]:
        """Discover MQTT devices."""
        # Simulate MQTT device discovery
        devices = [
            IoTDevice(
                device_id="camera_001",
                name="Security Camera 1",
                device_type=DeviceType.CAMERA,
                protocol=Protocol.MQTT,
                endpoint="mqtt://192.168.1.100:1883",
                capabilities=["video_stream", "motion_detection", "night_vision"]
            ),
            IoTDevice(
                device_id="sensor_001",
                name="Temperature Sensor",
                device_type=DeviceType.SENSOR,
                protocol=Protocol.MQTT,
                endpoint="mqtt://192.168.1.101:1883",
                capabilities=["temperature_reading", "humidity_reading"]
            )
        ]
        return devices
    
    async def _discover_coap_devices(self) -> List[IoTDevice]:
        """Discover CoAP devices."""
        # Simulate CoAP device discovery
        devices = [
            IoTDevice(
                device_id="sensor_002",
                name="Light Sensor",
                device_type=DeviceType.SENSOR,
                protocol=Protocol.COAP,
                endpoint="coap://192.168.1.102:5683",
                capabilities=["light_reading", "motion_detection"]
            )
        ]
        return devices
    
    async def _discover_http_devices(self) -> List[IoTDevice]:
        """Discover HTTP devices."""
        # Simulate HTTP device discovery
        devices = [
            IoTDevice(
                device_id="display_001",
                name="Smart Display",
                device_type=DeviceType.DISPLAY,
                protocol=Protocol.HTTP,
                endpoint="http://192.168.1.103:8080",
                capabilities=["video_display", "touch_input", "audio_output"]
            )
        ]
        return devices
    
    async def get_device(self, device_id: str) -> Optional[IoTDevice]:
        """Get device by ID."""
        return self.devices.get(device_id)
    
    async def update_device_status(self, device_id: str, status: str):
        """Update device status."""
        if device_id in self.devices:
            self.devices[device_id].status = status
            self.devices[device_id].last_seen = datetime.now()
    
    async def get_online_devices(self) -> List[IoTDevice]:
        """Get all online devices."""
        return [device for device in self.devices.values() if device.status == "online"]

class MQTTConnector:
    """MQTT communication connector."""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = None
        self.logger = structlog.get_logger("mqtt_connector")
        self.message_handlers: Dict[str, callable] = {}
        
    async def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            self.client = asyncio_mqtt.Client(hostname=self.broker_host, port=self.broker_port)
            await self.client.connect()
            self.logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"MQTT connection failed: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: callable):
        """Subscribe to MQTT topic."""
        try:
            await self.client.subscribe(topic)
            self.message_handlers[topic] = handler
            self.logger.info(f"Subscribed to MQTT topic: {topic}")
            
        except Exception as e:
            self.logger.error(f"MQTT subscription failed: {e}")
    
    async def publish(self, topic: str, message: str, qos: int = 0, retain: bool = False):
        """Publish message to MQTT topic."""
        try:
            await self.client.publish(topic, message, qos=qos, retain=retain)
            self.logger.info(f"Published message to MQTT topic: {topic}")
            
        except Exception as e:
            self.logger.error(f"MQTT publish failed: {e}")
    
    async def start_listening(self):
        """Start listening for MQTT messages."""
        try:
            async with self.client.messages() as messages:
                async for message in messages:
                    topic = message.topic.value
                    payload = message.payload.decode()
                    
                    if topic in self.message_handlers:
                        await self.message_handlers[topic](topic, payload)
                    
        except Exception as e:
            self.logger.error(f"MQTT listening failed: {e}")
    
    async def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            await self.client.disconnect()

class CoAPConnector:
    """CoAP communication connector."""
    
    def __init__(self):
        self.logger = structlog.get_logger("coap_connector")
        self.context = None
        
    async def connect(self) -> bool:
        """Connect to CoAP context."""
        try:
            self.context = await Context.create_client_context()
            self.logger.info("Connected to CoAP context")
            return True
            
        except Exception as e:
            self.logger.error(f"CoAP connection failed: {e}")
            return False
    
    async def get(self, uri: str) -> Optional[Dict[str, Any]]:
        """Send GET request to CoAP resource."""
        try:
            request = Message(code=Code.GET, uri=uri)
            response = await self.context.request(request).response
            
            if response.code.is_successful():
                data = json.loads(response.payload.decode())
                return data
            else:
                self.logger.error(f"CoAP GET failed: {response.code}")
                return None
                
        except Exception as e:
            self.logger.error(f"CoAP GET request failed: {e}")
            return None
    
    async def post(self, uri: str, data: Dict[str, Any]) -> bool:
        """Send POST request to CoAP resource."""
        try:
            payload = json.dumps(data).encode()
            request = Message(code=Code.POST, uri=uri, payload=payload)
            response = await self.context.request(request).response
            
            if response.code.is_successful():
                self.logger.info(f"CoAP POST successful: {uri}")
                return True
            else:
                self.logger.error(f"CoAP POST failed: {response.code}")
                return False
                
        except Exception as e:
            self.logger.error(f"CoAP POST request failed: {e}")
            return False

class HTTPConnector:
    """HTTP communication connector."""
    
    def __init__(self):
        self.logger = structlog.get_logger("http_connector")
        self.session = None
        
    async def connect(self) -> bool:
        """Create HTTP session."""
        try:
            self.session = aiohttp.ClientSession()
            self.logger.info("Created HTTP session")
            return True
            
        except Exception as e:
            self.logger.error(f"HTTP session creation failed: {e}")
            return False
    
    async def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Send GET request."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.error(f"HTTP GET failed: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"HTTP GET request failed: {e}")
            return None
    
    async def post(self, url: str, data: Dict[str, Any]) -> bool:
        """Send POST request."""
        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    self.logger.info(f"HTTP POST successful: {url}")
                    return True
                else:
                    self.logger.error(f"HTTP POST failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"HTTP POST request failed: {e}")
            return False
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()

class IoTDataProcessor:
    """IoT data processing system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("iot_data_processor")
        self.data_buffer: Dict[str, List[SensorData]] = {}
        self.processing_rules: Dict[str, callable] = {}
        
    async def process_sensor_data(self, data: SensorData) -> Dict[str, Any]:
        """Process sensor data."""
        try:
            # Add to buffer
            if data.device_id not in self.data_buffer:
                self.data_buffer[data.device_id] = []
            
            self.data_buffer[data.device_id].append(data)
            
            # Keep only last 1000 data points
            if len(self.data_buffer[data.device_id]) > 1000:
                self.data_buffer[data.device_id] = self.data_buffer[data.device_id][-1000:]
            
            # Process based on data type
            if data.data_type == DataType.VIDEO:
                result = await self._process_video_data(data)
            elif data.data_type == DataType.AUDIO:
                result = await self._process_audio_data(data)
            elif data.data_type == DataType.IMAGE:
                result = await self._process_image_data(data)
            elif data.data_type == DataType.SENSOR_DATA:
                result = await self._process_sensor_data(data)
            else:
                result = {"processed": True, "data_type": data.data_type.value}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sensor data processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_video_data(self, data: SensorData) -> Dict[str, Any]:
        """Process video data from IoT device."""
        try:
            # Decode base64 video data
            video_data = base64.b64decode(data.value)
            
            # Save to temporary file
            temp_path = f"/tmp/iot_video_{uuid.uuid4()}.mp4"
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            # Process video
            cap = cv2.VideoCapture(temp_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Analyze video content
            analysis = {
                "frame_count": frame_count,
                "fps": fps,
                "duration": frame_count / fps if fps > 0 else 0,
                "resolution": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                "motion_detected": await self._detect_motion(temp_path),
                "quality_score": await self._calculate_quality_score(temp_path)
            }
            
            cap.release()
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Video data processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_audio_data(self, data: SensorData) -> Dict[str, Any]:
        """Process audio data from IoT device."""
        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(data.value)
            
            # Save to temporary file
            temp_path = f"/tmp/iot_audio_{uuid.uuid4()}.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # Process audio
            import librosa
            audio, sr = librosa.load(temp_path, sr=None)
            
            # Analyze audio content
            analysis = {
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "rms_energy": float(np.sqrt(np.mean(audio**2))),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
                "loudness": float(np.mean(librosa.feature.rms(y=audio)))
            }
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Audio data processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_image_data(self, data: SensorData) -> Dict[str, Any]:
        """Process image data from IoT device."""
        try:
            # Decode base64 image data
            image_data = base64.b64decode(data.value)
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Failed to decode image"}
            
            # Analyze image
            analysis = {
                "resolution": image.shape[:2],
                "brightness": float(np.mean(image)),
                "contrast": float(np.std(image)),
                "color_histogram": {
                    "red": np.mean(image[:, :, 2]),
                    "green": np.mean(image[:, :, 1]),
                    "blue": np.mean(image[:, :, 0])
                },
                "edge_density": await self._calculate_edge_density(image),
                "blur_score": await self._calculate_blur_score(image)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Image data processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_sensor_data(self, data: SensorData) -> Dict[str, Any]:
        """Process sensor data from IoT device."""
        try:
            # Basic sensor data processing
            analysis = {
                "value": data.value,
                "unit": data.unit,
                "quality": data.quality,
                "timestamp": data.timestamp.isoformat(),
                "trend": await self._calculate_trend(data.device_id, data.value),
                "anomaly_detected": await self._detect_anomaly(data.device_id, data.value)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Sensor data processing failed: {e}")
            return {"error": str(e)}
    
    async def _detect_motion(self, video_path: str) -> bool:
        """Detect motion in video."""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, prev_frame = cap.read()
            if not ret:
                return False
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            motion_detected = False
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
                
                if np.sum(thresh) > 1000:  # Motion threshold
                    motion_detected = True
                    break
                
                prev_gray = gray
            
            cap.release()
            return motion_detected
            
        except Exception as e:
            self.logger.error(f"Motion detection failed: {e}")
            return False
    
    async def _calculate_quality_score(self, video_path: str) -> float:
        """Calculate video quality score."""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                return 0.0
            
            # Calculate Laplacian variance (blur detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale
            quality_score = min(laplacian_var / 1000, 1.0)
            
            cap.release()
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    async def _calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density in image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
            return float(edge_density)
            
        except Exception as e:
            self.logger.error(f"Edge density calculation failed: {e}")
            return 0.0
    
    async def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate blur score for image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = 1.0 / (1.0 + laplacian_var / 1000)
            return float(blur_score)
            
        except Exception as e:
            self.logger.error(f"Blur score calculation failed: {e}")
            return 0.0
    
    async def _calculate_trend(self, device_id: str, value: float) -> str:
        """Calculate trend for sensor data."""
        try:
            if device_id not in self.data_buffer or len(self.data_buffer[device_id]) < 2:
                return "unknown"
            
            recent_data = self.data_buffer[device_id][-10:]
            values = [d.value for d in recent_data if isinstance(d.value, (int, float))]
            
            if len(values) < 2:
                return "unknown"
            
            # Simple trend calculation
            if values[-1] > values[0] * 1.1:
                return "increasing"
            elif values[-1] < values[0] * 0.9:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error(f"Trend calculation failed: {e}")
            return "unknown"
    
    async def _detect_anomaly(self, device_id: str, value: float) -> bool:
        """Detect anomaly in sensor data."""
        try:
            if device_id not in self.data_buffer or len(self.data_buffer[device_id]) < 10:
                return False
            
            recent_data = self.data_buffer[device_id][-20:]
            values = [d.value for d in recent_data if isinstance(d.value, (int, float))]
            
            if len(values) < 10:
                return False
            
            # Simple anomaly detection using z-score
            mean = np.mean(values[:-1])
            std = np.std(values[:-1])
            
            if std == 0:
                return False
            
            z_score = abs((value - mean) / std)
            return z_score > 2.0  # Threshold for anomaly
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return False

class IoTOrchestrator:
    """Main IoT orchestrator system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("iot_orchestrator")
        self.device_manager = IoTDeviceManager()
        self.mqtt_connector = MQTTConnector()
        self.coap_connector = CoAPConnector()
        self.http_connector = HTTPConnector()
        self.data_processor = IoTDataProcessor()
        
        # Message handlers
        self.message_handlers = {
            "video_data": self._handle_video_data,
            "audio_data": self._handle_audio_data,
            "sensor_data": self._handle_sensor_data,
            "control_command": self._handle_control_command
        }
    
    async def initialize(self) -> bool:
        """Initialize IoT orchestrator."""
        try:
            # Connect to all protocols
            mqtt_connected = await self.mqtt_connector.connect()
            coap_connected = await self.coap_connector.connect()
            http_connected = await self.http_connector.connect()
            
            if not (mqtt_connected or coap_connected or http_connected):
                self.logger.error("Failed to connect to any IoT protocol")
                return False
            
            # Discover devices
            await self.device_manager.discover_devices()
            
            # Set up message handlers
            await self._setup_message_handlers()
            
            self.logger.info("IoT orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"IoT orchestrator initialization failed: {e}")
            return False
    
    async def _setup_message_handlers(self):
        """Set up message handlers for different protocols."""
        # MQTT handlers
        await self.mqtt_connector.subscribe("devices/+/video", self._handle_mqtt_message)
        await self.mqtt_connector.subscribe("devices/+/audio", self._handle_mqtt_message)
        await self.mqtt_connector.subscribe("devices/+/sensor", self._handle_mqtt_message)
        await self.mqtt_connector.subscribe("devices/+/control", self._handle_mqtt_message)
    
    async def _handle_mqtt_message(self, topic: str, payload: str):
        """Handle MQTT message."""
        try:
            # Parse topic to get device ID and message type
            topic_parts = topic.split('/')
            device_id = topic_parts[1]
            message_type = topic_parts[2]
            
            # Parse payload
            data = json.loads(payload)
            
            # Create IoT message
            message = IoTMessage(
                message_id=str(uuid.uuid4()),
                device_id=device_id,
                message_type=message_type,
                payload=data,
                timestamp=datetime.now()
            )
            
            # Process message
            await self._process_message(message)
            
        except Exception as e:
            self.logger.error(f"MQTT message handling failed: {e}")
    
    async def _process_message(self, message: IoTMessage):
        """Process IoT message."""
        try:
            message_type = message.message_type
            
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](message)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Message processing failed: {e}")
    
    async def _handle_video_data(self, message: IoTMessage):
        """Handle video data from IoT device."""
        try:
            # Create sensor data
            sensor_data = SensorData(
                device_id=message.device_id,
                data_type=DataType.VIDEO,
                timestamp=message.timestamp,
                value=message.payload.get("data"),
                metadata=message.payload.get("metadata", {})
            )
            
            # Process video data
            result = await self.data_processor.process_sensor_data(sensor_data)
            
            self.logger.info(f"Processed video data from device {message.device_id}")
            
        except Exception as e:
            self.logger.error(f"Video data handling failed: {e}")
    
    async def _handle_audio_data(self, message: IoTMessage):
        """Handle audio data from IoT device."""
        try:
            # Create sensor data
            sensor_data = SensorData(
                device_id=message.device_id,
                data_type=DataType.AUDIO,
                timestamp=message.timestamp,
                value=message.payload.get("data"),
                metadata=message.payload.get("metadata", {})
            )
            
            # Process audio data
            result = await self.data_processor.process_sensor_data(sensor_data)
            
            self.logger.info(f"Processed audio data from device {message.device_id}")
            
        except Exception as e:
            self.logger.error(f"Audio data handling failed: {e}")
    
    async def _handle_sensor_data(self, message: IoTMessage):
        """Handle sensor data from IoT device."""
        try:
            # Create sensor data
            sensor_data = SensorData(
                device_id=message.device_id,
                data_type=DataType.SENSOR_DATA,
                timestamp=message.timestamp,
                value=message.payload.get("value"),
                unit=message.payload.get("unit"),
                quality=message.payload.get("quality", 1.0),
                metadata=message.payload.get("metadata", {})
            )
            
            # Process sensor data
            result = await self.data_processor.process_sensor_data(sensor_data)
            
            self.logger.info(f"Processed sensor data from device {message.device_id}")
            
        except Exception as e:
            self.logger.error(f"Sensor data handling failed: {e}")
    
    async def _handle_control_command(self, message: IoTMessage):
        """Handle control command for IoT device."""
        try:
            device_id = message.device_id
            command = message.payload.get("command")
            parameters = message.payload.get("parameters", {})
            
            # Send control command to device
            await self._send_control_command(device_id, command, parameters)
            
            self.logger.info(f"Sent control command to device {device_id}")
            
        except Exception as e:
            self.logger.error(f"Control command handling failed: {e}")
    
    async def _send_control_command(self, device_id: str, command: str, parameters: Dict[str, Any]):
        """Send control command to IoT device."""
        try:
            device = await self.device_manager.get_device(device_id)
            if not device:
                self.logger.error(f"Device {device_id} not found")
                return
            
            # Send command based on device protocol
            if device.protocol == Protocol.MQTT:
                topic = f"devices/{device_id}/control"
                message = json.dumps({"command": command, "parameters": parameters})
                await self.mqtt_connector.publish(topic, message)
                
            elif device.protocol == Protocol.HTTP:
                url = f"{device.endpoint}/control"
                data = {"command": command, "parameters": parameters}
                await self.http_connector.post(url, data)
                
            elif device.protocol == Protocol.COAP:
                uri = f"{device.endpoint}/control"
                data = {"command": command, "parameters": parameters}
                await self.coap_connector.post(uri, data)
                
        except Exception as e:
            self.logger.error(f"Control command sending failed: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get IoT system status."""
        try:
            online_devices = await self.device_manager.get_online_devices()
            
            return {
                "total_devices": len(self.device_manager.devices),
                "online_devices": len(online_devices),
                "protocols": {
                    "mqtt": "connected" if self.mqtt_connector.client else "disconnected",
                    "coap": "connected" if self.coap_connector.context else "disconnected",
                    "http": "connected" if self.http_connector.session else "disconnected"
                },
                "data_processing": {
                    "buffered_devices": len(self.data_processor.data_buffer),
                    "total_data_points": sum(len(data) for data in self.data_processor.data_buffer.values())
                },
                "devices": [
                    {
                        "device_id": device.device_id,
                        "name": device.name,
                        "type": device.device_type.value,
                        "protocol": device.protocol.value,
                        "status": device.status,
                        "capabilities": device.capabilities
                    }
                    for device in online_devices
                ]
            }
            
        except Exception as e:
            self.logger.error(f"System status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup IoT orchestrator resources."""
        try:
            await self.mqtt_connector.disconnect()
            await self.http_connector.close()
            self.logger.info("IoT orchestrator cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Example usage
async def main():
    """Example usage of IoT connector."""
    orchestrator = IoTOrchestrator()
    
    # Initialize orchestrator
    success = await orchestrator.initialize()
    if not success:
        print("Failed to initialize IoT orchestrator")
        return
    
    # Get system status
    status = await orchestrator.get_system_status()
    print(f"IoT system status: {status}")
    
    # Start listening for messages
    await orchestrator.mqtt_connector.start_listening()

if __name__ == "__main__":
    asyncio.run(main())


