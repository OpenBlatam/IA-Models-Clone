"""
Gamma App - IoT and Edge Computing Integration Service
Advanced IoT device management, edge computing, and real-time data processing
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import numpy as np
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict, deque
import paho.mqtt.client as mqtt
import websockets
import aiohttp
import socket
import threading
import queue
import struct
import pickle
import zlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """IoT device types"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    SMART_DEVICE = "smart_device"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"
    INDUSTRIAL = "industrial"

class DataType(Enum):
    """Data types"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    SOUND = "sound"
    MOTION = "motion"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"

class ProcessingMode(Enum):
    """Processing modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    EDGE = "edge"
    CLOUD = "cloud"
    HYBRID = "hybrid"

class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class IoTDevice:
    """IoT device definition"""
    device_id: str
    name: str
    device_type: DeviceType
    location: Dict[str, float]  # lat, lon, alt
    capabilities: List[str]
    status: str = "offline"
    last_seen: datetime = None
    metadata: Dict[str, Any] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    created_at: datetime = None

@dataclass
class SensorData:
    """Sensor data structure"""
    device_id: str
    data_type: DataType
    value: Any
    timestamp: datetime
    quality: float = 1.0
    metadata: Dict[str, Any] = None

@dataclass
class ProcessingTask:
    """Edge processing task"""
    task_id: str
    device_id: str
    data: Any
    processing_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    deadline: Optional[datetime] = None
    status: str = "pending"
    result: Any = None
    created_at: datetime = None

@dataclass
class EdgeNode:
    """Edge computing node"""
    node_id: str
    name: str
    location: Dict[str, float]
    capabilities: List[str]
    processing_power: float
    memory: int
    storage: int
    network_bandwidth: float
    status: str = "offline"
    connected_devices: List[str] = None
    created_at: datetime = None

class AdvancedIoTEdgeService:
    """Advanced IoT and Edge Computing Service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "iot_edge.db")
        self.redis_client = None
        self.mqtt_client = None
        self.websocket_server = None
        self.devices = {}
        self.edge_nodes = {}
        self.processing_tasks = {}
        self.data_streams = {}
        self.anomaly_detectors = {}
        self.edge_models = {}
        self.encryption_key = None
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_encryption()
        self._init_mqtt()
        self._init_websocket()
        self._init_edge_models()
        self._init_anomaly_detectors()
    
    def _init_database(self):
        """Initialize IoT Edge database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create devices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iot_devices (
                    device_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    location TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    status TEXT DEFAULT 'offline',
                    last_seen DATETIME,
                    metadata TEXT,
                    security_level TEXT DEFAULT 'medium',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create sensor data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    data_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    quality REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (device_id) REFERENCES iot_devices (device_id)
                )
            """)
            
            # Create edge nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edge_nodes (
                    node_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    processing_power REAL NOT NULL,
                    memory INTEGER NOT NULL,
                    storage INTEGER NOT NULL,
                    network_bandwidth REAL NOT NULL,
                    status TEXT DEFAULT 'offline',
                    connected_devices TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create processing tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_tasks (
                    task_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    processing_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    deadline DATETIME,
                    status TEXT DEFAULT 'pending',
                    result TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (device_id) REFERENCES iot_devices (device_id)
                )
            """)
            
            conn.commit()
        
        logger.info("IoT Edge database initialized")
    
    def _init_redis(self):
        """Initialize Redis for real-time data"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized for IoT Edge")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _init_encryption(self):
        """Initialize encryption for secure communication"""
        try:
            # Generate or load encryption key
            key_file = Path("data/iot_encryption.key")
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                password = self.config.get("encryption_password", "default_password").encode()
                salt = b'salt_123456789012345678901234567890'  # In production, use random salt
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))
                self.encryption_key = key
                
                # Save key
                key_file.parent.mkdir(exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
            
            self.cipher = Fernet(self.encryption_key)
            logger.info("Encryption initialized for IoT Edge")
        except Exception as e:
            logger.error(f"Encryption initialization failed: {e}")
    
    def _init_mqtt(self):
        """Initialize MQTT client for device communication"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Connect to MQTT broker
            broker_host = self.config.get("mqtt_broker_host", "localhost")
            broker_port = self.config.get("mqtt_broker_port", 1883)
            self.mqtt_client.connect(broker_host, broker_port, 60)
            self.mqtt_client.loop_start()
            
            logger.info("MQTT client initialized for IoT Edge")
        except Exception as e:
            logger.warning(f"MQTT initialization failed: {e}")
    
    def _init_websocket(self):
        """Initialize WebSocket server for real-time communication"""
        try:
            self.websocket_server = websockets.serve(
                self._handle_websocket_connection,
                "localhost",
                8765
            )
            logger.info("WebSocket server initialized for IoT Edge")
        except Exception as e:
            logger.warning(f"WebSocket initialization failed: {e}")
    
    def _init_edge_models(self):
        """Initialize edge AI models"""
        try:
            # Initialize lightweight models for edge processing
            self.edge_models = {
                "anomaly_detection": IsolationForest(contamination=0.1, random_state=42),
                "clustering": DBSCAN(eps=0.5, min_samples=5),
                "classification": self._create_lightweight_classifier(),
                "regression": self._create_lightweight_regressor()
            }
            
            logger.info("Edge AI models initialized")
        except Exception as e:
            logger.error(f"Edge models initialization failed: {e}")
    
    def _init_anomaly_detectors(self):
        """Initialize anomaly detectors for different data types"""
        try:
            self.anomaly_detectors = {
                DataType.TEMPERATURE: IsolationForest(contamination=0.05, random_state=42),
                DataType.HUMIDITY: IsolationForest(contamination=0.05, random_state=42),
                DataType.PRESSURE: IsolationForest(contamination=0.05, random_state=42),
                DataType.MOTION: IsolationForest(contamination=0.1, random_state=42)
            }
            
            logger.info("Anomaly detectors initialized")
        except Exception as e:
            logger.error(f"Anomaly detectors initialization failed: {e}")
    
    def _create_lightweight_classifier(self):
        """Create lightweight classifier for edge processing"""
        class LightweightClassifier(nn.Module):
            def __init__(self, input_size=10, hidden_size=32, num_classes=3):
                super(LightweightClassifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return LightweightClassifier()
    
    def _create_lightweight_regressor(self):
        """Create lightweight regressor for edge processing"""
        class LightweightRegressor(nn.Module):
            def __init__(self, input_size=10, hidden_size=32):
                super(LightweightRegressor, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return LightweightRegressor()
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to device topics
            client.subscribe("devices/+/data")
            client.subscribe("devices/+/status")
            client.subscribe("devices/+/commands")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Parse topic to get device ID
            topic_parts = topic.split('/')
            if len(topic_parts) >= 3:
                device_id = topic_parts[1]
                message_type = topic_parts[2]
                
                if message_type == "data":
                    asyncio.create_task(self._handle_device_data(device_id, payload))
                elif message_type == "status":
                    asyncio.create_task(self._handle_device_status(device_id, payload))
                elif message_type == "commands":
                    asyncio.create_task(self._handle_device_command(device_id, payload))
        
        except Exception as e:
            logger.error(f"MQTT message handling failed: {e}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning(f"Disconnected from MQTT broker: {rc}")
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connections"""
        try:
            async for message in websocket:
                data = json.loads(message)
                await self._process_websocket_message(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket handling failed: {e}")
    
    async def _process_websocket_message(self, websocket, data):
        """Process WebSocket message"""
        try:
            message_type = data.get("type")
            
            if message_type == "device_data":
                await self._handle_realtime_data(websocket, data)
            elif message_type == "device_command":
                await self._handle_device_command_ws(websocket, data)
            elif message_type == "subscribe":
                await self._handle_subscription(websocket, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscription(websocket, data)
        
        except Exception as e:
            logger.error(f"WebSocket message processing failed: {e}")
    
    async def register_device(
        self,
        device_id: str,
        name: str,
        device_type: DeviceType,
        location: Dict[str, float],
        capabilities: List[str],
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a new IoT device"""
        
        try:
            device = IoTDevice(
                device_id=device_id,
                name=name,
                device_type=device_type,
                location=location,
                capabilities=capabilities,
                security_level=security_level,
                metadata=metadata or {},
                created_at=datetime.now()
            )
            
            # Store device
            self.devices[device_id] = device
            await self._store_device(device)
            
            # Subscribe to device topics
            if self.mqtt_client:
                self.mqtt_client.subscribe(f"devices/{device_id}/data")
                self.mqtt_client.subscribe(f"devices/{device_id}/status")
            
            logger.info(f"Device registered: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            return False
    
    async def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device"""
        
        try:
            if device_id in self.devices:
                del self.devices[device_id]
                
                # Unsubscribe from device topics
                if self.mqtt_client:
                    self.mqtt_client.unsubscribe(f"devices/{device_id}/data")
                    self.mqtt_client.unsubscribe(f"devices/{device_id}/status")
                
                # Remove from database
                await self._remove_device(device_id)
                
                logger.info(f"Device unregistered: {device_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Device unregistration failed: {e}")
            return False
    
    async def send_device_command(
        self,
        device_id: str,
        command: str,
        parameters: Dict[str, Any],
        priority: int = 1
    ) -> bool:
        """Send command to IoT device"""
        
        try:
            if device_id not in self.devices:
                logger.error(f"Device not found: {device_id}")
                return False
            
            command_data = {
                "command": command,
                "parameters": parameters,
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "command_id": str(uuid.uuid4())
            }
            
            # Encrypt command if high security
            device = self.devices[device_id]
            if device.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                command_data = self._encrypt_data(command_data)
            
            # Send via MQTT
            if self.mqtt_client:
                topic = f"devices/{device_id}/commands"
                self.mqtt_client.publish(topic, json.dumps(command_data))
            
            logger.info(f"Command sent to device {device_id}: {command}")
            return True
            
        except Exception as e:
            logger.error(f"Device command failed: {e}")
            return False
    
    async def process_sensor_data(
        self,
        device_id: str,
        data_type: DataType,
        value: Any,
        quality: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process sensor data with edge computing"""
        
        try:
            # Create sensor data object
            sensor_data = SensorData(
                device_id=device_id,
                data_type=data_type,
                value=value,
                timestamp=datetime.now(),
                quality=quality,
                metadata=metadata or {}
            )
            
            # Store data
            await self._store_sensor_data(sensor_data)
            
            # Process data based on type
            processing_result = await self._process_data_by_type(sensor_data)
            
            # Check for anomalies
            anomaly_result = await self._detect_anomalies(sensor_data)
            
            # Update device status
            await self._update_device_status(device_id, "online")
            
            # Cache in Redis for real-time access
            if self.redis_client:
                cache_key = f"sensor_data:{device_id}:{data_type.value}"
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour
                    json.dumps(asdict(sensor_data), default=str)
                )
            
            result = {
                "status": "success",
                "processing_result": processing_result,
                "anomaly_result": anomaly_result,
                "timestamp": sensor_data.timestamp.isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Sensor data processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _process_data_by_type(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Process data based on type"""
        
        data_type = sensor_data.data_type
        value = sensor_data.value
        
        if data_type == DataType.TEMPERATURE:
            return await self._process_temperature_data(value, sensor_data.metadata)
        elif data_type == DataType.HUMIDITY:
            return await self._process_humidity_data(value, sensor_data.metadata)
        elif data_type == DataType.PRESSURE:
            return await self._process_pressure_data(value, sensor_data.metadata)
        elif data_type == DataType.MOTION:
            return await self._process_motion_data(value, sensor_data.metadata)
        elif data_type == DataType.IMAGE:
            return await self._process_image_data(value, sensor_data.metadata)
        elif data_type == DataType.AUDIO:
            return await self._process_audio_data(value, sensor_data.metadata)
        else:
            return {"processed": True, "type": "generic"}
    
    async def _process_temperature_data(self, value: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process temperature data"""
        
        # Convert to different units
        celsius = value
        fahrenheit = (celsius * 9/5) + 32
        kelvin = celsius + 273.15
        
        # Calculate comfort level
        comfort_level = "comfortable"
        if celsius < 18:
            comfort_level = "cold"
        elif celsius > 26:
            comfort_level = "hot"
        
        return {
            "celsius": celsius,
            "fahrenheit": fahrenheit,
            "kelvin": kelvin,
            "comfort_level": comfort_level,
            "processed": True
        }
    
    async def _process_humidity_data(self, value: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process humidity data"""
        
        # Determine humidity level
        humidity_level = "normal"
        if value < 30:
            humidity_level = "low"
        elif value > 70:
            humidity_level = "high"
        
        return {
            "humidity_percent": value,
            "humidity_level": humidity_level,
            "processed": True
        }
    
    async def _process_pressure_data(self, value: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process pressure data"""
        
        # Convert units
        pascal = value
        bar = pascal / 100000
        psi = pascal / 6895
        
        # Determine pressure trend
        pressure_trend = "stable"
        # This would require historical data for trend analysis
        
        return {
            "pascal": pascal,
            "bar": bar,
            "psi": psi,
            "pressure_trend": pressure_trend,
            "processed": True
        }
    
    async def _process_motion_data(self, value: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process motion data"""
        
        # Extract motion components
        x = value.get("x", 0)
        y = value.get("y", 0)
        z = value.get("z", 0)
        
        # Calculate motion magnitude
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Determine motion type
        motion_type = "stationary"
        if magnitude > 0.1:
            motion_type = "moving"
        if magnitude > 1.0:
            motion_type = "fast_moving"
        
        return {
            "x": x,
            "y": y,
            "z": z,
            "magnitude": magnitude,
            "motion_type": motion_type,
            "processed": True
        }
    
    async def _process_image_data(self, value: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process image data"""
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(value)
            
            # Convert to PIL Image
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))
            
            # Basic image analysis
            width, height = image.size
            mode = image.mode
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Calculate basic statistics
            mean_color = np.mean(img_array, axis=(0, 1))
            brightness = np.mean(img_array)
            
            return {
                "width": width,
                "height": height,
                "mode": mode,
                "mean_color": mean_color.tolist(),
                "brightness": float(brightness),
                "processed": True
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {"processed": False, "error": str(e)}
    
    async def _process_audio_data(self, value: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio data"""
        
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(value)
            
            # Basic audio analysis
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate audio statistics
            rms = np.sqrt(np.mean(audio_array**2))
            max_amplitude = np.max(np.abs(audio_array))
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            
            return {
                "rms": float(rms),
                "max_amplitude": int(max_amplitude),
                "zero_crossings": int(zero_crossings),
                "duration_estimate": len(audio_array) / 44100,  # Assuming 44.1kHz
                "processed": True
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {"processed": False, "error": str(e)}
    
    async def _detect_anomalies(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Detect anomalies in sensor data"""
        
        try:
            data_type = sensor_data.data_type
            value = sensor_data.value
            
            if data_type not in self.anomaly_detectors:
                return {"anomaly_detected": False, "reason": "No detector available"}
            
            # Get historical data for training
            historical_data = await self._get_historical_data(
                sensor_data.device_id, data_type, limit=100
            )
            
            if len(historical_data) < 10:
                return {"anomaly_detected": False, "reason": "Insufficient historical data"}
            
            # Prepare data for anomaly detection
            data_array = np.array(historical_data + [value]).reshape(-1, 1)
            
            # Fit detector if not already fitted
            detector = self.anomaly_detectors[data_type]
            if not hasattr(detector, 'decision_function'):
                detector.fit(data_array[:-1])
            
            # Detect anomaly
            anomaly_score = detector.decision_function([[value]])[0]
            is_anomaly = detector.predict([[value]])[0] == -1
            
            return {
                "anomaly_detected": bool(is_anomaly),
                "anomaly_score": float(anomaly_score),
                "confidence": abs(anomaly_score),
                "threshold": -0.1  # Adjustable threshold
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"anomaly_detected": False, "error": str(e)}
    
    async def _get_historical_data(
        self,
        device_id: str,
        data_type: DataType,
        limit: int = 100
    ) -> List[float]:
        """Get historical data for analysis"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value FROM sensor_data
                    WHERE device_id = ? AND data_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (device_id, data_type.value, limit))
                
                rows = cursor.fetchall()
                return [float(row[0]) for row in rows]
                
        except Exception as e:
            logger.error(f"Historical data retrieval failed: {e}")
            return []
    
    async def create_edge_processing_task(
        self,
        device_id: str,
        data: Any,
        processing_type: str,
        parameters: Dict[str, Any],
        priority: int = 1,
        deadline: Optional[datetime] = None
    ) -> str:
        """Create edge processing task"""
        
        task_id = str(uuid.uuid4())
        
        task = ProcessingTask(
            task_id=task_id,
            device_id=device_id,
            data=data,
            processing_type=processing_type,
            parameters=parameters,
            priority=priority,
            deadline=deadline,
            created_at=datetime.now()
        )
        
        # Store task
        self.processing_tasks[task_id] = task
        await self._store_processing_task(task)
        
        # Process task
        asyncio.create_task(self._process_edge_task(task))
        
        logger.info(f"Edge processing task created: {task_id}")
        return task_id
    
    async def _process_edge_task(self, task: ProcessingTask):
        """Process edge computing task"""
        
        try:
            task.status = "processing"
            await self._update_processing_task(task)
            
            # Process based on type
            if task.processing_type == "anomaly_detection":
                result = await self._edge_anomaly_detection(task)
            elif task.processing_type == "classification":
                result = await self._edge_classification(task)
            elif task.processing_type == "regression":
                result = await self._edge_regression(task)
            elif task.processing_type == "clustering":
                result = await self._edge_clustering(task)
            elif task.processing_type == "image_processing":
                result = await self._edge_image_processing(task)
            elif task.processing_type == "audio_processing":
                result = await self._edge_audio_processing(task)
            else:
                result = {"error": f"Unsupported processing type: {task.processing_type}"}
            
            task.result = result
            task.status = "completed"
            await self._update_processing_task(task)
            
            logger.info(f"Edge processing task completed: {task.task_id}")
            
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            await self._update_processing_task(task)
            
            logger.error(f"Edge processing task failed: {task.task_id} - {e}")
    
    async def _edge_anomaly_detection(self, task: ProcessingTask) -> Dict[str, Any]:
        """Edge anomaly detection"""
        
        data = task.data
        parameters = task.parameters
        
        # Use isolation forest for anomaly detection
        detector = IsolationForest(
            contamination=parameters.get("contamination", 0.1),
            random_state=42
        )
        
        # Fit detector
        detector.fit([data])
        
        # Detect anomaly
        anomaly_score = detector.decision_function([data])[0]
        is_anomaly = detector.predict([data])[0] == -1
        
        return {
            "anomaly_detected": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "confidence": abs(anomaly_score)
        }
    
    async def _edge_classification(self, task: ProcessingTask) -> Dict[str, Any]:
        """Edge classification"""
        
        data = task.data
        parameters = task.parameters
        
        # Use lightweight classifier
        model = self.edge_models["classification"]
        
        # Convert data to tensor
        data_tensor = torch.FloatTensor([data])
        
        # Make prediction
        with torch.no_grad():
            output = model(data_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        
        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "class_probabilities": torch.softmax(output, dim=1).tolist()[0]
        }
    
    async def _edge_regression(self, task: ProcessingTask) -> Dict[str, Any]:
        """Edge regression"""
        
        data = task.data
        parameters = task.parameters
        
        # Use lightweight regressor
        model = self.edge_models["regression"]
        
        # Convert data to tensor
        data_tensor = torch.FloatTensor([data])
        
        # Make prediction
        with torch.no_grad():
            prediction = model(data_tensor).item()
        
        return {
            "prediction": float(prediction),
            "confidence": 0.8  # Simplified confidence
        }
    
    async def _edge_clustering(self, task: ProcessingTask) -> Dict[str, Any]:
        """Edge clustering"""
        
        data = task.data
        parameters = task.parameters
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(
            eps=parameters.get("eps", 0.5),
            min_samples=parameters.get("min_samples", 5)
        )
        
        # Fit clustering
        cluster_labels = clustering.fit_predict([data])
        
        return {
            "cluster_label": int(cluster_labels[0]),
            "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "is_outlier": cluster_labels[0] == -1
        }
    
    async def _edge_image_processing(self, task: ProcessingTask) -> Dict[str, Any]:
        """Edge image processing"""
        
        image_data = task.data
        parameters = task.parameters
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_bytes))
            
            # Apply processing
            if parameters.get("resize"):
                new_size = parameters["resize"]
                image = image.resize(new_size)
            
            if parameters.get("grayscale"):
                image = image.convert("L")
            
            if parameters.get("blur"):
                blur_radius = parameters["blur"]
                image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Convert back to base64
            output_buffer = io.BytesIO()
            image.save(output_buffer, format="JPEG")
            processed_image = base64.b64encode(output_buffer.getvalue()).decode()
            
            return {
                "processed_image": processed_image,
                "width": image.width,
                "height": image.height,
                "mode": image.mode
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _edge_audio_processing(self, task: ProcessingTask) -> Dict[str, Any]:
        """Edge audio processing"""
        
        audio_data = task.data
        parameters = task.parameters
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Apply processing
            if parameters.get("normalize"):
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            if parameters.get("filter"):
                filter_type = parameters["filter"]
                if filter_type == "lowpass":
                    # Simple low-pass filter
                    audio_array = signal.lfilter([1, 1], [1, -0.9], audio_array)
            
            # Convert back to base64
            processed_audio = base64.b64encode(audio_array.tobytes()).decode()
            
            return {
                "processed_audio": processed_audio,
                "length": len(audio_array),
                "sample_rate": parameters.get("sample_rate", 44100)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_device_data(self, device_id: str, data: Dict[str, Any]):
        """Handle device data from MQTT"""
        
        try:
            # Decrypt data if needed
            device = self.devices.get(device_id)
            if device and device.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                data = self._decrypt_data(data)
            
            # Process sensor data
            data_type = DataType(data["data_type"])
            value = data["value"]
            quality = data.get("quality", 1.0)
            metadata = data.get("metadata", {})
            
            result = await self.process_sensor_data(
                device_id, data_type, value, quality, metadata
            )
            
            logger.info(f"Device data processed: {device_id}")
            
        except Exception as e:
            logger.error(f"Device data handling failed: {e}")
    
    async def _handle_device_status(self, device_id: str, status: Dict[str, Any]):
        """Handle device status from MQTT"""
        
        try:
            device = self.devices.get(device_id)
            if device:
                device.status = status.get("status", "unknown")
                device.last_seen = datetime.now()
                
                # Update in database
                await self._update_device(device)
                
                logger.info(f"Device status updated: {device_id} - {device.status}")
            
        except Exception as e:
            logger.error(f"Device status handling failed: {e}")
    
    async def _handle_device_command(self, device_id: str, command: Dict[str, Any]):
        """Handle device command response"""
        
        try:
            # Process command response
            command_id = command.get("command_id")
            response = command.get("response")
            status = command.get("status")
            
            logger.info(f"Device command response: {device_id} - {command_id} - {status}")
            
        except Exception as e:
            logger.error(f"Device command handling failed: {e}")
    
    async def _handle_realtime_data(self, websocket, data: Dict[str, Any]):
        """Handle real-time data via WebSocket"""
        
        try:
            device_id = data.get("device_id")
            data_type = data.get("data_type")
            
            # Get latest data from Redis
            if self.redis_client:
                cache_key = f"sensor_data:{device_id}:{data_type}"
                cached_data = self.redis_client.get(cache_key)
                
                if cached_data:
                    response = {
                        "type": "realtime_data",
                        "data": json.loads(cached_data)
                    }
                    await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Real-time data handling failed: {e}")
    
    async def _handle_device_command_ws(self, websocket, data: Dict[str, Any]):
        """Handle device command via WebSocket"""
        
        try:
            device_id = data.get("device_id")
            command = data.get("command")
            parameters = data.get("parameters", {})
            
            success = await self.send_device_command(device_id, command, parameters)
            
            response = {
                "type": "command_response",
                "success": success,
                "device_id": device_id,
                "command": command
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Device command WebSocket handling failed: {e}")
    
    async def _handle_subscription(self, websocket, data: Dict[str, Any]):
        """Handle WebSocket subscription"""
        
        try:
            device_id = data.get("device_id")
            data_type = data.get("data_type")
            
            # Add to subscription list
            subscription_key = f"ws_subscription:{device_id}:{data_type}"
            if self.redis_client:
                self.redis_client.sadd("websocket_subscriptions", subscription_key)
            
            response = {
                "type": "subscription_confirmed",
                "device_id": device_id,
                "data_type": data_type
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Subscription handling failed: {e}")
    
    async def _handle_unsubscription(self, websocket, data: Dict[str, Any]):
        """Handle WebSocket unsubscription"""
        
        try:
            device_id = data.get("device_id")
            data_type = data.get("data_type")
            
            # Remove from subscription list
            subscription_key = f"ws_subscription:{device_id}:{data_type}"
            if self.redis_client:
                self.redis_client.srem("websocket_subscriptions", subscription_key)
            
            response = {
                "type": "unsubscription_confirmed",
                "device_id": device_id,
                "data_type": data_type
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Unsubscription handling failed: {e}")
    
    def _encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt data for secure transmission"""
        
        try:
            json_data = json.dumps(data)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            return {"encrypted": True, "data": base64.b64encode(encrypted_data).decode()}
        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            return data
    
    def _decrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt data from secure transmission"""
        
        try:
            if data.get("encrypted"):
                encrypted_data = base64.b64decode(data["data"])
                decrypted_data = self.cipher.decrypt(encrypted_data)
                return json.loads(decrypted_data.decode())
            return data
        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            return data
    
    async def _store_device(self, device: IoTDevice):
        """Store device in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO iot_devices
                (device_id, name, device_type, location, capabilities, status, last_seen, metadata, security_level, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                device.device_id,
                device.name,
                device.device_type.value,
                json.dumps(device.location),
                json.dumps(device.capabilities),
                device.status,
                device.last_seen.isoformat() if device.last_seen else None,
                json.dumps(device.metadata),
                device.security_level.value,
                device.created_at.isoformat()
            ))
            conn.commit()
    
    async def _update_device(self, device: IoTDevice):
        """Update device in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE iot_devices
                SET status = ?, last_seen = ?, metadata = ?
                WHERE device_id = ?
            """, (
                device.status,
                device.last_seen.isoformat() if device.last_seen else None,
                json.dumps(device.metadata),
                device.device_id
            ))
            conn.commit()
    
    async def _remove_device(self, device_id: str):
        """Remove device from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM iot_devices WHERE device_id = ?", (device_id,))
            conn.commit()
    
    async def _store_sensor_data(self, sensor_data: SensorData):
        """Store sensor data in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sensor_data
                (data_id, device_id, data_type, value, timestamp, quality, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                sensor_data.device_id,
                sensor_data.data_type.value,
                json.dumps(sensor_data.value),
                sensor_data.timestamp.isoformat(),
                sensor_data.quality,
                json.dumps(sensor_data.metadata),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    async def _store_processing_task(self, task: ProcessingTask):
        """Store processing task in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_tasks
                (task_id, device_id, data, processing_type, parameters, priority, deadline, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.device_id,
                json.dumps(task.data),
                task.processing_type,
                json.dumps(task.parameters),
                task.priority,
                task.deadline.isoformat() if task.deadline else None,
                task.status,
                task.created_at.isoformat()
            ))
            conn.commit()
    
    async def _update_processing_task(self, task: ProcessingTask):
        """Update processing task in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE processing_tasks
                SET status = ?, result = ?
                WHERE task_id = ?
            """, (
                task.status,
                json.dumps(task.result) if task.result else None,
                task.task_id
            ))
            conn.commit()
    
    async def _update_device_status(self, device_id: str, status: str):
        """Update device status"""
        
        device = self.devices.get(device_id)
        if device:
            device.status = status
            device.last_seen = datetime.now()
            await self._update_device(device)
    
    async def get_device_analytics(self, device_id: str) -> Dict[str, Any]:
        """Get device analytics"""
        
        try:
            device = self.devices.get(device_id)
            if not device:
                return {"error": "Device not found"}
            
            # Get recent data
            recent_data = await self._get_recent_data(device_id, limit=100)
            
            # Calculate analytics
            analytics = {
                "device_info": asdict(device),
                "data_summary": await self._calculate_data_summary(recent_data),
                "anomaly_summary": await self._calculate_anomaly_summary(recent_data),
                "performance_metrics": await self._calculate_performance_metrics(device_id),
                "generated_at": datetime.now().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Device analytics failed: {e}")
            return {"error": str(e)}
    
    async def _get_recent_data(self, device_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent sensor data"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data_type, value, timestamp, quality, metadata
                    FROM sensor_data
                    WHERE device_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (device_id, limit))
                
                rows = cursor.fetchall()
                return [
                    {
                        "data_type": row[0],
                        "value": json.loads(row[1]),
                        "timestamp": row[2],
                        "quality": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {}
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Recent data retrieval failed: {e}")
            return []
    
    async def _calculate_data_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate data summary"""
        
        if not data:
            return {"total_records": 0}
        
        # Group by data type
        data_by_type = defaultdict(list)
        for record in data:
            data_by_type[record["data_type"]].append(record["value"])
        
        summary = {"total_records": len(data)}
        
        for data_type, values in data_by_type.items():
            if values and isinstance(values[0], (int, float)):
                summary[data_type] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            else:
                summary[data_type] = {
                    "count": len(values),
                    "type": "non_numeric"
                }
        
        return summary
    
    async def _calculate_anomaly_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate anomaly summary"""
        
        # This would analyze the data for anomalies
        # For now, return a placeholder
        return {
            "total_anomalies": 0,
            "anomaly_rate": 0.0,
            "last_anomaly": None
        }
    
    async def _calculate_performance_metrics(self, device_id: str) -> Dict[str, Any]:
        """Calculate device performance metrics"""
        
        device = self.devices.get(device_id)
        if not device:
            return {}
        
        # Calculate uptime
        uptime_hours = 0
        if device.last_seen:
            uptime_hours = (datetime.now() - device.created_at).total_seconds() / 3600
        
        return {
            "uptime_hours": uptime_hours,
            "status": device.status,
            "last_seen": device.last_seen.isoformat() if device.last_seen else None,
            "security_level": device.security_level.value
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("IoT Edge service cleanup completed")

# Global instance
iot_edge_service = None

async def get_iot_edge_service() -> AdvancedIoTEdgeService:
    """Get global IoT Edge service instance"""
    global iot_edge_service
    if not iot_edge_service:
        config = {
            "database_path": "data/iot_edge.db",
            "redis_url": "redis://localhost:6379",
            "mqtt_broker_host": "localhost",
            "mqtt_broker_port": 1883,
            "encryption_password": "your_encryption_password"
        }
        iot_edge_service = AdvancedIoTEdgeService(config)
    return iot_edge_service



