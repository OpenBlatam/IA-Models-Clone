"""
Gamma App - Real Improvement IoT
IoT system for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import socket
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import paho.mqtt.client as mqtt
import requests
import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

class IoTDeviceType(Enum):
    """IoT device types"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    MICROCONTROLLER = "microcontroller"
    EMBEDDED = "embedded"
    WEARABLE = "wearable"
    SMART_HOME = "smart_home"

class IoTProtocol(Enum):
    """IoT protocols"""
    MQTT = "mqtt"
    HTTP = "http"
    COAP = "coap"
    WEBSOCKET = "websocket"
    TCP = "tcp"
    UDP = "udp"
    BLE = "ble"
    ZIGBEE = "zigbee"

@dataclass
class IoTDevice:
    """IoT device"""
    device_id: str
    name: str
    type: IoTDeviceType
    protocol: IoTProtocol
    ip_address: str
    port: int
    status: str
    configuration: Dict[str, Any]
    last_seen: datetime = None
    created_at: datetime = None
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()

@dataclass
class IoTSensorData:
    """IoT sensor data"""
    data_id: str
    device_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime = None
    location: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

class RealImprovementIoT:
    """
    IoT system for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize IoT system"""
        self.project_root = Path(project_root)
        self.devices: Dict[str, IoTDevice] = {}
        self.sensor_data: Dict[str, List[IoTSensorData]] = {}
        self.iot_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.mqtt_client = None
        self.data_buffer: List[IoTSensorData] = []
        self.data_lock = threading.Lock()
        
        # Initialize MQTT client
        self._initialize_mqtt()
        
        # Initialize with default devices
        self._initialize_default_devices()
        
        # Start data collection
        self._start_data_collection()
        
        logger.info(f"Real Improvement IoT initialized for {self.project_root}")
    
    def _initialize_mqtt(self):
        """Initialize MQTT client"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Connect to MQTT broker
            self.mqtt_client.connect("localhost", 1883, 60)
            self.mqtt_client.loop_start()
            
        except Exception as e:
            logger.warning(f"Failed to initialize MQTT client: {e}")
    
    def _initialize_default_devices(self):
        """Initialize default IoT devices"""
        # Temperature sensor
        temp_sensor = IoTDevice(
            device_id="temp_sensor_001",
            name="Temperature Sensor 001",
            type=IoTDeviceType.SENSOR,
            protocol=IoTProtocol.MQTT,
            ip_address="192.168.1.100",
            port=1883,
            status="online",
            configuration={
                "sensor_type": "temperature",
                "unit": "celsius",
                "sampling_rate": 1.0,
                "threshold_min": -10.0,
                "threshold_max": 50.0
            },
            battery_level=85.0,
            signal_strength=-45.0
        )
        self.devices[temp_sensor.device_id] = temp_sensor
        
        # Humidity sensor
        humidity_sensor = IoTDevice(
            device_id="humidity_sensor_001",
            name="Humidity Sensor 001",
            type=IoTDeviceType.SENSOR,
            protocol=IoTProtocol.MQTT,
            ip_address="192.168.1.101",
            port=1883,
            status="online",
            configuration={
                "sensor_type": "humidity",
                "unit": "percent",
                "sampling_rate": 2.0,
                "threshold_min": 0.0,
                "threshold_max": 100.0
            },
            battery_level=92.0,
            signal_strength=-38.0
        )
        self.devices[humidity_sensor.device_id] = humidity_sensor
        
        # Smart light actuator
        smart_light = IoTDevice(
            device_id="smart_light_001",
            name="Smart Light 001",
            type=IoTDeviceType.ACTUATOR,
            protocol=IoTProtocol.HTTP,
            ip_address="192.168.1.102",
            port=8080,
            status="online",
            configuration={
                "actuator_type": "light",
                "brightness_range": [0, 100],
                "color_support": True,
                "power_consumption": 15.0
            },
            battery_level=None,
            signal_strength=-42.0
        )
        self.devices[smart_light.device_id] = smart_light
        
        # Security camera
        security_camera = IoTDevice(
            device_id="camera_001",
            name="Security Camera 001",
            type=IoTDeviceType.CAMERA,
            protocol=IoTProtocol.HTTP,
            ip_address="192.168.1.103",
            port=8081,
            status="online",
            configuration={
                "resolution": "1920x1080",
                "fps": 30,
                "night_vision": True,
                "motion_detection": True,
                "storage_capacity": "64GB"
            },
            battery_level=None,
            signal_strength=-35.0
        )
        self.devices[security_camera.device_id] = security_camera
    
    def _start_data_collection(self):
        """Start data collection from IoT devices"""
        try:
            # Start data collection thread
            data_thread = threading.Thread(target=self._collect_sensor_data, daemon=True)
            data_thread.start()
            
            # Start device monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_devices, daemon=True)
            monitor_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start data collection: {e}")
    
    def _collect_sensor_data(self):
        """Collect sensor data from IoT devices"""
        while True:
            try:
                for device_id, device in self.devices.items():
                    if device.type == IoTDeviceType.SENSOR and device.status == "online":
                        # Simulate sensor data collection
                        sensor_data = self._simulate_sensor_data(device)
                        
                        with self.data_lock:
                            self.data_buffer.append(sensor_data)
                            
                            # Store in device-specific buffer
                            if device_id not in self.sensor_data:
                                self.sensor_data[device_id] = []
                            self.sensor_data[device_id].append(sensor_data)
                            
                            # Keep only last 1000 readings per device
                            if len(self.sensor_data[device_id]) > 1000:
                                self.sensor_data[device_id] = self.sensor_data[device_id][-1000:]
                
                time.sleep(1)  # Collect data every second
                
            except Exception as e:
                logger.error(f"Failed to collect sensor data: {e}")
                time.sleep(5)
    
    def _simulate_sensor_data(self, device: IoTDevice) -> IoTSensorData:
        """Simulate sensor data for a device"""
        try:
            sensor_type = device.configuration.get("sensor_type", "unknown")
            
            # Generate realistic sensor data
            if sensor_type == "temperature":
                # Temperature varies between 15-25°C with some noise
                base_temp = 20.0
                variation = np.random.normal(0, 2.0)
                value = base_temp + variation
                unit = "celsius"
            elif sensor_type == "humidity":
                # Humidity varies between 40-70% with some noise
                base_humidity = 55.0
                variation = np.random.normal(0, 5.0)
                value = max(0, min(100, base_humidity + variation))
                unit = "percent"
            elif sensor_type == "pressure":
                # Pressure varies around 1013 hPa
                base_pressure = 1013.0
                variation = np.random.normal(0, 10.0)
                value = base_pressure + variation
                unit = "hPa"
            elif sensor_type == "light":
                # Light varies between 0-1000 lux
                base_light = 500.0
                variation = np.random.normal(0, 100.0)
                value = max(0, min(1000, base_light + variation))
                unit = "lux"
            else:
                # Generic sensor
                value = np.random.uniform(0, 100)
                unit = "units"
            
            return IoTSensorData(
                data_id=f"data_{int(time.time() * 1000)}",
                device_id=device.device_id,
                sensor_type=sensor_type,
                value=value,
                unit=unit,
                location={"lat": 40.7128, "lon": -74.0060},  # NYC coordinates
                metadata={
                    "battery_level": device.battery_level,
                    "signal_strength": device.signal_strength,
                    "device_name": device.name
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to simulate sensor data: {e}")
            return IoTSensorData(
                data_id=f"data_{int(time.time() * 1000)}",
                device_id=device.device_id,
                sensor_type="unknown",
                value=0.0,
                unit="units"
            )
    
    def _monitor_devices(self):
        """Monitor IoT devices"""
        while True:
            try:
                for device_id, device in self.devices.items():
                    # Check device connectivity
                    is_online = self._check_device_connectivity(device)
                    
                    if is_online and device.status != "online":
                        device.status = "online"
                        device.last_seen = datetime.utcnow()
                        self._log_iot("device_online", f"Device {device.name} came online")
                    elif not is_online and device.status == "online":
                        device.status = "offline"
                        self._log_iot("device_offline", f"Device {device.name} went offline")
                    
                    # Update battery level (simulate battery drain)
                    if device.battery_level is not None:
                        device.battery_level = max(0, device.battery_level - 0.01)
                        if device.battery_level < 10:
                            self._log_iot("low_battery", f"Device {device.name} has low battery: {device.battery_level:.1f}%")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Failed to monitor devices: {e}")
                time.sleep(60)
    
    def _check_device_connectivity(self, device: IoTDevice) -> bool:
        """Check if device is reachable"""
        try:
            if device.protocol == IoTProtocol.HTTP:
                # Check HTTP connectivity
                response = requests.get(f"http://{device.ip_address}:{device.port}/health", timeout=5)
                return response.status_code == 200
            elif device.protocol == IoTProtocol.MQTT:
                # Check MQTT connectivity (simplified)
                return True  # Assume MQTT devices are always reachable
            else:
                # Check TCP connectivity
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((device.ip_address, device.port))
                sock.close()
                return result == 0
                
        except Exception:
            return False
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self._log_iot("mqtt_connected", "Connected to MQTT broker")
            # Subscribe to all device topics
            client.subscribe("iot/devices/+/data")
            client.subscribe("iot/devices/+/status")
        else:
            self._log_iot("mqtt_connection_failed", f"Failed to connect to MQTT broker: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Parse topic to get device ID
            topic_parts = topic.split('/')
            if len(topic_parts) >= 3:
                device_id = topic_parts[2]
                
                if topic.endswith('/data'):
                    # Handle sensor data
                    self._handle_mqtt_sensor_data(device_id, payload)
                elif topic.endswith('/status'):
                    # Handle device status
                    self._handle_mqtt_device_status(device_id, payload)
            
        except Exception as e:
            logger.error(f"Failed to handle MQTT message: {e}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self._log_iot("mqtt_disconnected", f"Disconnected from MQTT broker: {rc}")
    
    def _handle_mqtt_sensor_data(self, device_id: str, payload: Dict[str, Any]):
        """Handle MQTT sensor data"""
        try:
            if device_id in self.devices:
                sensor_data = IoTSensorData(
                    data_id=f"mqtt_{int(time.time() * 1000)}",
                    device_id=device_id,
                    sensor_type=payload.get("sensor_type", "unknown"),
                    value=payload.get("value", 0.0),
                    unit=payload.get("unit", "units"),
                    location=payload.get("location"),
                    metadata=payload.get("metadata", {})
                )
                
                with self.data_lock:
                    self.data_buffer.append(sensor_data)
                    
                    if device_id not in self.sensor_data:
                        self.sensor_data[device_id] = []
                    self.sensor_data[device_id].append(sensor_data)
                
                self._log_iot("mqtt_data_received", f"Received data from device {device_id}")
                
        except Exception as e:
            logger.error(f"Failed to handle MQTT sensor data: {e}")
    
    def _handle_mqtt_device_status(self, device_id: str, payload: Dict[str, Any]):
        """Handle MQTT device status"""
        try:
            if device_id in self.devices:
                device = self.devices[device_id]
                device.status = payload.get("status", "unknown")
                device.last_seen = datetime.utcnow()
                device.battery_level = payload.get("battery_level")
                device.signal_strength = payload.get("signal_strength")
                
                self._log_iot("mqtt_status_received", f"Received status from device {device_id}")
                
        except Exception as e:
            logger.error(f"Failed to handle MQTT device status: {e}")
    
    def add_iot_device(self, name: str, type: IoTDeviceType, protocol: IoTProtocol,
                      ip_address: str, port: int, configuration: Dict[str, Any]) -> str:
        """Add IoT device"""
        try:
            device_id = f"device_{int(time.time() * 1000)}"
            
            device = IoTDevice(
                device_id=device_id,
                name=name,
                type=type,
                protocol=protocol,
                ip_address=ip_address,
                port=port,
                status="offline",
                configuration=configuration
            )
            
            self.devices[device_id] = device
            
            # Initialize sensor data buffer
            self.sensor_data[device_id] = []
            
            self._log_iot("device_added", f"Added device {name} with ID {device_id}")
            
            return device_id
            
        except Exception as e:
            logger.error(f"Failed to add IoT device: {e}")
            raise
    
    def remove_iot_device(self, device_id: str) -> bool:
        """Remove IoT device"""
        try:
            if device_id in self.devices:
                device_name = self.devices[device_id].name
                del self.devices[device_id]
                
                # Remove sensor data
                if device_id in self.sensor_data:
                    del self.sensor_data[device_id]
                
                self._log_iot("device_removed", f"Removed device {device_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove IoT device: {e}")
            return False
    
    def get_device_data(self, device_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get device sensor data"""
        try:
            if device_id not in self.sensor_data:
                return []
            
            data = self.sensor_data[device_id][-limit:]
            
            return [
                {
                    "data_id": d.data_id,
                    "device_id": d.device_id,
                    "sensor_type": d.sensor_type,
                    "value": d.value,
                    "unit": d.unit,
                    "timestamp": d.timestamp.isoformat(),
                    "location": d.location,
                    "metadata": d.metadata
                }
                for d in data
            ]
            
        except Exception as e:
            logger.error(f"Failed to get device data: {e}")
            return []
    
    def get_device_summary(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device summary"""
        try:
            if device_id not in self.devices:
                return None
            
            device = self.devices[device_id]
            data_count = len(self.sensor_data.get(device_id, []))
            
            # Calculate statistics
            if device_id in self.sensor_data and self.sensor_data[device_id]:
                values = [d.value for d in self.sensor_data[device_id]]
                stats = {
                    "min_value": min(values),
                    "max_value": max(values),
                    "avg_value": np.mean(values),
                    "std_value": np.std(values)
                }
            else:
                stats = {}
            
            return {
                "device_id": device_id,
                "name": device.name,
                "type": device.type.value,
                "protocol": device.protocol.value,
                "ip_address": device.ip_address,
                "port": device.port,
                "status": device.status,
                "last_seen": device.last_seen.isoformat(),
                "battery_level": device.battery_level,
                "signal_strength": device.signal_strength,
                "data_count": data_count,
                "statistics": stats,
                "configuration": device.configuration
            }
            
        except Exception as e:
            logger.error(f"Failed to get device summary: {e}")
            return None
    
    def control_actuator(self, device_id: str, action: str, parameters: Dict[str, Any]) -> bool:
        """Control IoT actuator"""
        try:
            if device_id not in self.devices:
                return False
            
            device = self.devices[device_id]
            if device.type != IoTDeviceType.ACTUATOR:
                return False
            
            if device.protocol == IoTProtocol.HTTP:
                # Send HTTP command
                url = f"http://{device.ip_address}:{device.port}/control"
                payload = {
                    "action": action,
                    "parameters": parameters,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                response = requests.post(url, json=payload, timeout=10)
                success = response.status_code == 200
                
            elif device.protocol == IoTProtocol.MQTT:
                # Send MQTT command
                topic = f"iot/devices/{device_id}/control"
                payload = {
                    "action": action,
                    "parameters": parameters,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                self.mqtt_client.publish(topic, json.dumps(payload))
                success = True
                
            else:
                success = False
            
            if success:
                self._log_iot("actuator_controlled", f"Controlled actuator {device.name}: {action}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to control actuator: {e}")
            return False
    
    def get_iot_summary(self) -> Dict[str, Any]:
        """Get IoT system summary"""
        total_devices = len(self.devices)
        online_devices = len([d for d in self.devices.values() if d.status == "online"])
        offline_devices = total_devices - online_devices
        
        # Count by type
        type_counts = {}
        for device in self.devices.values():
            device_type = device.type.value
            type_counts[device_type] = type_counts.get(device_type, 0) + 1
        
        # Count by protocol
        protocol_counts = {}
        for device in self.devices.values():
            protocol = device.protocol.value
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        
        # Calculate total data points
        total_data_points = sum(len(data) for data in self.sensor_data.values())
        
        return {
            "total_devices": total_devices,
            "online_devices": online_devices,
            "offline_devices": offline_devices,
            "type_distribution": type_counts,
            "protocol_distribution": protocol_counts,
            "total_data_points": total_data_points,
            "data_buffer_size": len(self.data_buffer),
            "mqtt_connected": self.mqtt_client.is_connected() if self.mqtt_client else False
        }
    
    def _log_iot(self, event: str, message: str):
        """Log IoT event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if "iot_logs" not in self.iot_logs:
            self.iot_logs["iot_logs"] = []
        
        self.iot_logs["iot_logs"].append(log_entry)
        
        logger.info(f"IoT: {event} - {message}")
    
    def get_iot_logs(self) -> List[Dict[str, Any]]:
        """Get IoT logs"""
        return self.iot_logs.get("iot_logs", [])
    
    def shutdown(self):
        """Shutdown IoT system"""
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            
            self._log_iot("shutdown", "IoT system shutdown completed")
            
        except Exception as e:
            logger.error(f"Failed to shutdown IoT system: {e}")

# Global IoT instance
improvement_iot = None

def get_improvement_iot() -> RealImprovementIoT:
    """Get improvement IoT instance"""
    global improvement_iot
    if not improvement_iot:
        improvement_iot = RealImprovementIoT()
    return improvement_iot













