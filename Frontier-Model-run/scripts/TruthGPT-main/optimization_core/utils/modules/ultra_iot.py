"""
Ultra-Advanced IoT Integration Module for TruthGPT
Implements comprehensive IoT device management, sensor networks, and edge processing.
"""

import asyncio
import json
import time
import random
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IoTDeviceType(Enum):
    """Types of IoT devices."""
    TEMPERATURE_SENSOR = "temperature_sensor"
    HUMIDITY_SENSOR = "humidity_sensor"
    PRESSURE_SENSOR = "pressure_sensor"
    MOTION_SENSOR = "motion_sensor"
    LIGHT_SENSOR = "light_sensor"
    SOUND_SENSOR = "sound_sensor"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    GPS_TRACKER = "gps_tracker"
    SMART_SWITCH = "smart_switch"
    SMART_LOCK = "smart_lock"
    SMART_THERMOSTAT = "smart_thermostat"
    AIR_QUALITY_SENSOR = "air_quality_sensor"
    WATER_LEAK_SENSOR = "water_leak_sensor"
    SMOKE_DETECTOR = "smoke_detector"

class IoTProtocol(Enum):
    """IoT communication protocols."""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    WEBSOCKET = "websocket"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"
    Z_WAVE = "z_wave"
    LORA = "lora"
    NB_IOT = "nb_iot"
    LTE_M = "lte_m"

class DataType(Enum):
    """Types of IoT data."""
    SENSOR_DATA = "sensor_data"
    IMAGE_DATA = "image_data"
    AUDIO_DATA = "audio_data"
    VIDEO_DATA = "video_data"
    LOCATION_DATA = "location_data"
    STATUS_DATA = "status_data"
    EVENT_DATA = "event_data"
    TELEMETRY_DATA = "telemetry_data"

@dataclass
class IoTDevice:
    """IoT device representation."""
    device_id: str
    device_type: IoTDeviceType
    device_name: str
    location: Tuple[float, float, float]  # (latitude, longitude, altitude)
    protocol: IoTProtocol
    status: str = "offline"
    battery_level: float = 100.0
    signal_strength: float = 0.0
    last_seen: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IoTData:
    """IoT data packet."""
    data_id: str
    device_id: str
    data_type: DataType
    timestamp: float
    value: Any
    unit: str = ""
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IoTGateway:
    """IoT gateway representation."""
    gateway_id: str
    gateway_name: str
    location: Tuple[float, float, float]
    supported_protocols: List[IoTProtocol]
    connected_devices: List[str] = field(default_factory=list)
    status: str = "offline"
    processing_capacity: int = 1000
    storage_capacity: float = 1000.0  # GB
    bandwidth: float = 100.0  # Mbps

@dataclass
class IoTRule:
    """IoT automation rule."""
    rule_id: str
    rule_name: str
    trigger_condition: Dict[str, Any]
    action: Dict[str, Any]
    enabled: bool = True
    priority: int = 1
    created_at: float = 0.0
    last_triggered: float = 0.0

class IoTDeviceManager:
    """
    IoT device management system.
    """

    def __init__(self):
        """Initialize the IoT device manager."""
        self.devices: Dict[str, IoTDevice] = {}
        self.gateways: Dict[str, IoTGateway] = {}
        self.data_streams: Dict[str, List[IoTData]] = {}
        self.rules: Dict[str, IoTRule] = {}
        self.device_groups: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            'total_devices': 0,
            'online_devices': 0,
            'total_data_points': 0,
            'rules_triggered': 0,
            'data_processed': 0,
            'alerts_generated': 0
        }
        
        logger.info("IoT Device Manager initialized")

    def register_device(self, device: IoTDevice) -> None:
        """
        Register an IoT device.

        Args:
            device: IoT device to register
        """
        self.devices[device.device_id] = device
        self.data_streams[device.device_id] = []
        
        # Group devices by type
        device_type = device.device_type.value
        if device_type not in self.device_groups:
            self.device_groups[device_type] = []
        self.device_groups[device_type].append(device.device_id)
        
        self.stats['total_devices'] += 1
        logger.info(f"IoT device {device.device_id} registered")

    def register_gateway(self, gateway: IoTGateway) -> None:
        """
        Register an IoT gateway.

        Args:
            gateway: IoT gateway to register
        """
        self.gateways[gateway.gateway_id] = gateway
        logger.info(f"IoT gateway {gateway.gateway_id} registered")

    async def connect_device(self, device_id: str) -> bool:
        """
        Connect an IoT device.

        Args:
            device_id: Device identifier

        Returns:
            Connection status
        """
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        device = self.devices[device_id]
        
        # Simulate connection process
        await asyncio.sleep(random.uniform(0.1, 1.0))
        
        device.status = "online"
        device.last_seen = time.time()
        device.signal_strength = random.uniform(70, 100)
        
        self.stats['online_devices'] += 1
        
        logger.info(f"Device {device_id} connected successfully")
        return True

    async def disconnect_device(self, device_id: str) -> bool:
        """
        Disconnect an IoT device.

        Args:
            device_id: Device identifier

        Returns:
            Disconnection status
        """
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        device = self.devices[device_id]
        device.status = "offline"
        
        if self.stats['online_devices'] > 0:
            self.stats['online_devices'] -= 1
        
        logger.info(f"Device {device_id} disconnected")
        return True

    async def send_data(self, device_id: str, data: IoTData) -> bool:
        """
        Send data from an IoT device.

        Args:
            device_id: Device identifier
            data: Data to send

        Returns:
            Send status
        """
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        device = self.devices[device_id]
        if device.status != "online":
            logger.error(f"Device {device_id} is offline")
            return False
        
        # Store data
        self.data_streams[device_id].append(data)
        
        # Update device status
        device.last_seen = time.time()
        
        # Update statistics
        self.stats['total_data_points'] += 1
        self.stats['data_processed'] += 1
        
        # Check rules
        await self._check_rules(device_id, data)
        
        logger.debug(f"Data sent from device {device_id}: {data.value}")
        return True

    async def _check_rules(self, device_id: str, data: IoTData) -> None:
        """Check automation rules for device data."""
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this device
            if self._rule_applies(rule, device_id, data):
                await self._execute_rule(rule, device_id, data)

    def _rule_applies(self, rule: IoTRule, device_id: str, data: IoTData) -> bool:
        """Check if a rule applies to the given data."""
        condition = rule.trigger_condition
        
        # Check device type
        if 'device_type' in condition:
            device = self.devices[device_id]
            if device.device_type.value != condition['device_type']:
                return False
        
        # Check data type
        if 'data_type' in condition:
            if data.data_type.value != condition['data_type']:
                return False
        
        # Check value conditions
        if 'value_range' in condition:
            min_val, max_val = condition['value_range']
            if not (min_val <= data.value <= max_val):
                return False
        
        # Check threshold conditions
        if 'threshold' in condition:
            threshold = condition['threshold']
            operator = condition.get('operator', '>')
            
            if operator == '>' and data.value <= threshold:
                return False
            elif operator == '<' and data.value >= threshold:
                return False
            elif operator == '=' and data.value != threshold:
                return False
        
        return True

    async def _execute_rule(self, rule: IoTRule, device_id: str, data: IoTData) -> None:
        """Execute an automation rule."""
        logger.info(f"Executing rule {rule.rule_name} for device {device_id}")
        
        action = rule.action
        rule.last_triggered = time.time()
        self.stats['rules_triggered'] += 1
        
        # Execute different types of actions
        if action['type'] == 'notification':
            await self._send_notification(action, device_id, data)
        elif action['type'] == 'device_control':
            await self._control_device(action, device_id)
        elif action['type'] == 'data_processing':
            await self._process_data(action, device_id, data)
        elif action['type'] == 'alert':
            await self._generate_alert(action, device_id, data)

    async def _send_notification(self, action: Dict[str, Any], device_id: str, data: IoTData) -> None:
        """Send notification based on rule action."""
        message = action.get('message', f"Alert from device {device_id}")
        logger.info(f"Notification: {message}")

    async def _control_device(self, action: Dict[str, Any], device_id: str) -> None:
        """Control device based on rule action."""
        target_device_id = action.get('target_device', device_id)
        command = action.get('command', 'toggle')
        
        logger.info(f"Controlling device {target_device_id}: {command}")

    async def _process_data(self, action: Dict[str, Any], device_id: str, data: IoTData) -> None:
        """Process data based on rule action."""
        processing_type = action.get('processing_type', 'aggregate')
        logger.info(f"Processing data from device {device_id}: {processing_type}")

    async def _generate_alert(self, action: Dict[str, Any], device_id: str, data: IoTData) -> None:
        """Generate alert based on rule action."""
        alert_level = action.get('level', 'warning')
        message = action.get('message', f"Alert from device {device_id}")
        
        self.stats['alerts_generated'] += 1
        logger.warning(f"ALERT [{alert_level}]: {message}")

    def create_rule(
        self,
        rule_name: str,
        trigger_condition: Dict[str, Any],
        action: Dict[str, Any],
        priority: int = 1
    ) -> IoTRule:
        """
        Create an automation rule.

        Args:
            rule_name: Name of the rule
            trigger_condition: Trigger condition
            action: Action to execute
            priority: Rule priority

        Returns:
            Created rule
        """
        rule = IoTRule(
            rule_id=str(uuid.uuid4()),
            rule_name=rule_name,
            trigger_condition=trigger_condition,
            action=action,
            priority=priority,
            created_at=time.time()
        )
        
        self.rules[rule.rule_id] = rule
        logger.info(f"Rule created: {rule_name}")
        return rule

    def get_device_data(self, device_id: str, limit: int = 100) -> List[IoTData]:
        """Get data from a specific device."""
        return self.data_streams.get(device_id, [])[-limit:]

    def get_devices_by_type(self, device_type: IoTDeviceType) -> List[IoTDevice]:
        """Get devices by type."""
        return [device for device in self.devices.values() if device.device_type == device_type]

    def get_online_devices(self) -> List[IoTDevice]:
        """Get all online devices."""
        return [device for device in self.devices.values() if device.status == "online"]

    def get_device_statistics(self) -> Dict[str, Any]:
        """Get device statistics."""
        return {
            'total_devices': self.stats['total_devices'],
            'online_devices': self.stats['online_devices'],
            'offline_devices': self.stats['total_devices'] - self.stats['online_devices'],
            'device_types': len(self.device_groups),
            'total_data_points': self.stats['total_data_points'],
            'rules_count': len(self.rules),
            'rules_triggered': self.stats['rules_triggered'],
            'alerts_generated': self.stats['alerts_generated']
        }

class IoTSensorNetwork:
    """
    IoT sensor network management.
    """

    def __init__(self, device_manager: IoTDeviceManager):
        """
        Initialize the IoT sensor network.

        Args:
            device_manager: IoT device manager instance
        """
        self.device_manager = device_manager
        self.sensor_clusters: Dict[str, List[str]] = {}
        self.data_aggregators: Dict[str, Callable] = {}
        self.network_topology: Dict[str, List[str]] = {}
        
        logger.info("IoT Sensor Network initialized")

    def create_sensor_cluster(self, cluster_id: str, sensor_ids: List[str]) -> None:
        """
        Create a sensor cluster.

        Args:
            cluster_id: Cluster identifier
            sensor_ids: List of sensor device IDs
        """
        self.sensor_clusters[cluster_id] = sensor_ids
        
        # Create network topology
        for sensor_id in sensor_ids:
            self.network_topology[sensor_id] = [s for s in sensor_ids if s != sensor_id]
        
        logger.info(f"Sensor cluster {cluster_id} created with {len(sensor_ids)} sensors")

    async def aggregate_sensor_data(self, cluster_id: str, time_window: float = 60.0) -> Dict[str, Any]:
        """
        Aggregate sensor data from a cluster.

        Args:
            cluster_id: Cluster identifier
            time_window: Time window in seconds

        Returns:
            Aggregated data
        """
        if cluster_id not in self.sensor_clusters:
            raise Exception(f"Cluster {cluster_id} not found")
        
        sensor_ids = self.sensor_clusters[cluster_id]
        current_time = time.time()
        
        aggregated_data = {
            'cluster_id': cluster_id,
            'timestamp': current_time,
            'sensor_count': len(sensor_ids),
            'data_points': {},
            'statistics': {}
        }
        
        # Collect data from all sensors in cluster
        for sensor_id in sensor_ids:
            sensor_data = self.device_manager.get_device_data(sensor_id, limit=100)
            
            # Filter data within time window
            recent_data = [
                data for data in sensor_data
                if current_time - data.timestamp <= time_window
            ]
            
            if recent_data:
                values = [data.value for data in recent_data]
                aggregated_data['data_points'][sensor_id] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'latest': values[-1] if values else None
                }
        
        # Calculate cluster statistics
        all_values = []
        for sensor_data in aggregated_data['data_points'].values():
            if sensor_data['latest'] is not None:
                all_values.append(sensor_data['latest'])
        
        if all_values:
            aggregated_data['statistics'] = {
                'cluster_mean': np.mean(all_values),
                'cluster_min': np.min(all_values),
                'cluster_max': np.max(all_values),
                'cluster_std': np.std(all_values),
                'data_quality': len(all_values) / len(sensor_ids)
            }
        
        return aggregated_data

    def set_data_aggregator(self, cluster_id: str, aggregator_func: Callable) -> None:
        """
        Set custom data aggregator for a cluster.

        Args:
            cluster_id: Cluster identifier
            aggregator_func: Aggregation function
        """
        self.data_aggregators[cluster_id] = aggregator_func
        logger.info(f"Custom aggregator set for cluster {cluster_id}")

class IoTEdgeProcessor:
    """
    IoT edge processing system.
    """

    def __init__(self, device_manager: IoTDeviceManager):
        """
        Initialize the IoT edge processor.

        Args:
            device_manager: IoT device manager instance
        """
        self.device_manager = device_manager
        self.processing_pipelines: Dict[str, List[Callable]] = {}
        self.edge_models: Dict[str, Any] = {}
        self.processing_stats: Dict[str, Any] = {}
        
        logger.info("IoT Edge Processor initialized")

    def create_processing_pipeline(
        self,
        pipeline_id: str,
        processing_steps: List[Callable]
    ) -> None:
        """
        Create a data processing pipeline.

        Args:
            pipeline_id: Pipeline identifier
            processing_steps: List of processing functions
        """
        self.processing_pipelines[pipeline_id] = processing_steps
        self.processing_stats[pipeline_id] = {
            'total_processed': 0,
            'processing_time': 0.0,
            'errors': 0
        }
        
        logger.info(f"Processing pipeline {pipeline_id} created with {len(processing_steps)} steps")

    async def process_data(
        self,
        pipeline_id: str,
        data: IoTData
    ) -> Any:
        """
        Process data through a pipeline.

        Args:
            pipeline_id: Pipeline identifier
            data: Data to process

        Returns:
            Processed data
        """
        if pipeline_id not in self.processing_pipelines:
            raise Exception(f"Pipeline {pipeline_id} not found")
        
        start_time = time.time()
        
        try:
            processed_data = data
            
            # Apply each processing step
            for step in self.processing_pipelines[pipeline_id]:
                processed_data = await step(processed_data)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats[pipeline_id]['total_processed'] += 1
            self.processing_stats[pipeline_id]['processing_time'] += processing_time
            
            logger.debug(f"Data processed through pipeline {pipeline_id} in {processing_time:.3f}s")
            return processed_data
            
        except Exception as e:
            self.processing_stats[pipeline_id]['errors'] += 1
            logger.error(f"Error processing data through pipeline {pipeline_id}: {e}")
            raise

    def deploy_edge_model(self, model_id: str, model: Any) -> None:
        """
        Deploy a machine learning model to the edge.

        Args:
            model_id: Model identifier
            model: Model to deploy
        """
        self.edge_models[model_id] = model
        logger.info(f"Edge model {model_id} deployed")

    async def run_edge_inference(
        self,
        model_id: str,
        input_data: Any
    ) -> Any:
        """
        Run inference on an edge model.

        Args:
            model_id: Model identifier
            input_data: Input data

        Returns:
            Inference result
        """
        if model_id not in self.edge_models:
            raise Exception(f"Model {model_id} not found")
        
        model = self.edge_models[model_id]
        
        # Simulate model inference
        await asyncio.sleep(random.uniform(0.01, 0.1))
        
        # Simple simulation - in real implementation, this would run the actual model
        result = {
            'model_id': model_id,
            'prediction': random.uniform(0, 1),
            'confidence': random.uniform(0.8, 1.0),
            'timestamp': time.time()
        }
        
        logger.debug(f"Edge inference completed for model {model_id}")
        return result

class TruthGPTIoTManager:
    """
    TruthGPT IoT Manager.
    Main orchestrator for IoT operations.
    """

    def __init__(self):
        """Initialize the TruthGPT IoT Manager."""
        self.device_manager = IoTDeviceManager()
        self.sensor_network = IoTSensorNetwork(self.device_manager)
        self.edge_processor = IoTEdgeProcessor(self.device_manager)
        
        # IoT statistics
        self.stats = {
            'total_devices_registered': 0,
            'total_data_processed': 0,
            'total_rules_executed': 0,
            'total_alerts_generated': 0,
            'network_uptime': 0.0,
            'data_throughput': 0.0
        }
        
        logger.info("TruthGPT IoT Manager initialized")

    async def initialize_iot_network(self) -> bool:
        """
        Initialize the IoT network.

        Returns:
            Initialization status
        """
        logger.info("Initializing TruthGPT IoT Network...")
        
        # Create sample devices
        await self._create_sample_devices()
        
        # Create sample rules
        self._create_sample_rules()
        
        # Create processing pipelines
        self._create_processing_pipelines()
        
        logger.info("TruthGPT IoT Network initialized successfully")
        return True

    async def _create_sample_devices(self) -> None:
        """Create sample IoT devices."""
        sample_devices = [
            IoTDevice(
                device_id="temp_sensor_001",
                device_type=IoTDeviceType.TEMPERATURE_SENSOR,
                device_name="Living Room Temperature Sensor",
                location=(40.7128, -74.0060, 10.0),
                protocol=IoTProtocol.MQTT,
                capabilities=["temperature_reading", "battery_monitoring"],
                configuration={"sampling_rate": 60, "threshold": 25.0}
            ),
            IoTDevice(
                device_id="humidity_sensor_001",
                device_type=IoTDeviceType.HUMIDITY_SENSOR,
                device_name="Kitchen Humidity Sensor",
                location=(40.7128, -74.0060, 10.0),
                protocol=IoTProtocol.MQTT,
                capabilities=["humidity_reading", "battery_monitoring"],
                configuration={"sampling_rate": 60, "threshold": 60.0}
            ),
            IoTDevice(
                device_id="motion_sensor_001",
                device_type=IoTDeviceType.MOTION_SENSOR,
                device_name="Front Door Motion Sensor",
                location=(40.7128, -74.0060, 10.0),
                protocol=IoTProtocol.ZIGBEE,
                capabilities=["motion_detection", "battery_monitoring"],
                configuration={"sensitivity": 0.8, "timeout": 30}
            ),
            IoTDevice(
                device_id="smart_switch_001",
                device_type=IoTDeviceType.SMART_SWITCH,
                device_name="Living Room Light Switch",
                location=(40.7128, -74.0060, 10.0),
                protocol=IoTProtocol.Z_WAVE,
                capabilities=["on_off_control", "dimming", "energy_monitoring"],
                configuration={"default_brightness": 80}
            )
        ]
        
        for device in sample_devices:
            self.device_manager.register_device(device)
            await self.device_manager.connect_device(device.device_id)
        
        self.stats['total_devices_registered'] += len(sample_devices)

    def _create_sample_rules(self) -> None:
        """Create sample automation rules."""
        # Temperature alert rule
        self.device_manager.create_rule(
            rule_name="High Temperature Alert",
            trigger_condition={
                'device_type': 'temperature_sensor',
                'threshold': 30.0,
                'operator': '>'
            },
            action={
                'type': 'alert',
                'level': 'warning',
                'message': 'High temperature detected!'
            },
            priority=1
        )
        
        # Motion detection rule
        self.device_manager.create_rule(
            rule_name="Motion Detection",
            trigger_condition={
                'device_type': 'motion_sensor',
                'value_range': [1, 1]
            },
            action={
                'type': 'device_control',
                'target_device': 'smart_switch_001',
                'command': 'turn_on'
            },
            priority=2
        )
        
        # Humidity monitoring rule
        self.device_manager.create_rule(
            rule_name="Humidity Monitoring",
            trigger_condition={
                'device_type': 'humidity_sensor',
                'threshold': 70.0,
                'operator': '>'
            },
            action={
                'type': 'notification',
                'message': 'High humidity detected in kitchen'
            },
            priority=3
        )

    def _create_processing_pipelines(self) -> None:
        """Create data processing pipelines."""
        # Temperature data processing pipeline
        async def temperature_processing(data: IoTData) -> IoTData:
            # Apply temperature calibration
            calibrated_value = data.value * 1.02 + 0.5
            data.value = calibrated_value
            return data
        
        async def data_validation(data: IoTData) -> IoTData:
            # Validate data quality
            if data.quality < 0.8:
                data.quality = 0.8
            return data
        
        self.edge_processor.create_processing_pipeline(
            pipeline_id="temperature_pipeline",
            processing_steps=[temperature_processing, data_validation]
        )

    async def simulate_iot_data(self, duration: int = 60) -> None:
        """
        Simulate IoT data generation.

        Args:
            duration: Simulation duration in seconds
        """
        logger.info(f"Starting IoT data simulation for {duration} seconds")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Generate data for each device
            for device in self.device_manager.get_online_devices():
                await self._generate_device_data(device)
            
            # Wait before next data generation
            await asyncio.sleep(1.0)
        
        logger.info("IoT data simulation completed")

    async def _generate_device_data(self, device: IoTDevice) -> None:
        """Generate data for a specific device."""
        data_value = None
        data_type = None
        unit = ""
        
        if device.device_type == IoTDeviceType.TEMPERATURE_SENSOR:
            data_value = random.uniform(18.0, 35.0)
            data_type = DataType.SENSOR_DATA
            unit = "°C"
        elif device.device_type == IoTDeviceType.HUMIDITY_SENSOR:
            data_value = random.uniform(30.0, 80.0)
            data_type = DataType.SENSOR_DATA
            unit = "%"
        elif device.device_type == IoTDeviceType.MOTION_SENSOR:
            data_value = random.choice([0, 1])
            data_type = DataType.EVENT_DATA
            unit = "binary"
        elif device.device_type == IoTDeviceType.SMART_SWITCH:
            data_value = random.choice([0, 1])
            data_type = DataType.STATUS_DATA
            unit = "binary"
        
        if data_value is not None:
            data = IoTData(
                data_id=str(uuid.uuid4()),
                device_id=device.device_id,
                data_type=data_type,
                timestamp=time.time(),
                value=data_value,
                unit=unit,
                quality=random.uniform(0.9, 1.0)
            )
            
            await self.device_manager.send_data(device.device_id, data)
            self.stats['total_data_processed'] += 1

    def get_iot_statistics(self) -> Dict[str, Any]:
        """Get comprehensive IoT statistics."""
        device_stats = self.device_manager.get_device_statistics()
        
        return {
            'device_statistics': device_stats,
            'sensor_clusters': len(self.sensor_network.sensor_clusters),
            'processing_pipelines': len(self.edge_processor.processing_pipelines),
            'edge_models': len(self.edge_processor.edge_models),
            'overall_statistics': self.stats
        }

# Utility functions
def create_iot_manager() -> TruthGPTIoTManager:
    """Create an IoT manager."""
    return TruthGPTIoTManager()

def create_iot_device(
    device_type: IoTDeviceType,
    device_name: str,
    location: Tuple[float, float, float],
    protocol: IoTProtocol = IoTProtocol.MQTT
) -> IoTDevice:
    """Create an IoT device."""
    return IoTDevice(
        device_id=str(uuid.uuid4()),
        device_type=device_type,
        device_name=device_name,
        location=location,
        protocol=protocol
    )

def create_iot_gateway(
    gateway_name: str,
    location: Tuple[float, float, float],
    supported_protocols: List[IoTProtocol]
) -> IoTGateway:
    """Create an IoT gateway."""
    return IoTGateway(
        gateway_id=str(uuid.uuid4()),
        gateway_name=gateway_name,
        location=location,
        supported_protocols=supported_protocols
    )

# Example usage
async def example_iot_integration():
    """Example of IoT integration."""
    print("🌐 Ultra IoT Integration Example")
    print("=" * 50)
    
    # Create IoT manager
    iot_manager = create_iot_manager()
    
    # Initialize IoT network
    initialized = await iot_manager.initialize_iot_network()
    if not initialized:
        print("❌ Failed to initialize IoT network")
        return
    
    print("✅ IoT network initialized successfully")
    
    # Get IoT statistics
    stats = iot_manager.get_iot_statistics()
    print(f"\n📊 IoT Statistics:")
    print(f"Total Devices: {stats['device_statistics']['total_devices']}")
    print(f"Online Devices: {stats['device_statistics']['online_devices']}")
    print(f"Device Types: {stats['device_statistics']['device_types']}")
    print(f"Rules Count: {stats['device_statistics']['rules_count']}")
    print(f"Sensor Clusters: {stats['sensor_clusters']}")
    print(f"Processing Pipelines: {stats['processing_pipelines']}")
    print(f"Edge Models: {stats['edge_models']}")
    
    # Create sensor cluster
    print(f"\n🔗 Creating sensor cluster...")
    iot_manager.sensor_network.create_sensor_cluster(
        cluster_id="living_room_cluster",
        sensor_ids=["temp_sensor_001", "humidity_sensor_001", "motion_sensor_001"]
    )
    
    # Simulate IoT data
    print(f"\n📡 Simulating IoT data...")
    await iot_manager.simulate_iot_data(duration=10)
    
    # Aggregate sensor data
    print(f"\n📈 Aggregating sensor data...")
    aggregated_data = await iot_manager.sensor_network.aggregate_sensor_data(
        cluster_id="living_room_cluster",
        time_window=60.0
    )
    
    print(f"Cluster: {aggregated_data['cluster_id']}")
    print(f"Sensor Count: {aggregated_data['sensor_count']}")
    print(f"Data Points: {len(aggregated_data['data_points'])}")
    
    if aggregated_data['statistics']:
        stats = aggregated_data['statistics']
        print(f"Cluster Mean: {stats['cluster_mean']:.2f}")
        print(f"Cluster Min: {stats['cluster_min']:.2f}")
        print(f"Cluster Max: {stats['cluster_max']:.2f}")
        print(f"Data Quality: {stats['data_quality']:.2f}")
    
    # Create processing pipeline
    print(f"\n⚙️ Creating processing pipeline...")
    async def data_enhancement(data: IoTData) -> IoTData:
        # Enhance data with additional metadata
        data.metadata['enhanced'] = True
        data.metadata['processing_timestamp'] = time.time()
        return data
    
    iot_manager.edge_processor.create_processing_pipeline(
        pipeline_id="data_enhancement_pipeline",
        processing_steps=[data_enhancement]
    )
    
    # Deploy edge model
    print(f"\n🤖 Deploying edge model...")
    iot_manager.edge_processor.deploy_edge_model(
        model_id="anomaly_detection_model",
        model="simulated_model"
    )
    
    # Run edge inference
    print(f"\n🔮 Running edge inference...")
    inference_result = await iot_manager.edge_processor.run_edge_inference(
        model_id="anomaly_detection_model",
        input_data={"temperature": 25.5, "humidity": 60.0}
    )
    
    print(f"Inference Result:")
    print(f"  Model: {inference_result['model_id']}")
    print(f"  Prediction: {inference_result['prediction']:.3f}")
    print(f"  Confidence: {inference_result['confidence']:.3f}")
    
    # Final statistics
    final_stats = iot_manager.get_iot_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"Total Data Processed: {final_stats['overall_statistics']['total_data_processed']}")
    print(f"Rules Executed: {final_stats['device_statistics']['rules_triggered']}")
    print(f"Alerts Generated: {final_stats['device_statistics']['alerts_generated']}")
    
    print("\n✅ IoT integration example completed successfully!")

if __name__ == "__main__":
    asyncio.run(example_iot_integration())

