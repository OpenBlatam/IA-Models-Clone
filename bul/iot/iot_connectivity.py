"""
BUL IoT Connectivity System
===========================

IoT integration for smart document generation and real-time data processing.
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import aiohttp
import paho.mqtt.client as mqtt
import redis.asyncio as redis
import numpy as np
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class IoTDeviceType(str, Enum):
    """Types of IoT devices"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    SMART_DISPLAY = "smart_display"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"
    INDUSTRIAL = "industrial"
    SMART_HOME = "smart_home"

class DataType(str, Enum):
    """Types of IoT data"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    SOUND = "sound"
    MOTION = "motion"
    LOCATION = "location"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    JSON = "json"

class CommunicationProtocol(str, Enum):
    """IoT communication protocols"""
    MQTT = "mqtt"
    HTTP = "http"
    COAP = "coap"
    WEBSOCKET = "websocket"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    LORA = "lora"
    ZIGBEE = "zigbee"
    THREAD = "thread"
    MATTER = "matter"

@dataclass
class IoTDevice:
    """IoT device representation"""
    id: str
    name: str
    device_type: IoTDeviceType
    location: Dict[str, Any]
    capabilities: List[str]
    data_types: List[DataType]
    communication_protocol: CommunicationProtocol
    status: str  # online, offline, error, maintenance
    last_seen: datetime
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None
    firmware_version: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class IoTDataPoint:
    """IoT data point"""
    id: str
    device_id: str
    data_type: DataType
    value: Any
    unit: Optional[str]
    timestamp: datetime
    quality: float  # 0.0 to 1.0
    location: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

@dataclass
class IoTDataStream:
    """IoT data stream"""
    id: str
    name: str
    device_ids: List[str]
    data_types: List[DataType]
    sampling_rate: float  # Hz
    buffer_size: int
    processing_rules: List[Dict[str, Any]]
    active: bool
    created_at: datetime

@dataclass
class SmartDocumentTrigger:
    """Smart document generation trigger"""
    id: str
    name: str
    trigger_conditions: List[Dict[str, Any]]
    document_template: str
    data_sources: List[str]
    generation_rules: Dict[str, Any]
    active: bool
    last_triggered: Optional[datetime] = None

class IoTConnectivityManager:
    """IoT connectivity and data management system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        
        # IoT infrastructure
        self.devices: Dict[str, IoTDevice] = {}
        self.data_streams: Dict[str, IoTDataStream] = {}
        self.smart_triggers: Dict[str, SmartDocumentTrigger] = {}
        
        # Communication
        self.mqtt_client: Optional[mqtt.Client] = None
        self.redis_client: Optional[redis.Redis] = None
        self.http_client: Optional[aiohttp.ClientSession] = None
        
        # Data processing
        self.data_buffer: Dict[str, List[IoTDataPoint]] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}
        
        # Initialize IoT services
        self._initialize_iot_services()
    
    def _initialize_iot_services(self):
        """Initialize IoT connectivity services"""
        try:
            # Initialize MQTT client
            self._initialize_mqtt_client()
            
            # Initialize Redis for data caching
            self.redis_client = redis.from_url("redis://localhost:6379/2")
            
            # Initialize HTTP client
            self.http_client = aiohttp.ClientSession()
            
            # Start background tasks
            asyncio.create_task(self._device_monitor())
            asyncio.create_task(self._data_processor())
            asyncio.create_task(self._smart_trigger_processor())
            
            self.logger.info("IoT connectivity services initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize IoT services: {e}")
    
    def _initialize_mqtt_client(self):
        """Initialize MQTT client for IoT communication"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Connect to MQTT broker
            broker_host = getattr(self.config, 'mqtt_broker_host', 'localhost')
            broker_port = getattr(self.config, 'mqtt_broker_port', 1883)
            
            self.mqtt_client.connect(broker_host, broker_port, 60)
            self.mqtt_client.loop_start()
            
            self.logger.info(f"MQTT client connected to {broker_host}:{broker_port}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize MQTT client: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.logger.info("MQTT client connected successfully")
            # Subscribe to device topics
            client.subscribe("iot/devices/+/data")
            client.subscribe("iot/devices/+/status")
            client.subscribe("iot/devices/+/commands")
        else:
            self.logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Process message based on topic
            if "/data" in topic:
                asyncio.create_task(self._process_device_data(topic, payload))
            elif "/status" in topic:
                asyncio.create_task(self._process_device_status(topic, payload))
            elif "/commands" in topic:
                asyncio.create_task(self._process_device_command(topic, payload))
        
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.logger.warning(f"MQTT client disconnected with code {rc}")
    
    async def register_iot_device(
        self,
        device_id: str,
        name: str,
        device_type: IoTDeviceType,
        location: Dict[str, Any],
        capabilities: List[str],
        data_types: List[DataType],
        communication_protocol: CommunicationProtocol,
        metadata: Dict[str, Any] = None
    ) -> IoTDevice:
        """Register a new IoT device"""
        try:
            device = IoTDevice(
                id=device_id,
                name=name,
                device_type=device_type,
                location=location,
                capabilities=capabilities,
                data_types=data_types,
                communication_protocol=communication_protocol,
                status="online",
                last_seen=datetime.now(),
                metadata=metadata or {}
            )
            
            self.devices[device_id] = device
            
            # Subscribe to device topics
            if self.mqtt_client:
                self.mqtt_client.subscribe(f"iot/devices/{device_id}/+")
            
            # Initialize data buffer for device
            self.data_buffer[device_id] = []
            
            self.logger.info(f"IoT device registered: {name} ({device_type.value})")
            return device
        
        except Exception as e:
            self.logger.error(f"Error registering IoT device: {e}")
            raise
    
    async def _process_device_data(self, topic: str, payload: Dict[str, Any]):
        """Process device data from MQTT"""
        try:
            device_id = topic.split('/')[2]
            
            if device_id not in self.devices:
                self.logger.warning(f"Received data from unregistered device: {device_id}")
                return
            
            device = self.devices[device_id]
            
            # Create data point
            data_point = IoTDataPoint(
                id=str(uuid.uuid4()),
                device_id=device_id,
                data_type=DataType(payload.get('data_type', 'numeric')),
                value=payload.get('value'),
                unit=payload.get('unit'),
                timestamp=datetime.now(),
                quality=payload.get('quality', 1.0),
                location=payload.get('location'),
                metadata=payload.get('metadata', {})
            )
            
            # Add to data buffer
            self.data_buffer[device_id].append(data_point)
            
            # Limit buffer size
            max_buffer_size = 1000
            if len(self.data_buffer[device_id]) > max_buffer_size:
                self.data_buffer[device_id] = self.data_buffer[device_id][-max_buffer_size:]
            
            # Update device last seen
            device.last_seen = datetime.now()
            
            # Process smart triggers
            await self._check_smart_triggers(device_id, data_point)
            
            # Store in Redis for persistence
            if self.redis_client:
                await self.redis_client.lpush(
                    f"iot:data:{device_id}",
                    json.dumps(asdict(data_point), default=str)
                )
                await self.redis_client.expire(f"iot:data:{device_id}", 86400)  # 24 hours
        
        except Exception as e:
            self.logger.error(f"Error processing device data: {e}")
    
    async def _process_device_status(self, topic: str, payload: Dict[str, Any]):
        """Process device status update"""
        try:
            device_id = topic.split('/')[2]
            
            if device_id in self.devices:
                device = self.devices[device_id]
                device.status = payload.get('status', 'unknown')
                device.last_seen = datetime.now()
                device.battery_level = payload.get('battery_level')
                device.signal_strength = payload.get('signal_strength')
                
                self.logger.info(f"Device {device_id} status updated: {device.status}")
        
        except Exception as e:
            self.logger.error(f"Error processing device status: {e}")
    
    async def _process_device_command(self, topic: str, payload: Dict[str, Any]):
        """Process device command"""
        try:
            device_id = topic.split('/')[2]
            command = payload.get('command')
            parameters = payload.get('parameters', {})
            
            if device_id in self.devices:
                # Execute command on device
                result = await self._execute_device_command(device_id, command, parameters)
                
                # Send response back
                response_topic = f"iot/devices/{device_id}/response"
                response_payload = {
                    'command': command,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                
                if self.mqtt_client:
                    self.mqtt_client.publish(
                        response_topic,
                        json.dumps(response_payload)
                    )
        
        except Exception as e:
            self.logger.error(f"Error processing device command: {e}")
    
    async def _execute_device_command(
        self,
        device_id: str,
        command: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute command on IoT device"""
        try:
            device = self.devices[device_id]
            
            if command == "get_status":
                return {
                    'status': device.status,
                    'battery_level': device.battery_level,
                    'signal_strength': device.signal_strength,
                    'last_seen': device.last_seen.isoformat()
                }
            
            elif command == "get_data":
                data_type = parameters.get('data_type')
                limit = parameters.get('limit', 10)
                
                device_data = self.data_buffer.get(device_id, [])
                if data_type:
                    device_data = [dp for dp in device_data if dp.data_type.value == data_type]
                
                return {
                    'data_points': [asdict(dp) for dp in device_data[-limit:]],
                    'count': len(device_data)
                }
            
            elif command == "configure":
                # Update device configuration
                if 'sampling_rate' in parameters:
                    # Update sampling rate
                    pass
                
                return {'success': True, 'message': 'Configuration updated'}
            
            else:
                return {'success': False, 'message': f'Unknown command: {command}'}
        
        except Exception as e:
            self.logger.error(f"Error executing device command: {e}")
            return {'success': False, 'error': str(e)}
    
    async def create_data_stream(
        self,
        name: str,
        device_ids: List[str],
        data_types: List[DataType],
        sampling_rate: float,
        buffer_size: int,
        processing_rules: List[Dict[str, Any]]
    ) -> IoTDataStream:
        """Create IoT data stream"""
        try:
            stream_id = str(uuid.uuid4())
            
            stream = IoTDataStream(
                id=stream_id,
                name=name,
                device_ids=device_ids,
                data_types=data_types,
                sampling_rate=sampling_rate,
                buffer_size=buffer_size,
                processing_rules=processing_rules,
                active=True,
                created_at=datetime.now()
            )
            
            self.data_streams[stream_id] = stream
            
            # Start data processing thread
            self._start_data_processing_thread(stream)
            
            self.logger.info(f"Data stream created: {name}")
            return stream
        
        except Exception as e:
            self.logger.error(f"Error creating data stream: {e}")
            raise
    
    def _start_data_processing_thread(self, stream: IoTDataStream):
        """Start data processing thread for stream"""
        try:
            def process_stream_data():
                while stream.active:
                    try:
                        # Collect data from devices
                        stream_data = []
                        for device_id in stream.device_ids:
                            if device_id in self.data_buffer:
                                device_data = self.data_buffer[device_id]
                                # Filter by data types
                                filtered_data = [
                                    dp for dp in device_data
                                    if dp.data_type in stream.data_types
                                ]
                                stream_data.extend(filtered_data)
                        
                        # Apply processing rules
                        processed_data = self._apply_processing_rules(stream_data, stream.processing_rules)
                        
                        # Store processed data
                        if processed_data:
                            self._store_processed_data(stream.id, processed_data)
                        
                        time.sleep(1.0 / stream.sampling_rate)
                    
                    except Exception as e:
                        self.logger.error(f"Error in data processing thread: {e}")
                        time.sleep(1)
            
            thread = threading.Thread(target=process_stream_data, daemon=True)
            thread.start()
            self.processing_threads[stream.id] = thread
        
        except Exception as e:
            self.logger.error(f"Error starting data processing thread: {e}")
    
    def _apply_processing_rules(
        self,
        data_points: List[IoTDataPoint],
        processing_rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply processing rules to data points"""
        try:
            processed_data = []
            
            for rule in processing_rules:
                rule_type = rule.get('type')
                
                if rule_type == 'filter':
                    # Filter data based on conditions
                    filtered_data = self._filter_data(data_points, rule.get('conditions', {}))
                    processed_data.extend([asdict(dp) for dp in filtered_data])
                
                elif rule_type == 'aggregate':
                    # Aggregate data
                    aggregated = self._aggregate_data(data_points, rule.get('aggregation', {}))
                    processed_data.append(aggregated)
                
                elif rule_type == 'transform':
                    # Transform data
                    transformed = self._transform_data(data_points, rule.get('transformation', {}))
                    processed_data.extend(transformed)
            
            return processed_data
        
        except Exception as e:
            self.logger.error(f"Error applying processing rules: {e}")
            return []
    
    def _filter_data(
        self,
        data_points: List[IoTDataPoint],
        conditions: Dict[str, Any]
    ) -> List[IoTDataPoint]:
        """Filter data based on conditions"""
        try:
            filtered = data_points
            
            if 'min_value' in conditions:
                filtered = [dp for dp in filtered if dp.value >= conditions['min_value']]
            
            if 'max_value' in conditions:
                filtered = [dp for dp in filtered if dp.value <= conditions['max_value']]
            
            if 'data_type' in conditions:
                filtered = [dp for dp in filtered if dp.data_type.value == conditions['data_type']]
            
            if 'quality_threshold' in conditions:
                filtered = [dp for dp in filtered if dp.quality >= conditions['quality_threshold']]
            
            return filtered
        
        except Exception as e:
            self.logger.error(f"Error filtering data: {e}")
            return data_points
    
    def _aggregate_data(
        self,
        data_points: List[IoTDataPoint],
        aggregation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate data points"""
        try:
            agg_type = aggregation.get('type', 'average')
            
            if not data_points:
                return {}
            
            values = [dp.value for dp in data_points if isinstance(dp.value, (int, float))]
            
            if not values:
                return {}
            
            if agg_type == 'average':
                result = np.mean(values)
            elif agg_type == 'sum':
                result = np.sum(values)
            elif agg_type == 'min':
                result = np.min(values)
            elif agg_type == 'max':
                result = np.max(values)
            elif agg_type == 'count':
                result = len(values)
            else:
                result = np.mean(values)
            
            return {
                'aggregation_type': agg_type,
                'value': result,
                'count': len(values),
                'timestamp': datetime.now().isoformat(),
                'data_type': data_points[0].data_type.value if data_points else 'unknown'
            }
        
        except Exception as e:
            self.logger.error(f"Error aggregating data: {e}")
            return {}
    
    def _transform_data(
        self,
        data_points: List[IoTDataPoint],
        transformation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Transform data points"""
        try:
            transform_type = transformation.get('type', 'normalize')
            transformed = []
            
            for dp in data_points:
                if transform_type == 'normalize':
                    # Normalize value to 0-1 range
                    if isinstance(dp.value, (int, float)):
                        min_val = transformation.get('min_value', 0)
                        max_val = transformation.get('max_value', 100)
                        normalized = (dp.value - min_val) / (max_val - min_val)
                        transformed.append({
                            'original_value': dp.value,
                            'transformed_value': normalized,
                            'data_type': dp.data_type.value,
                            'timestamp': dp.timestamp.isoformat()
                        })
                
                elif transform_type == 'scale':
                    # Scale value by factor
                    if isinstance(dp.value, (int, float)):
                        scale_factor = transformation.get('scale_factor', 1.0)
                        scaled = dp.value * scale_factor
                        transformed.append({
                            'original_value': dp.value,
                            'transformed_value': scaled,
                            'data_type': dp.data_type.value,
                            'timestamp': dp.timestamp.isoformat()
                        })
            
            return transformed
        
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            return []
    
    def _store_processed_data(self, stream_id: str, processed_data: List[Dict[str, Any]]):
        """Store processed data"""
        try:
            # Store in cache
            cache_key = f"iot:processed:{stream_id}"
            self.cache_manager.set(cache_key, processed_data, ttl=3600)
            
            # Store in Redis for persistence
            if self.redis_client:
                asyncio.create_task(self._store_in_redis(cache_key, processed_data))
        
        except Exception as e:
            self.logger.error(f"Error storing processed data: {e}")
    
    async def _store_in_redis(self, key: str, data: List[Dict[str, Any]]):
        """Store data in Redis"""
        try:
            if self.redis_client:
                await self.redis_client.lpush(key, json.dumps(data))
                await self.redis_client.expire(key, 86400)  # 24 hours
        
        except Exception as e:
            self.logger.error(f"Error storing in Redis: {e}")
    
    async def create_smart_document_trigger(
        self,
        name: str,
        trigger_conditions: List[Dict[str, Any]],
        document_template: str,
        data_sources: List[str],
        generation_rules: Dict[str, Any]
    ) -> SmartDocumentTrigger:
        """Create smart document generation trigger"""
        try:
            trigger_id = str(uuid.uuid4())
            
            trigger = SmartDocumentTrigger(
                id=trigger_id,
                name=name,
                trigger_conditions=trigger_conditions,
                document_template=document_template,
                data_sources=data_sources,
                generation_rules=generation_rules,
                active=True
            )
            
            self.smart_triggers[trigger_id] = trigger
            
            self.logger.info(f"Smart document trigger created: {name}")
            return trigger
        
        except Exception as e:
            self.logger.error(f"Error creating smart document trigger: {e}")
            raise
    
    async def _check_smart_triggers(self, device_id: str, data_point: IoTDataPoint):
        """Check if smart document triggers should be activated"""
        try:
            for trigger in self.smart_triggers.values():
                if not trigger.active:
                    continue
                
                if device_id not in trigger.data_sources:
                    continue
                
                # Check trigger conditions
                if await self._evaluate_trigger_conditions(trigger, data_point):
                    await self._execute_smart_document_generation(trigger, data_point)
        
        except Exception as e:
            self.logger.error(f"Error checking smart triggers: {e}")
    
    async def _evaluate_trigger_conditions(
        self,
        trigger: SmartDocumentTrigger,
        data_point: IoTDataPoint
    ) -> bool:
        """Evaluate trigger conditions"""
        try:
            for condition in trigger.trigger_conditions:
                condition_type = condition.get('type')
                
                if condition_type == 'threshold':
                    # Check if value exceeds threshold
                    threshold = condition.get('threshold')
                    operator = condition.get('operator', '>')
                    
                    if isinstance(data_point.value, (int, float)):
                        if operator == '>' and data_point.value > threshold:
                            return True
                        elif operator == '<' and data_point.value < threshold:
                            return True
                        elif operator == '==' and data_point.value == threshold:
                            return True
                        elif operator == '>=' and data_point.value >= threshold:
                            return True
                        elif operator == '<=' and data_point.value <= threshold:
                            return True
                
                elif condition_type == 'pattern':
                    # Check for data patterns
                    pattern = condition.get('pattern')
                    if self._matches_pattern(data_point, pattern):
                        return True
                
                elif condition_type == 'time_based':
                    # Check time-based conditions
                    if self._matches_time_condition(condition):
                        return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error evaluating trigger conditions: {e}")
            return False
    
    def _matches_pattern(self, data_point: IoTDataPoint, pattern: Dict[str, Any]) -> bool:
        """Check if data point matches pattern"""
        try:
            pattern_type = pattern.get('type')
            
            if pattern_type == 'anomaly':
                # Simple anomaly detection
                return self._detect_anomaly(data_point)
            
            elif pattern_type == 'trend':
                # Trend detection
                return self._detect_trend(data_point, pattern)
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error matching pattern: {e}")
            return False
    
    def _detect_anomaly(self, data_point: IoTDataPoint) -> bool:
        """Simple anomaly detection"""
        try:
            # Get recent data for the same device and data type
            device_data = self.data_buffer.get(data_point.device_id, [])
            recent_data = [
                dp for dp in device_data[-10:]  # Last 10 data points
                if dp.data_type == data_point.data_type
                and isinstance(dp.value, (int, float))
            ]
            
            if len(recent_data) < 3:
                return False
            
            values = [dp.value for dp in recent_data]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return False
            
            # Check if current value is more than 2 standard deviations from mean
            z_score = abs(data_point.value - mean_val) / std_val
            return z_score > 2.0
        
        except Exception as e:
            self.logger.error(f"Error detecting anomaly: {e}")
            return False
    
    def _detect_trend(self, data_point: IoTDataPoint, pattern: Dict[str, Any]) -> bool:
        """Detect trend in data"""
        try:
            trend_type = pattern.get('trend_type', 'increasing')
            
            # Get recent data
            device_data = self.data_buffer.get(data_point.device_id, [])
            recent_data = [
                dp for dp in device_data[-5:]  # Last 5 data points
                if dp.data_type == data_point.data_type
                and isinstance(dp.value, (int, float))
            ]
            
            if len(recent_data) < 3:
                return False
            
            values = [dp.value for dp in recent_data]
            
            if trend_type == 'increasing':
                return all(values[i] <= values[i+1] for i in range(len(values)-1))
            elif trend_type == 'decreasing':
                return all(values[i] >= values[i+1] for i in range(len(values)-1))
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error detecting trend: {e}")
            return False
    
    def _matches_time_condition(self, condition: Dict[str, Any]) -> bool:
        """Check time-based conditions"""
        try:
            current_time = datetime.now()
            condition_type = condition.get('condition_type')
            
            if condition_type == 'hourly':
                # Trigger every hour
                return current_time.minute == 0
            elif condition_type == 'daily':
                # Trigger daily at specific time
                target_hour = condition.get('hour', 9)
                return current_time.hour == target_hour and current_time.minute == 0
            elif condition_type == 'weekly':
                # Trigger weekly on specific day
                target_day = condition.get('day', 0)  # Monday
                return current_time.weekday() == target_day and current_time.hour == 9
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error matching time condition: {e}")
            return False
    
    async def _execute_smart_document_generation(
        self,
        trigger: SmartDocumentTrigger,
        data_point: IoTDataPoint
    ):
        """Execute smart document generation"""
        try:
            # Collect relevant data
            document_data = await self._collect_document_data(trigger, data_point)
            
            # Generate document using template
            generated_document = await self._generate_document_from_template(
                trigger.document_template,
                document_data,
                trigger.generation_rules
            )
            
            # Store generated document
            document_id = str(uuid.uuid4())
            await self._store_generated_document(document_id, generated_document, trigger)
            
            # Update trigger last triggered time
            trigger.last_triggered = datetime.now()
            
            self.logger.info(f"Smart document generated: {document_id} from trigger {trigger.name}")
        
        except Exception as e:
            self.logger.error(f"Error executing smart document generation: {e}")
    
    async def _collect_document_data(
        self,
        trigger: SmartDocumentTrigger,
        data_point: IoTDataPoint
    ) -> Dict[str, Any]:
        """Collect data for document generation"""
        try:
            document_data = {
                'trigger_data': asdict(data_point),
                'timestamp': datetime.now().isoformat(),
                'device_info': asdict(self.devices.get(data_point.device_id, {})),
                'context_data': {}
            }
            
            # Collect additional data from data sources
            for source_id in trigger.data_sources:
                if source_id in self.data_buffer:
                    source_data = self.data_buffer[source_id]
                    document_data['context_data'][source_id] = [
                        asdict(dp) for dp in source_data[-10:]  # Last 10 data points
                    ]
            
            return document_data
        
        except Exception as e:
            self.logger.error(f"Error collecting document data: {e}")
            return {}
    
    async def _generate_document_from_template(
        self,
        template: str,
        data: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> str:
        """Generate document from template"""
        try:
            # Simple template processing
            generated_doc = template
            
            # Replace placeholders with data
            for key, value in data.items():
                placeholder = f"{{{key}}}"
                if placeholder in generated_doc:
                    generated_doc = generated_doc.replace(placeholder, str(value))
            
            # Apply generation rules
            if rules.get('add_timestamp'):
                generated_doc += f"\n\nGenerated at: {datetime.now().isoformat()}"
            
            if rules.get('add_device_info'):
                device_info = data.get('device_info', {})
                generated_doc += f"\n\nDevice: {device_info.get('name', 'Unknown')}"
            
            return generated_doc
        
        except Exception as e:
            self.logger.error(f"Error generating document from template: {e}")
            return template
    
    async def _store_generated_document(
        self,
        document_id: str,
        content: str,
        trigger: SmartDocumentTrigger
    ):
        """Store generated document"""
        try:
            document_data = {
                'id': document_id,
                'content': content,
                'trigger_id': trigger.id,
                'trigger_name': trigger.name,
                'generated_at': datetime.now().isoformat(),
                'template': trigger.document_template
            }
            
            # Store in cache
            self.cache_manager.set(f"iot:document:{document_id}", document_data, ttl=86400)
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.set(
                    f"iot:document:{document_id}",
                    json.dumps(document_data),
                    ex=86400
                )
        
        except Exception as e:
            self.logger.error(f"Error storing generated document: {e}")
    
    async def _device_monitor(self):
        """Monitor IoT device status"""
        while True:
            try:
                current_time = datetime.now()
                offline_devices = []
                
                for device_id, device in self.devices.items():
                    # Check if device is offline (no data for 5 minutes)
                    if (current_time - device.last_seen).total_seconds() > 300:
                        if device.status == "online":
                            device.status = "offline"
                            offline_devices.append(device_id)
                
                if offline_devices:
                    self.logger.warning(f"Devices went offline: {offline_devices}")
                
                await asyncio.sleep(60)  # Check every minute
            
            except Exception as e:
                self.logger.error(f"Error in device monitor: {e}")
                await asyncio.sleep(60)
    
    async def _data_processor(self):
        """Background data processor"""
        while True:
            try:
                # Process data streams
                for stream in self.data_streams.values():
                    if stream.active:
                        # Data processing is handled by individual threads
                        pass
                
                await asyncio.sleep(10)  # Check every 10 seconds
            
            except Exception as e:
                self.logger.error(f"Error in data processor: {e}")
                await asyncio.sleep(10)
    
    async def _smart_trigger_processor(self):
        """Background smart trigger processor"""
        while True:
            try:
                # Process time-based triggers
                for trigger in self.smart_triggers.values():
                    if not trigger.active:
                        continue
                    
                    # Check for time-based conditions
                    for condition in trigger.trigger_conditions:
                        if condition.get('type') == 'time_based':
                            if self._matches_time_condition(condition):
                                # Create dummy data point for time-based trigger
                                dummy_data_point = IoTDataPoint(
                                    id=str(uuid.uuid4()),
                                    device_id='system',
                                    data_type=DataType.TEXT,
                                    value='time_trigger',
                                    timestamp=datetime.now(),
                                    quality=1.0
                                )
                                await self._execute_smart_document_generation(trigger, dummy_data_point)
                
                await asyncio.sleep(60)  # Check every minute
            
            except Exception as e:
                self.logger.error(f"Error in smart trigger processor: {e}")
                await asyncio.sleep(60)
    
    async def get_iot_status(self) -> Dict[str, Any]:
        """Get IoT connectivity status"""
        try:
            total_devices = len(self.devices)
            online_devices = len([d for d in self.devices.values() if d.status == "online"])
            total_streams = len(self.data_streams)
            active_triggers = len([t for t in self.smart_triggers.values() if t.active])
            
            # Calculate data throughput
            total_data_points = sum(len(buffer) for buffer in self.data_buffer.values())
            
            return {
                'total_devices': total_devices,
                'online_devices': online_devices,
                'offline_devices': total_devices - online_devices,
                'total_data_streams': total_streams,
                'active_smart_triggers': active_triggers,
                'total_data_points': total_data_points,
                'mqtt_connected': self.mqtt_client is not None and self.mqtt_client.is_connected(),
                'redis_connected': self.redis_client is not None
            }
        
        except Exception as e:
            self.logger.error(f"Error getting IoT status: {e}")
            return {}

# Global IoT connectivity manager
_iot_connectivity_manager: Optional[IoTConnectivityManager] = None

def get_iot_connectivity_manager() -> IoTConnectivityManager:
    """Get the global IoT connectivity manager"""
    global _iot_connectivity_manager
    if _iot_connectivity_manager is None:
        _iot_connectivity_manager = IoTConnectivityManager()
    return _iot_connectivity_manager

# IoT router
iot_router = APIRouter(prefix="/iot", tags=["IoT Connectivity"])

@iot_router.post("/register-device")
async def register_iot_device_endpoint(
    device_id: str = Field(..., description="Device ID"),
    name: str = Field(..., description="Device name"),
    device_type: IoTDeviceType = Field(..., description="Device type"),
    location: Dict[str, Any] = Field(..., description="Device location"),
    capabilities: List[str] = Field(..., description="Device capabilities"),
    data_types: List[DataType] = Field(..., description="Data types"),
    communication_protocol: CommunicationProtocol = Field(..., description="Communication protocol"),
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Device metadata")
):
    """Register a new IoT device"""
    try:
        manager = get_iot_connectivity_manager()
        device = await manager.register_iot_device(
            device_id, name, device_type, location, capabilities,
            data_types, communication_protocol, metadata
        )
        return {"device": asdict(device), "success": True}
    
    except Exception as e:
        logger.error(f"Error registering IoT device: {e}")
        raise HTTPException(status_code=500, detail="Failed to register IoT device")

@iot_router.post("/create-data-stream")
async def create_data_stream_endpoint(
    name: str = Field(..., description="Stream name"),
    device_ids: List[str] = Field(..., description="Device IDs"),
    data_types: List[DataType] = Field(..., description="Data types"),
    sampling_rate: float = Field(..., description="Sampling rate (Hz)"),
    buffer_size: int = Field(..., description="Buffer size"),
    processing_rules: List[Dict[str, Any]] = Field(..., description="Processing rules")
):
    """Create IoT data stream"""
    try:
        manager = get_iot_connectivity_manager()
        stream = await manager.create_data_stream(
            name, device_ids, data_types, sampling_rate, buffer_size, processing_rules
        )
        return {"stream": asdict(stream), "success": True}
    
    except Exception as e:
        logger.error(f"Error creating data stream: {e}")
        raise HTTPException(status_code=500, detail="Failed to create data stream")

@iot_router.post("/create-smart-trigger")
async def create_smart_trigger_endpoint(
    name: str = Field(..., description="Trigger name"),
    trigger_conditions: List[Dict[str, Any]] = Field(..., description="Trigger conditions"),
    document_template: str = Field(..., description="Document template"),
    data_sources: List[str] = Field(..., description="Data sources"),
    generation_rules: Dict[str, Any] = Field(..., description="Generation rules")
):
    """Create smart document generation trigger"""
    try:
        manager = get_iot_connectivity_manager()
        trigger = await manager.create_smart_document_trigger(
            name, trigger_conditions, document_template, data_sources, generation_rules
        )
        return {"trigger": asdict(trigger), "success": True}
    
    except Exception as e:
        logger.error(f"Error creating smart trigger: {e}")
        raise HTTPException(status_code=500, detail="Failed to create smart trigger")

@iot_router.get("/status")
async def get_iot_status_endpoint():
    """Get IoT connectivity status"""
    try:
        manager = get_iot_connectivity_manager()
        status = await manager.get_iot_status()
        return {"status": status}
    
    except Exception as e:
        logger.error(f"Error getting IoT status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get IoT status")

@iot_router.get("/devices")
async def get_iot_devices_endpoint():
    """Get all IoT devices"""
    try:
        manager = get_iot_connectivity_manager()
        devices = [asdict(device) for device in manager.devices.values()]
        return {"devices": devices, "count": len(devices)}
    
    except Exception as e:
        logger.error(f"Error getting IoT devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get IoT devices")

@iot_router.get("/data/{device_id}")
async def get_device_data_endpoint(
    device_id: str,
    data_type: Optional[str] = None,
    limit: int = 100
):
    """Get data from IoT device"""
    try:
        manager = get_iot_connectivity_manager()
        
        if device_id not in manager.data_buffer:
            raise HTTPException(status_code=404, detail="Device not found")
        
        device_data = manager.data_buffer[device_id]
        
        if data_type:
            device_data = [dp for dp in device_data if dp.data_type.value == data_type]
        
        return {
            "device_id": device_id,
            "data_points": [asdict(dp) for dp in device_data[-limit:]],
            "count": len(device_data)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting device data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get device data")


