"""
IoT Service - Advanced Implementation
===================================

Advanced IoT service with device management, sensor data processing, and real-time monitoring.
"""

from __future__ import annotations
import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib

from .analytics_service import analytics_service
from .ai_service import ai_service

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Device type enumeration"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    SMART_DEVICE = "smart_device"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"
    INDUSTRIAL_EQUIPMENT = "industrial_equipment"


class SensorType(str, Enum):
    """Sensor type enumeration"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"
    SOUND = "sound"
    VIBRATION = "vibration"
    GPS = "gps"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    PROXIMITY = "proximity"
    AIR_QUALITY = "air_quality"
    WATER_QUALITY = "water_quality"
    POWER_CONSUMPTION = "power_consumption"


class DataType(str, Enum):
    """Data type enumeration"""
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    STRING = "string"
    JSON = "json"
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class IoTService:
    """Advanced IoT service with device management and sensor data processing"""
    
    def __init__(self):
        self.devices = {}
        self.sensors = {}
        self.sensor_data = {}
        self.device_groups = {}
        self.iot_networks = {}
        self.alerts = {}
        
        self.iot_stats = {
            "total_devices": 0,
            "active_devices": 0,
            "total_sensors": 0,
            "active_sensors": 0,
            "total_data_points": 0,
            "total_alerts": 0,
            "devices_by_type": {device_type.value: 0 for device_type in DeviceType},
            "sensors_by_type": {sensor_type.value: 0 for sensor_type in SensorType},
            "data_by_type": {data_type.value: 0 for data_type in DataType}
        }
        
        # IoT infrastructure
        self.device_protocols = {}
        self.data_processors = {}
        self.alert_rules = {}
        self.device_firmware = {}
    
    async def register_device(
        self,
        device_id: str,
        device_type: DeviceType,
        device_name: str,
        location: Dict[str, float],
        capabilities: List[str],
        protocol: str = "mqtt",
        firmware_version: str = "1.0.0"
    ) -> str:
        """Register a new IoT device"""
        try:
            device = {
                "id": device_id,
                "type": device_type.value,
                "name": device_name,
                "location": location,
                "capabilities": capabilities,
                "protocol": protocol,
                "firmware_version": firmware_version,
                "status": "active",
                "last_seen": datetime.utcnow().isoformat(),
                "registered_at": datetime.utcnow().isoformat(),
                "battery_level": 100.0,
                "signal_strength": 0.0,
                "data_sent": 0,
                "data_received": 0,
                "uptime": 0,
                "configuration": {}
            }
            
            self.devices[device_id] = device
            self.iot_stats["total_devices"] += 1
            self.iot_stats["active_devices"] += 1
            self.iot_stats["devices_by_type"][device_type.value] += 1
            
            # Setup device protocol
            await self._setup_device_protocol(device_id, protocol)
            
            logger.info(f"IoT device registered: {device_id} - {device_name}")
            return device_id
        
        except Exception as e:
            logger.error(f"Failed to register IoT device: {e}")
            raise
    
    async def add_sensor(
        self,
        device_id: str,
        sensor_id: str,
        sensor_type: SensorType,
        sensor_name: str,
        data_type: DataType,
        sampling_rate: float = 1.0,
        unit: str = "",
        calibration_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a sensor to a device"""
        try:
            if device_id not in self.devices:
                raise ValueError(f"Device not found: {device_id}")
            
            sensor = {
                "id": sensor_id,
                "device_id": device_id,
                "type": sensor_type.value,
                "name": sensor_name,
                "data_type": data_type.value,
                "sampling_rate": sampling_rate,
                "unit": unit,
                "calibration_data": calibration_data or {},
                "status": "active",
                "last_reading": None,
                "created_at": datetime.utcnow().isoformat(),
                "readings_count": 0,
                "min_value": None,
                "max_value": None,
                "avg_value": None
            }
            
            self.sensors[sensor_id] = sensor
            self.iot_stats["total_sensors"] += 1
            self.iot_stats["active_sensors"] += 1
            self.iot_stats["sensors_by_type"][sensor_type.value] += 1
            self.iot_stats["data_by_type"][data_type.value] += 1
            
            # Initialize sensor data storage
            self.sensor_data[sensor_id] = []
            
            logger.info(f"Sensor added: {sensor_id} to device {device_id}")
            return sensor_id
        
        except Exception as e:
            logger.error(f"Failed to add sensor: {e}")
            raise
    
    async def send_sensor_data(
        self,
        sensor_id: str,
        value: Any,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send sensor data"""
        try:
            if sensor_id not in self.sensors:
                raise ValueError(f"Sensor not found: {sensor_id}")
            
            sensor = self.sensors[sensor_id]
            device_id = sensor["device_id"]
            
            if device_id not in self.devices:
                raise ValueError(f"Device not found: {device_id}")
            
            data_point = {
                "id": str(uuid.uuid4()),
                "sensor_id": sensor_id,
                "device_id": device_id,
                "value": value,
                "timestamp": timestamp or datetime.utcnow(),
                "metadata": metadata or {},
                "processed": False
            }
            
            # Store sensor data
            self.sensor_data[sensor_id].append(data_point)
            
            # Update sensor statistics
            sensor["last_reading"] = data_point["timestamp"].isoformat()
            sensor["readings_count"] += 1
            
            # Update min/max/avg values
            await self._update_sensor_statistics(sensor_id, value)
            
            # Update device statistics
            device = self.devices[device_id]
            device["data_received"] += 1
            device["last_seen"] = datetime.utcnow().isoformat()
            
            # Update global statistics
            self.iot_stats["total_data_points"] += 1
            
            # Process sensor data
            await self._process_sensor_data(data_point)
            
            # Check for alerts
            await self._check_sensor_alerts(sensor_id, value, data_point)
            
            # Track analytics
            await analytics_service.track_event(
                "sensor_data_received",
                {
                    "sensor_id": sensor_id,
                    "device_id": device_id,
                    "sensor_type": sensor["type"],
                    "value": value,
                    "data_type": sensor["data_type"]
                }
            )
            
            logger.info(f"Sensor data received: {sensor_id} - {value}")
            return data_point["id"]
        
        except Exception as e:
            logger.error(f"Failed to send sensor data: {e}")
            raise
    
    async def get_sensor_data(
        self,
        sensor_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get sensor data within time range"""
        try:
            if sensor_id not in self.sensor_data:
                return []
            
            data_points = self.sensor_data[sensor_id]
            
            # Filter by time range
            if start_time:
                data_points = [dp for dp in data_points if dp["timestamp"] >= start_time]
            
            if end_time:
                data_points = [dp for dp in data_points if dp["timestamp"] <= end_time]
            
            # Sort by timestamp (newest first)
            data_points.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Limit results
            return data_points[:limit]
        
        except Exception as e:
            logger.error(f"Failed to get sensor data: {e}")
            return []
    
    async def create_device_group(
        self,
        group_name: str,
        device_ids: List[str],
        group_type: str = "logical",
        description: str = ""
    ) -> str:
        """Create a device group"""
        try:
            # Validate devices
            for device_id in device_ids:
                if device_id not in self.devices:
                    raise ValueError(f"Device not found: {device_id}")
            
            group_id = f"group_{len(self.device_groups) + 1}"
            
            device_group = {
                "id": group_id,
                "name": group_name,
                "device_ids": device_ids,
                "group_type": group_type,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            self.device_groups[group_id] = device_group
            
            logger.info(f"Device group created: {group_id} - {group_name}")
            return group_id
        
        except Exception as e:
            logger.error(f"Failed to create device group: {e}")
            raise
    
    async def create_iot_network(
        self,
        network_name: str,
        network_type: str,
        devices: List[str],
        network_config: Dict[str, Any]
    ) -> str:
        """Create an IoT network"""
        try:
            # Validate devices
            for device_id in devices:
                if device_id not in self.devices:
                    raise ValueError(f"Device not found: {device_id}")
            
            network_id = f"network_{len(self.iot_networks) + 1}"
            
            iot_network = {
                "id": network_id,
                "name": network_name,
                "type": network_type,
                "devices": devices,
                "config": network_config,
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "performance_metrics": {
                    "network_latency": 0.0,
                    "bandwidth": 0.0,
                    "reliability": 0.0,
                    "throughput": 0.0
                }
            }
            
            self.iot_networks[network_id] = iot_network
            
            logger.info(f"IoT network created: {network_id} - {network_name}")
            return network_id
        
        except Exception as e:
            logger.error(f"Failed to create IoT network: {e}")
            raise
    
    async def create_alert_rule(
        self,
        rule_name: str,
        sensor_id: str,
        condition: str,
        threshold: Any,
        alert_type: str = "notification",
        actions: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Create an alert rule"""
        try:
            if sensor_id not in self.sensors:
                raise ValueError(f"Sensor not found: {sensor_id}")
            
            rule_id = f"rule_{len(self.alert_rules) + 1}"
            
            alert_rule = {
                "id": rule_id,
                "name": rule_name,
                "sensor_id": sensor_id,
                "condition": condition,
                "threshold": threshold,
                "alert_type": alert_type,
                "actions": actions or [],
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "triggered_count": 0,
                "last_triggered": None
            }
            
            self.alert_rules[rule_id] = alert_rule
            
            logger.info(f"Alert rule created: {rule_id} - {rule_name}")
            return rule_id
        
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            raise
    
    async def send_device_command(
        self,
        device_id: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send command to device"""
        try:
            if device_id not in self.devices:
                raise ValueError(f"Device not found: {device_id}")
            
            device = self.devices[device_id]
            
            if device["status"] != "active":
                raise ValueError(f"Device is not active: {device_id}")
            
            command_id = str(uuid.uuid4())
            
            # Simulate command execution
            await asyncio.sleep(0.1)
            
            # Update device statistics
            device["data_sent"] += 1
            device["last_seen"] = datetime.utcnow().isoformat()
            
            # Track analytics
            await analytics_service.track_event(
                "device_command_sent",
                {
                    "device_id": device_id,
                    "command": command,
                    "parameters": parameters,
                    "device_type": device["type"]
                }
            )
            
            logger.info(f"Device command sent: {device_id} - {command}")
            return command_id
        
        except Exception as e:
            logger.error(f"Failed to send device command: {e}")
            raise
    
    async def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device status and metrics"""
        try:
            if device_id not in self.devices:
                return None
            
            device = self.devices[device_id]
            
            # Get device sensors
            device_sensors = [s for s in self.sensors.values() if s["device_id"] == device_id]
            
            return {
                "id": device["id"],
                "type": device["type"],
                "name": device["name"],
                "status": device["status"],
                "location": device["location"],
                "capabilities": device["capabilities"],
                "protocol": device["protocol"],
                "firmware_version": device["firmware_version"],
                "battery_level": device["battery_level"],
                "signal_strength": device["signal_strength"],
                "data_sent": device["data_sent"],
                "data_received": device["data_received"],
                "uptime": device["uptime"],
                "sensors_count": len(device_sensors),
                "last_seen": device["last_seen"],
                "registered_at": device["registered_at"]
            }
        
        except Exception as e:
            logger.error(f"Failed to get device status: {e}")
            return None
    
    async def get_sensor_status(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get sensor status and metrics"""
        try:
            if sensor_id not in self.sensors:
                return None
            
            sensor = self.sensors[sensor_id]
            
            return {
                "id": sensor["id"],
                "device_id": sensor["device_id"],
                "type": sensor["type"],
                "name": sensor["name"],
                "data_type": sensor["data_type"],
                "sampling_rate": sensor["sampling_rate"],
                "unit": sensor["unit"],
                "status": sensor["status"],
                "last_reading": sensor["last_reading"],
                "readings_count": sensor["readings_count"],
                "min_value": sensor["min_value"],
                "max_value": sensor["max_value"],
                "avg_value": sensor["avg_value"],
                "created_at": sensor["created_at"]
            }
        
        except Exception as e:
            logger.error(f"Failed to get sensor status: {e}")
            return None
    
    async def get_iot_stats(self) -> Dict[str, Any]:
        """Get IoT service statistics"""
        try:
            return {
                "total_devices": self.iot_stats["total_devices"],
                "active_devices": self.iot_stats["active_devices"],
                "total_sensors": self.iot_stats["total_sensors"],
                "active_sensors": self.iot_stats["active_sensors"],
                "total_data_points": self.iot_stats["total_data_points"],
                "total_alerts": self.iot_stats["total_alerts"],
                "devices_by_type": self.iot_stats["devices_by_type"],
                "sensors_by_type": self.iot_stats["sensors_by_type"],
                "data_by_type": self.iot_stats["data_by_type"],
                "total_groups": len(self.device_groups),
                "total_networks": len(self.iot_networks),
                "total_alert_rules": len(self.alert_rules),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get IoT stats: {e}")
            return {"error": str(e)}
    
    async def _setup_device_protocol(self, device_id: str, protocol: str):
        """Setup device protocol"""
        try:
            self.device_protocols[device_id] = {
                "device_id": device_id,
                "protocol": protocol,
                "status": "connected",
                "connected_at": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to setup device protocol: {e}")
    
    async def _update_sensor_statistics(self, sensor_id: str, value: Any):
        """Update sensor statistics"""
        try:
            sensor = self.sensors[sensor_id]
            
            # Convert value to float if possible
            try:
                numeric_value = float(value)
                
                if sensor["min_value"] is None or numeric_value < sensor["min_value"]:
                    sensor["min_value"] = numeric_value
                
                if sensor["max_value"] is None or numeric_value > sensor["max_value"]:
                    sensor["max_value"] = numeric_value
                
                # Update average (simplified)
                if sensor["avg_value"] is None:
                    sensor["avg_value"] = numeric_value
                else:
                    sensor["avg_value"] = (sensor["avg_value"] + numeric_value) / 2
            
            except (ValueError, TypeError):
                # Non-numeric value, skip statistics update
                pass
        
        except Exception as e:
            logger.error(f"Failed to update sensor statistics: {e}")
    
    async def _process_sensor_data(self, data_point: Dict[str, Any]):
        """Process sensor data"""
        try:
            # Mark as processed
            data_point["processed"] = True
            
            # Add to data processor if available
            sensor_id = data_point["sensor_id"]
            if sensor_id in self.data_processors:
                processor = self.data_processors[sensor_id]
                await processor.process(data_point)
        
        except Exception as e:
            logger.error(f"Failed to process sensor data: {e}")
    
    async def _check_sensor_alerts(self, sensor_id: str, value: Any, data_point: Dict[str, Any]):
        """Check for sensor alerts"""
        try:
            # Find alert rules for this sensor
            sensor_rules = [rule for rule in self.alert_rules.values() if rule["sensor_id"] == sensor_id and rule["status"] == "active"]
            
            for rule in sensor_rules:
                if await self._evaluate_alert_condition(value, rule["condition"], rule["threshold"]):
                    await self._trigger_alert(rule, data_point)
        
        except Exception as e:
            logger.error(f"Failed to check sensor alerts: {e}")
    
    async def _evaluate_alert_condition(self, value: Any, condition: str, threshold: Any) -> bool:
        """Evaluate alert condition"""
        try:
            # Convert value to float if possible
            try:
                numeric_value = float(value)
                numeric_threshold = float(threshold)
                
                if condition == "greater_than":
                    return numeric_value > numeric_threshold
                elif condition == "less_than":
                    return numeric_value < numeric_threshold
                elif condition == "equal_to":
                    return numeric_value == numeric_threshold
                elif condition == "not_equal_to":
                    return numeric_value != numeric_threshold
                else:
                    return False
            
            except (ValueError, TypeError):
                # Non-numeric comparison
                if condition == "equal_to":
                    return str(value) == str(threshold)
                elif condition == "not_equal_to":
                    return str(value) != str(threshold)
                else:
                    return False
        
        except Exception as e:
            logger.error(f"Failed to evaluate alert condition: {e}")
            return False
    
    async def _trigger_alert(self, rule: Dict[str, Any], data_point: Dict[str, Any]):
        """Trigger an alert"""
        try:
            alert_id = str(uuid.uuid4())
            
            alert = {
                "id": alert_id,
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "sensor_id": rule["sensor_id"],
                "alert_type": rule["alert_type"],
                "value": data_point["value"],
                "threshold": rule["threshold"],
                "condition": rule["condition"],
                "timestamp": datetime.utcnow().isoformat(),
                "status": "active",
                "actions_executed": []
            }
            
            self.alerts[alert_id] = alert
            
            # Update rule statistics
            rule["triggered_count"] += 1
            rule["last_triggered"] = datetime.utcnow().isoformat()
            
            # Update global statistics
            self.iot_stats["total_alerts"] += 1
            
            # Execute alert actions
            for action in rule["actions"]:
                await self._execute_alert_action(alert_id, action)
                alert["actions_executed"].append(action)
            
            # Track analytics
            await analytics_service.track_event(
                "iot_alert_triggered",
                {
                    "alert_id": alert_id,
                    "rule_id": rule["id"],
                    "sensor_id": rule["sensor_id"],
                    "alert_type": rule["alert_type"],
                    "value": data_point["value"]
                }
            )
            
            logger.info(f"Alert triggered: {alert_id} - {rule['name']}")
        
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    async def _execute_alert_action(self, alert_id: str, action: Dict[str, Any]):
        """Execute alert action"""
        try:
            action_type = action.get("type", "notification")
            
            if action_type == "notification":
                # Send notification
                await self._send_notification(alert_id, action)
            elif action_type == "command":
                # Send command to device
                await self._send_device_command_from_alert(alert_id, action)
            elif action_type == "webhook":
                # Send webhook
                await self._send_webhook(alert_id, action)
        
        except Exception as e:
            logger.error(f"Failed to execute alert action: {e}")
    
    async def _send_notification(self, alert_id: str, action: Dict[str, Any]):
        """Send notification"""
        try:
            # Simulate notification sending
            await asyncio.sleep(0.1)
            logger.info(f"Notification sent for alert: {alert_id}")
        
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def _send_device_command_from_alert(self, alert_id: str, action: Dict[str, Any]):
        """Send device command from alert"""
        try:
            device_id = action.get("device_id")
            command = action.get("command")
            parameters = action.get("parameters", {})
            
            if device_id and command:
                await self.send_device_command(device_id, command, parameters)
        
        except Exception as e:
            logger.error(f"Failed to send device command from alert: {e}")
    
    async def _send_webhook(self, alert_id: str, action: Dict[str, Any]):
        """Send webhook"""
        try:
            # Simulate webhook sending
            await asyncio.sleep(0.1)
            logger.info(f"Webhook sent for alert: {alert_id}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")


# Global IoT service instance
iot_service = IoTService()

