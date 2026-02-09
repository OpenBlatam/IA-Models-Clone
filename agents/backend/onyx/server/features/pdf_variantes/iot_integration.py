"""
PDF Variantes - Internet of Things Integration
=============================================

Internet of Things integration for smart PDF processing and device interaction.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class IoTDeviceType(str, Enum):
    """IoT device types."""
    SMART_PRINTER = "smart_printer"
    SMART_SCANNER = "smart_scanner"
    SMART_DISPLAY = "smart_display"
    SMART_SPEAKER = "smart_speaker"
    SMART_CAMERA = "smart_camera"
    SMART_SENSOR = "smart_sensor"
    SMART_LIGHT = "smart_light"
    SMART_LOCK = "smart_lock"
    SMART_THERMOSTAT = "smart_thermostat"
    SMART_PHONE = "smart_phone"
    SMART_TABLET = "smart_tablet"
    SMART_WATCH = "smart_watch"
    SMART_GLASSES = "smart_glasses"
    SMART_DESK = "smart_desk"
    SMART_CHAIR = "smart_chair"


class IoTProtocol(str, Enum):
    """IoT communication protocols."""
    MQTT = "mqtt"
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    COAP = "coap"
    AMQP = "amqp"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    ZIGBEE = "zigbee"
    Z_WAVE = "z_wave"
    LORA = "lora"
    NB_IOT = "nb_iot"


class IoTDeviceStatus(str, Enum):
    """IoT device status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SLEEPING = "sleeping"
    UPDATING = "updating"


@dataclass
class IoTDevice:
    """IoT device."""
    device_id: str
    name: str
    device_type: IoTDeviceType
    protocol: IoTProtocol
    status: IoTDeviceStatus
    capabilities: List[str]
    location: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    firmware_version: str = "1.0.0"
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None
    last_heartbeat: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    device_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type.value,
            "protocol": self.protocol.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "location": self.location,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "firmware_version": self.firmware_version,
            "battery_level": self.battery_level,
            "signal_strength": self.signal_strength,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "created_at": self.created_at.isoformat(),
            "device_data": self.device_data
        }


@dataclass
class IoTTask:
    """IoT task."""
    task_id: str
    device_id: str
    task_type: str
    priority: int
    data: Dict[str, Any]
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "device_id": self.device_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "data": self.data,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error_message": self.error_message
        }


@dataclass
class IoTSensorData:
    """IoT sensor data."""
    sensor_id: str
    device_id: str
    sensor_type: str
    value: Any
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    location: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": sensor_id,
            "device_id": self.device_id,
            "sensor_type": self.sensor_type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "metadata": self.metadata
        }


class InternetOfThingsIntegration:
    """Internet of Things integration for PDF processing."""
    
    def __init__(self):
        self.devices: Dict[str, IoTDevice] = {}
        self.tasks: Dict[str, IoTTask] = {}
        self.sensor_data: Dict[str, List[IoTSensorData]] = {}  # device_id -> sensor data
        self.device_groups: Dict[str, List[str]] = {}  # group_name -> device_ids
        self.automation_rules: Dict[str, Dict[str, Any]] = {}
        self.device_commands: Dict[str, List[Dict[str, Any]]] = {}  # device_id -> commands
        logger.info("Initialized Internet of Things Integration")
    
    async def register_device(
        self,
        device_id: str,
        name: str,
        device_type: IoTDeviceType,
        protocol: IoTProtocol,
        capabilities: List[str],
        location: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        mac_address: Optional[str] = None
    ) -> IoTDevice:
        """Register IoT device."""
        device = IoTDevice(
            device_id=device_id,
            name=name,
            device_type=device_type,
            protocol=protocol,
            status=IoTDeviceStatus.ONLINE,
            capabilities=capabilities,
            location=location,
            ip_address=ip_address,
            mac_address=mac_address,
            last_heartbeat=datetime.utcnow()
        )
        
        self.devices[device_id] = device
        self.sensor_data[device_id] = []
        self.device_commands[device_id] = []
        
        logger.info(f"Registered IoT device: {device_id}")
        return device
    
    async def submit_device_task(
        self,
        device_id: str,
        task_type: str,
        data: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """Submit task to IoT device."""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        task_id = f"iot_task_{device_id}_{datetime.utcnow().timestamp()}"
        
        task = IoTTask(
            task_id=task_id,
            device_id=device_id,
            task_type=task_type,
            priority=priority,
            data=data
        )
        
        self.tasks[task_id] = task
        
        # Start task execution
        asyncio.create_task(self._execute_device_task(task_id))
        
        logger.info(f"Submitted IoT task: {task_id}")
        return task_id
    
    async def _execute_device_task(self, task_id: str):
        """Execute IoT device task."""
        try:
            task = self.tasks[task_id]
            device = self.devices[task.device_id]
            
            # Update task status
            task.status = "running"
            task.started_at = datetime.utcnow()
            
            # Simulate task execution based on device type and task type
            result = await self._process_device_task(task, device)
            
            # Complete task
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.result = result
            
            logger.info(f"Completed IoT task: {task_id}")
            
        except Exception as e:
            task = self.tasks[task_id]
            task.status = "failed"
            task.error_message = str(e)
            logger.error(f"IoT task failed {task_id}: {e}")
    
    async def _process_device_task(self, task: IoTTask, device: IoTDevice) -> Dict[str, Any]:
        """Process device task based on device type."""
        task_type = task.task_type
        data = task.data
        
        if device.device_type == IoTDeviceType.SMART_PRINTER:
            return await self._process_printer_task(task_type, data)
        elif device.device_type == IoTDeviceType.SMART_SCANNER:
            return await self._process_scanner_task(task_type, data)
        elif device.device_type == IoTDeviceType.SMART_DISPLAY:
            return await self._process_display_task(task_type, data)
        elif device.device_type == IoTDeviceType.SMART_SPEAKER:
            return await self._process_speaker_task(task_type, data)
        elif device.device_type == IoTDeviceType.SMART_CAMERA:
            return await self._process_camera_task(task_type, data)
        elif device.device_type == IoTDeviceType.SMART_SENSOR:
            return await self._process_sensor_task(task_type, data)
        else:
            return {"error": "Unknown device type"}
    
    async def _process_printer_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process smart printer task."""
        if task_type == "print_pdf":
            return {
                "task_type": "print_pdf",
                "pages_printed": data.get("pages", 1),
                "print_quality": data.get("quality", "high"),
                "result": "PDF printed successfully",
                "printer_status": "ready"
            }
        elif task_type == "print_status":
            return {
                "task_type": "print_status",
                "ink_level": 85,
                "paper_level": 90,
                "status": "ready"
            }
        else:
            return {"error": "Unknown printer task"}
    
    async def _process_scanner_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process smart scanner task."""
        if task_type == "scan_document":
            return {
                "task_type": "scan_document",
                "pages_scanned": data.get("pages", 1),
                "resolution": data.get("resolution", "300dpi"),
                "format": data.get("format", "PDF"),
                "result": "Document scanned successfully"
            }
        elif task_type == "scan_status":
            return {
                "task_type": "scan_status",
                "scanner_status": "ready",
                "calibration": "ok"
            }
        else:
            return {"error": "Unknown scanner task"}
    
    async def _process_display_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process smart display task."""
        if task_type == "display_pdf":
            return {
                "task_type": "display_pdf",
                "document_id": data.get("document_id"),
                "page_number": data.get("page_number", 1),
                "zoom_level": data.get("zoom_level", 100),
                "result": "PDF displayed successfully"
            }
        elif task_type == "display_status":
            return {
                "task_type": "display_status",
                "brightness": 80,
                "resolution": "1920x1080",
                "status": "active"
            }
        else:
            return {"error": "Unknown display task"}
    
    async def _process_speaker_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process smart speaker task."""
        if task_type == "read_text":
            return {
                "task_type": "read_text",
                "text": data.get("text", ""),
                "voice": data.get("voice", "default"),
                "speed": data.get("speed", 1.0),
                "result": "Text read successfully"
            }
        elif task_type == "speaker_status":
            return {
                "task_type": "speaker_status",
                "volume": 70,
                "status": "ready"
            }
        else:
            return {"error": "Unknown speaker task"}
    
    async def _process_camera_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process smart camera task."""
        if task_type == "capture_document":
            return {
                "task_type": "capture_document",
                "resolution": data.get("resolution", "4K"),
                "format": data.get("format", "JPEG"),
                "result": "Document captured successfully"
            }
        elif task_type == "camera_status":
            return {
                "task_type": "camera_status",
                "battery_level": 90,
                "storage_available": "8GB",
                "status": "ready"
            }
        else:
            return {"error": "Unknown camera task"}
    
    async def _process_sensor_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process smart sensor task."""
        if task_type == "read_sensor":
            return {
                "task_type": "read_sensor",
                "sensor_type": data.get("sensor_type", "temperature"),
                "value": 22.5,
                "unit": "celsius",
                "timestamp": datetime.utcnow().isoformat()
            }
        elif task_type == "sensor_status":
            return {
                "task_type": "sensor_status",
                "battery_level": 95,
                "signal_strength": -45,
                "status": "active"
            }
        else:
            return {"error": "Unknown sensor task"}
    
    async def record_sensor_data(
        self,
        device_id: str,
        sensor_id: str,
        sensor_type: str,
        value: Any,
        unit: str,
        location: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IoTSensorData:
        """Record sensor data."""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        sensor_data = IoTSensorData(
            sensor_id=sensor_id,
            device_id=device_id,
            sensor_type=sensor_type,
            value=value,
            unit=unit,
            location=location,
            metadata=metadata or {}
        )
        
        self.sensor_data[device_id].append(sensor_data)
        
        # Keep only last 1000 sensor readings per device
        if len(self.sensor_data[device_id]) > 1000:
            self.sensor_data[device_id] = self.sensor_data[device_id][-1000:]
        
        logger.info(f"Recorded sensor data: {sensor_id}")
        return sensor_data
    
    async def create_device_group(
        self,
        group_name: str,
        device_ids: List[str],
        group_type: str = "general"
    ) -> Dict[str, Any]:
        """Create device group."""
        # Validate devices exist
        for device_id in device_ids:
            if device_id not in self.devices:
                raise ValueError(f"Device {device_id} not found")
        
        group = {
            "group_name": group_name,
            "device_ids": device_ids,
            "group_type": group_type,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.device_groups[group_name] = device_ids
        
        logger.info(f"Created device group: {group_name}")
        return group
    
    async def send_group_command(
        self,
        group_name: str,
        command: str,
        data: Dict[str, Any]
    ) -> List[str]:
        """Send command to device group."""
        if group_name not in self.device_groups:
            raise ValueError(f"Device group {group_name} not found")
        
        device_ids = self.device_groups[group_name]
        task_ids = []
        
        for device_id in device_ids:
            task_id = await self.submit_device_task(
                device_id=device_id,
                task_type=command,
                data=data,
                priority=1
            )
            task_ids.append(task_id)
        
        logger.info(f"Sent group command to {len(device_ids)} devices")
        return task_ids
    
    async def create_automation_rule(
        self,
        rule_id: str,
        name: str,
        trigger_condition: Dict[str, Any],
        actions: List[Dict[str, Any]],
        enabled: bool = True
    ) -> Dict[str, Any]:
        """Create automation rule."""
        rule = {
            "rule_id": rule_id,
            "name": name,
            "trigger_condition": trigger_condition,
            "actions": actions,
            "enabled": enabled,
            "created_at": datetime.utcnow().isoformat(),
            "last_triggered": None,
            "trigger_count": 0
        }
        
        self.automation_rules[rule_id] = rule
        
        logger.info(f"Created automation rule: {rule_id}")
        return rule
    
    async def check_automation_rules(self, sensor_data: IoTSensorData):
        """Check automation rules against sensor data."""
        for rule_id, rule in self.automation_rules.items():
            if not rule["enabled"]:
                continue
            
            trigger_condition = rule["trigger_condition"]
            
            # Check if sensor data matches trigger condition
            if await self._evaluate_trigger_condition(sensor_data, trigger_condition):
                await self._execute_automation_actions(rule_id, rule["actions"])
                
                # Update rule statistics
                rule["last_triggered"] = datetime.utcnow().isoformat()
                rule["trigger_count"] += 1
    
    async def _evaluate_trigger_condition(
        self,
        sensor_data: IoTSensorData,
        condition: Dict[str, Any]
    ) -> bool:
        """Evaluate trigger condition."""
        condition_type = condition.get("type", "unknown")
        
        if condition_type == "sensor_value":
            sensor_type = condition.get("sensor_type")
            operator = condition.get("operator", "equals")
            threshold = condition.get("threshold")
            
            if sensor_data.sensor_type == sensor_type:
                if operator == "greater_than":
                    return float(sensor_data.value) > threshold
                elif operator == "less_than":
                    return float(sensor_data.value) < threshold
                elif operator == "equals":
                    return sensor_data.value == threshold
        
        return False
    
    async def _execute_automation_actions(self, rule_id: str, actions: List[Dict[str, Any]]):
        """Execute automation actions."""
        for action in actions:
            action_type = action.get("type", "unknown")
            
            if action_type == "send_command":
                device_id = action.get("device_id")
                command = action.get("command")
                data = action.get("data", {})
                
                if device_id in self.devices:
                    await self.submit_device_task(device_id, command, data)
            
            elif action_type == "send_notification":
                message = action.get("message", "Automation triggered")
                logger.info(f"Automation notification: {message}")
    
    async def update_device_heartbeat(self, device_id: str) -> bool:
        """Update device heartbeat."""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        device.last_heartbeat = datetime.utcnow()
        device.status = IoTDeviceStatus.ONLINE
        
        logger.debug(f"Updated heartbeat for device: {device_id}")
        return True
    
    async def get_device_status(self, device_id: str) -> Optional[IoTDevice]:
        """Get device status."""
        return self.devices.get(device_id)
    
    async def get_task_status(self, task_id: str) -> Optional[IoTTask]:
        """Get task status."""
        return self.tasks.get(task_id)
    
    async def get_sensor_data(
        self,
        device_id: str,
        sensor_type: Optional[str] = None,
        limit: int = 100
    ) -> List[IoTSensorData]:
        """Get sensor data."""
        if device_id not in self.sensor_data:
            return []
        
        data = self.sensor_data[device_id]
        
        if sensor_type:
            data = [d for d in data if d.sensor_type == sensor_type]
        
        return data[-limit:] if limit else data
    
    def get_iot_stats(self) -> Dict[str, Any]:
        """Get IoT integration statistics."""
        total_devices = len(self.devices)
        online_devices = sum(1 for d in self.devices.values() if d.status == IoTDeviceStatus.ONLINE)
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for t in self.tasks.values() if t.status == "completed")
        total_sensor_data = sum(len(data) for data in self.sensor_data.values())
        total_groups = len(self.device_groups)
        total_rules = len(self.automation_rules)
        
        return {
            "total_devices": total_devices,
            "online_devices": online_devices,
            "offline_devices": total_devices - online_devices,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "total_sensor_data": total_sensor_data,
            "total_groups": total_groups,
            "total_rules": total_rules,
            "device_types": list(set(d.device_type.value for d in self.devices.values())),
            "protocols": list(set(d.protocol.value for d in self.devices.values())),
            "automation_rules_enabled": sum(1 for r in self.automation_rules.values() if r["enabled"])
        }
    
    async def export_iot_data(self) -> Dict[str, Any]:
        """Export IoT data."""
        return {
            "devices": [device.to_dict() for device in self.devices.values()],
            "tasks": [task.to_dict() for task in self.tasks.values()],
            "sensor_data": {
                device_id: [data.to_dict() for data in data_list]
                for device_id, data_list in self.sensor_data.items()
            },
            "device_groups": self.device_groups,
            "automation_rules": self.automation_rules,
            "exported_at": datetime.utcnow().isoformat()
        }


# Global instance
internet_of_things_integration = InternetOfThingsIntegration()
