"""
Device Manager
==============

IoT device management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceStatus(Enum):
    """Device status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class IoTDevice:
    """IoT device definition."""
    id: str
    name: str
    type: str
    status: DeviceStatus
    location: Optional[str] = None
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DeviceManager:
    """IoT device manager."""
    
    def __init__(self):
        self._devices: Dict[str, IoTDevice] = {}
        self._device_groups: Dict[str, List[str]] = {}  # group -> device_ids
    
    def register_device(
        self,
        device_id: str,
        name: str,
        device_type: str,
        location: Optional[str] = None
    ) -> IoTDevice:
        """Register IoT device."""
        device = IoTDevice(
            id=device_id,
            name=name,
            type=device_type,
            status=DeviceStatus.OFFLINE,
            location=location
        )
        
        self._devices[device_id] = device
        logger.info(f"Registered device: {device_id}")
        return device
    
    def update_device_status(self, device_id: str, status: DeviceStatus):
        """Update device status."""
        if device_id in self._devices:
            self._devices[device_id].status = status
            self._devices[device_id].last_seen = datetime.now()
            logger.info(f"Device {device_id} status: {status.value}")
    
    def add_to_group(self, device_id: str, group_name: str):
        """Add device to group."""
        if group_name not in self._device_groups:
            self._device_groups[group_name] = []
        
        if device_id not in self._device_groups[group_name]:
            self._device_groups[group_name].append(device_id)
            logger.info(f"Added device {device_id} to group {group_name}")
    
    def get_devices_by_type(self, device_type: str) -> List[IoTDevice]:
        """Get devices by type."""
        return [d for d in self._devices.values() if d.type == device_type]
    
    def get_online_devices(self) -> List[IoTDevice]:
        """Get online devices."""
        return [d for d in self._devices.values() if d.status == DeviceStatus.ONLINE]
    
    def get_device_stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        return {
            "total_devices": len(self._devices),
            "online_devices": len(self.get_online_devices()),
            "by_status": {
                status.value: sum(1 for d in self._devices.values() if d.status == status)
                for status in DeviceStatus
            },
            "by_type": {
                device_type: sum(1 for d in self._devices.values() if d.type == device_type)
                for device_type in set(d.type for d in self._devices.values())
            },
            "total_groups": len(self._device_groups)
        }















