"""
IoT Integration - Sistema de integración IoT
============================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Tipos de dispositivos IoT"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    PRINTER = "printer"


class IoTIntegration:
    """Sistema de integración IoT"""
    
    def __init__(self):
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.device_data: Dict[str, List[Dict[str, Any]]] = {}
        self.commands: List[Dict[str, Any]] = []
    
    def register_device(self, device_id: str, device_type: DeviceType,
                       name: str, location: Optional[str] = None,
                       capabilities: Optional[List[str]] = None) -> Dict[str, Any]:
        """Registra un dispositivo IoT"""
        device = {
            "id": device_id,
            "type": device_type.value,
            "name": name,
            "location": location,
            "capabilities": capabilities or [],
            "status": "online",
            "registered_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        }
        
        self.devices[device_id] = device
        
        logger.info(f"Dispositivo IoT registrado: {device_id} - {name}")
        return device
    
    def send_data(self, device_id: str, data: Dict[str, Any]):
        """Envía datos desde un dispositivo"""
        device = self.devices.get(device_id)
        if not device:
            raise ValueError(f"Dispositivo no encontrado: {device_id}")
        
        data_entry = {
            "device_id": device_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if device_id not in self.device_data:
            self.device_data[device_id] = []
        
        self.device_data[device_id].append(data_entry)
        
        # Mantener solo últimos 10000 datos por dispositivo
        if len(self.device_data[device_id]) > 10000:
            self.device_data[device_id] = self.device_data[device_id][-10000:]
        
        # Actualizar last_seen
        device["last_seen"] = datetime.now().isoformat()
        
        logger.debug(f"Datos recibidos de dispositivo: {device_id}")
    
    def send_command(self, device_id: str, command: str, parameters: Optional[Dict[str, Any]] = None):
        """Envía comando a un dispositivo"""
        device = self.devices.get(device_id)
        if not device:
            raise ValueError(f"Dispositivo no encontrado: {device_id}")
        
        command_entry = {
            "device_id": device_id,
            "command": command,
            "parameters": parameters or {},
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        self.commands.append(command_entry)
        
        # Mantener solo últimos 1000 comandos
        if len(self.commands) > 1000:
            self.commands = self.commands[-1000:]
        
        logger.info(f"Comando enviado a dispositivo: {device_id} - {command}")
        return command_entry
    
    def get_device_data(self, device_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene datos de un dispositivo"""
        data = self.device_data.get(device_id, [])
        return sorted(data, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de un dispositivo"""
        device = self.devices.get(device_id)
        if not device:
            return None
        
        last_seen = datetime.fromisoformat(device["last_seen"])
        time_since_last_seen = (datetime.now() - last_seen).total_seconds()
        
        # Considerar offline si no se ha visto en 5 minutos
        if time_since_last_seen > 300:
            device["status"] = "offline"
        
        return {
            "device": device,
            "data_count": len(self.device_data.get(device_id, [])),
            "last_data": self.device_data.get(device_id, [])[-1] if self.device_data.get(device_id) else None
        }
    
    def list_devices(self, device_type: Optional[DeviceType] = None) -> List[Dict[str, Any]]:
        """Lista dispositivos"""
        devices = list(self.devices.values())
        
        if device_type:
            devices = [d for d in devices if d["type"] == device_type.value]
        
        return devices




