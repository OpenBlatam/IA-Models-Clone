"""
Sistema de integración con dispositivos IoT
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


class DeviceType(str, Enum):
    """Tipos de dispositivos IoT"""
    CAMERA = "camera"
    SENSOR = "sensor"
    SMART_MIRROR = "smart_mirror"
    MOBILE_APP = "mobile_app"
    WEARABLE = "wearable"


@dataclass
class IoTDevice:
    """Dispositivo IoT"""
    id: str
    user_id: str
    type: DeviceType
    name: str
    model: Optional[str] = None
    firmware_version: Optional[str] = None
    capabilities: List[str] = None
    connected: bool = True
    last_seen: str = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.now().isoformat()
        if self.capabilities is None:
            self.capabilities = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type.value,
            "name": self.name,
            "model": self.model,
            "firmware_version": self.firmware_version,
            "capabilities": self.capabilities,
            "connected": self.connected,
            "last_seen": self.last_seen,
            "metadata": self.metadata
        }


class IoTIntegration:
    """Sistema de integración IoT"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.devices: Dict[str, IoTDevice] = {}
        self.device_data: Dict[str, List[Dict]] = {}  # device_id -> [data_points]
    
    def register_device(self, user_id: str, device_type: DeviceType,
                       name: str, model: Optional[str] = None,
                       capabilities: Optional[List[str]] = None) -> str:
        """
        Registra un dispositivo IoT
        
        Args:
            user_id: ID del usuario
            device_type: Tipo de dispositivo
            name: Nombre del dispositivo
            model: Modelo (opcional)
            capabilities: Capacidades (opcional)
            
        Returns:
            ID del dispositivo
        """
        device_id = str(uuid.uuid4())
        
        device = IoTDevice(
            id=device_id,
            user_id=user_id,
            type=device_type,
            name=name,
            model=model,
            capabilities=capabilities or []
        )
        
        self.devices[device_id] = device
        return device_id
    
    def get_device(self, device_id: str) -> Optional[IoTDevice]:
        """Obtiene un dispositivo"""
        return self.devices.get(device_id)
    
    def get_user_devices(self, user_id: str) -> List[IoTDevice]:
        """Obtiene dispositivos de un usuario"""
        return [d for d in self.devices.values() if d.user_id == user_id]
    
    def record_device_data(self, device_id: str, data: Dict):
        """
        Registra datos de un dispositivo
        
        Args:
            device_id: ID del dispositivo
            data: Datos del dispositivo
        """
        if device_id not in self.device_data:
            self.device_data[device_id] = []
        
        data_point = {
            **data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.device_data[device_id].append(data_point)
        
        # Mantener solo últimos 1000 puntos
        if len(self.device_data[device_id]) > 1000:
            self.device_data[device_id] = self.device_data[device_id][-1000:]
        
        # Actualizar last_seen
        if device_id in self.devices:
            self.devices[device_id].last_seen = datetime.now().isoformat()
    
    def get_device_data(self, device_id: str, limit: int = 100) -> List[Dict]:
        """Obtiene datos de un dispositivo"""
        data = self.device_data.get(device_id, [])
        return data[-limit:]
    
    def trigger_analysis_from_device(self, device_id: str, image_data: bytes) -> str:
        """
        Dispara análisis desde dispositivo IoT
        
        Args:
            device_id: ID del dispositivo
            image_data: Datos de imagen
            
        Returns:
            ID del análisis
        """
        # Placeholder - en producción se integraría con el analizador
        import hashlib
        analysis_id = hashlib.md5(f"{device_id}{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Registrar evento
        self.record_device_data(device_id, {
            "event": "analysis_triggered",
            "analysis_id": analysis_id,
            "data_size": len(image_data)
        })
        
        return analysis_id






