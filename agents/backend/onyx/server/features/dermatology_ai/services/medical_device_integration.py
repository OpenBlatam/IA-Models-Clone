"""
Sistema de integración con dispositivos médicos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class MedicalDevice:
    """Dispositivo médico"""
    id: str
    user_id: str
    device_type: str  # "dermascope", "skin_analyzer", "moisture_meter", "uv_meter"
    device_name: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    calibration_date: Optional[str] = None
    last_sync: Optional[str] = None
    registered_at: str = None
    
    def __post_init__(self):
        if self.registered_at is None:
            self.registered_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "calibration_date": self.calibration_date,
            "last_sync": self.last_sync,
            "registered_at": self.registered_at
        }


@dataclass
class DeviceReading:
    """Lectura de dispositivo"""
    id: str
    device_id: str
    reading_type: str
    value: float
    unit: str
    timestamp: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "device_id": self.device_id,
            "reading_type": self.reading_type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class MedicalDeviceIntegration:
    """Sistema de integración con dispositivos médicos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.devices: Dict[str, List[MedicalDevice]] = {}  # user_id -> [devices]
        self.readings: Dict[str, List[DeviceReading]] = {}  # device_id -> [readings]
    
    def register_device(self, user_id: str, device_type: str, device_name: str,
                      manufacturer: Optional[str] = None, model: Optional[str] = None,
                      calibration_date: Optional[str] = None) -> MedicalDevice:
        """Registra un dispositivo médico"""
        device = MedicalDevice(
            id=str(uuid.uuid4()),
            user_id=user_id,
            device_type=device_type,
            device_name=device_name,
            manufacturer=manufacturer,
            model=model,
            calibration_date=calibration_date
        )
        
        if user_id not in self.devices:
            self.devices[user_id] = []
        
        self.devices[user_id].append(device)
        return device
    
    def sync_device_reading(self, device_id: str, reading_type: str,
                           value: float, unit: str, timestamp: str,
                           metadata: Optional[Dict] = None) -> DeviceReading:
        """Sincroniza lectura de dispositivo"""
        reading = DeviceReading(
            id=str(uuid.uuid4()),
            device_id=device_id,
            reading_type=reading_type,
            value=value,
            unit=unit,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        if device_id not in self.readings:
            self.readings[device_id] = []
        
        self.readings[device_id].append(reading)
        
        # Actualizar última sincronización del dispositivo
        for user_devices in self.devices.values():
            for device in user_devices:
                if device.id == device_id:
                    device.last_sync = datetime.now().isoformat()
                    break
        
        return reading
    
    def get_device_readings(self, device_id: str, limit: int = 100) -> List[DeviceReading]:
        """Obtiene lecturas de un dispositivo"""
        readings = self.readings.get(device_id, [])
        readings.sort(key=lambda x: x.timestamp, reverse=True)
        return readings[:limit]
    
    def get_user_devices(self, user_id: str) -> List[MedicalDevice]:
        """Obtiene dispositivos del usuario"""
        return self.devices.get(user_id, [])
    
    def analyze_device_data(self, device_id: str, days: int = 30) -> Dict:
        """Analiza datos del dispositivo"""
        readings = self.get_device_readings(device_id)
        
        # Filtrar por días
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_readings = [
            r for r in readings
            if datetime.fromisoformat(r.timestamp).timestamp() >= cutoff
        ]
        
        if not recent_readings:
            return {"error": "No data available"}
        
        # Agrupar por tipo de lectura
        by_type = {}
        for reading in recent_readings:
            if reading.reading_type not in by_type:
                by_type[reading.reading_type] = []
            by_type[reading.reading_type].append(reading.value)
        
        # Calcular estadísticas
        analysis = {}
        for reading_type, values in by_type.items():
            analysis[reading_type] = {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return analysis






