"""
IoT Service - Integración con IoT y sensores
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """Tipos de sensores"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    OCCUPANCY = "occupancy"
    LIGHT = "light"
    AIR_QUALITY = "air_quality"
    FOOT_TRAFFIC = "foot_traffic"
    NOISE = "noise"


class IoTService:
    """Servicio para integración con IoT"""
    
    def __init__(self):
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.sensors: Dict[str, List[Dict[str, Any]]] = {}
        self.readings: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_device(
        self,
        store_id: str,
        device_name: str,
        device_type: str,
        location: str,
        capabilities: List[str]
    ) -> Dict[str, Any]:
        """Registrar dispositivo IoT"""
        
        device_id = f"iot_{store_id}_{len(self.devices.get(store_id, [])) + 1}"
        
        device = {
            "device_id": device_id,
            "store_id": store_id,
            "name": device_name,
            "type": device_type,
            "location": location,
            "capabilities": capabilities,
            "is_active": True,
            "registered_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        }
        
        if store_id not in self.devices:
            self.devices[store_id] = {}
        
        self.devices[store_id][device_id] = device
        
        return device
    
    def add_sensor(
        self,
        device_id: str,
        sensor_type: SensorType,
        unit: str = "default"
    ) -> Dict[str, Any]:
        """Agregar sensor a dispositivo"""
        
        sensor_id = f"sensor_{device_id}_{len(self.sensors.get(device_id, [])) + 1}"
        
        sensor = {
            "sensor_id": sensor_id,
            "device_id": device_id,
            "type": sensor_type.value,
            "unit": unit,
            "is_active": True,
            "added_at": datetime.now().isoformat()
        }
        
        if device_id not in self.sensors:
            self.sensors[device_id] = []
        
        self.sensors[device_id].append(sensor)
        
        return sensor
    
    def record_reading(
        self,
        sensor_id: str,
        value: float,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registrar lectura de sensor"""
        
        reading = {
            "reading_id": f"read_{sensor_id}_{len(self.readings.get(sensor_id, [])) + 1}",
            "sensor_id": sensor_id,
            "value": value,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        
        if sensor_id not in self.readings:
            self.readings[sensor_id] = []
        
        self.readings[sensor_id].append(reading)
        
        # Mantener solo últimas 1000 lecturas
        if len(self.readings[sensor_id]) > 1000:
            self.readings[sensor_id] = self.readings[sensor_id][-1000:]
        
        return reading
    
    def get_sensor_readings(
        self,
        sensor_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener lecturas de sensor"""
        
        readings = self.readings.get(sensor_id, [])
        
        if start_time:
            readings = [r for r in readings if datetime.fromisoformat(r["timestamp"]) >= start_time]
        
        if end_time:
            readings = [r for r in readings if datetime.fromisoformat(r["timestamp"]) <= end_time]
        
        # Ordenar por timestamp
        readings.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return readings[:limit]
    
    def get_store_analytics(
        self,
        store_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Obtener analytics de la tienda desde IoT"""
        
        devices = self.devices.get(store_id, {})
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        analytics = {
            "store_id": store_id,
            "period_hours": hours,
            "devices_count": len(devices),
            "sensors": [],
            "summary": {}
        }
        
        # Agregar datos de cada sensor
        for device_id, device in devices.items():
            sensors = self.sensors.get(device_id, [])
            
            for sensor in sensors:
                readings = self.get_sensor_readings(
                    sensor["sensor_id"],
                    start_time,
                    end_time,
                    limit=1000
                )
                
                if readings:
                    values = [r["value"] for r in readings]
                    
                    sensor_data = {
                        "sensor_id": sensor["sensor_id"],
                        "type": sensor["type"],
                        "device": device["name"],
                        "location": device["location"],
                        "readings_count": len(readings),
                        "average": sum(values) / len(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "latest": readings[0]["value"] if readings else None
                    }
                    
                    analytics["sensors"].append(sensor_data)
        
        # Resumen por tipo de sensor
        summary = {}
        for sensor_data in analytics["sensors"]:
            sensor_type = sensor_data["type"]
            if sensor_type not in summary:
                summary[sensor_type] = {
                    "count": 0,
                    "average": 0,
                    "latest": None
                }
            
            summary[sensor_type]["count"] += 1
            summary[sensor_type]["average"] = (
                (summary[sensor_type]["average"] * (summary[sensor_type]["count"] - 1) + sensor_data["average"]) /
                summary[sensor_type]["count"]
            )
            if sensor_data["latest"] is not None:
                summary[sensor_type]["latest"] = sensor_data["latest"]
        
        analytics["summary"] = summary
        
        return analytics
    
    def detect_anomalies(
        self,
        sensor_id: str,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Detectar anomalías en lecturas"""
        
        readings = self.readings.get(sensor_id, [])
        
        if len(readings) < 10:
            return []
        
        values = [r["value"] for r in readings[-100:]]  # Últimas 100
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        threshold_value = threshold or (mean + 2 * std_dev)
        
        anomalies = []
        for reading in readings[-100:]:
            if abs(reading["value"] - mean) > threshold_value:
                anomalies.append({
                    "reading": reading,
                    "deviation": reading["value"] - mean,
                    "severity": "high" if abs(reading["value"] - mean) > 3 * std_dev else "medium"
                })
        
        return anomalies




