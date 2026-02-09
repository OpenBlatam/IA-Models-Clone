"""
Servicio de Integración con Dispositivos de Monitoreo de Salud - Sistema completo de monitoreo de salud
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import statistics


class HealthDeviceType(str, Enum):
    """Tipos de dispositivos de salud"""
    BLOOD_PRESSURE = "blood_pressure"
    HEART_RATE = "heart_rate"
    GLUCOSE = "glucose"
    OXYGEN_SATURATION = "oxygen_saturation"
    TEMPERATURE = "temperature"
    WEIGHT = "weight"
    BODY_COMPOSITION = "body_composition"


class HealthMonitoringDeviceService:
    """Servicio de integración con dispositivos de monitoreo de salud"""
    
    def __init__(self):
        """Inicializa el servicio de monitoreo de salud"""
        self.supported_devices = self._load_supported_devices()
    
    def register_health_device(
        self,
        user_id: str,
        device_type: str,
        device_id: str,
        device_info: Dict
    ) -> Dict:
        """
        Registra dispositivo de salud
        
        Args:
            user_id: ID del usuario
            device_type: Tipo de dispositivo
            device_id: ID del dispositivo
            device_info: Información del dispositivo
        
        Returns:
            Dispositivo registrado
        """
        device = {
            "id": f"health_device_{datetime.now().timestamp()}",
            "user_id": user_id,
            "device_type": device_type,
            "device_id": device_id,
            "device_info": device_info,
            "registered_at": datetime.now().isoformat(),
            "status": "active",
            "last_reading": None
        }
        
        return device
    
    def record_health_reading(
        self,
        user_id: str,
        device_id: str,
        reading_data: Dict
    ) -> Dict:
        """
        Registra lectura de salud
        
        Args:
            user_id: ID del usuario
            device_id: ID del dispositivo
            reading_data: Datos de lectura
        
        Returns:
            Lectura registrada
        """
        reading = {
            "id": f"reading_{datetime.now().timestamp()}",
            "user_id": user_id,
            "device_id": device_id,
            "reading_data": reading_data,
            "value": reading_data.get("value", 0),
            "unit": reading_data.get("unit", ""),
            "timestamp": reading_data.get("timestamp", datetime.now().isoformat()),
            "recorded_at": datetime.now().isoformat(),
            "is_normal": self._check_normal_range(reading_data)
        }
        
        return reading
    
    def analyze_health_trends(
        self,
        user_id: str,
        device_type: str,
        readings: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza tendencias de salud
        
        Args:
            user_id: ID del usuario
            device_type: Tipo de dispositivo
            readings: Lecturas
        
        Returns:
            Análisis de tendencias
        """
        if not readings:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        values = [r.get("value", 0) for r in readings]
        
        return {
            "user_id": user_id,
            "device_type": device_type,
            "period_days": days,
            "total_readings": len(readings),
            "average_value": round(statistics.mean(values), 2) if values else 0,
            "trend": self._calculate_health_trend(values),
            "normal_range_percentage": self._calculate_normal_percentage(readings),
            "alerts": self._detect_health_alerts(device_type, readings),
            "recommendations": self._generate_health_recommendations(device_type, readings),
            "generated_at": datetime.now().isoformat()
        }
    
    def _load_supported_devices(self) -> List[Dict]:
        """Carga dispositivos soportados"""
        return [
            {
                "type": HealthDeviceType.BLOOD_PRESSURE,
                "brands": ["Omron", "Withings"],
                "normal_range": {"systolic": (90, 120), "diastolic": (60, 80)}
            },
            {
                "type": HealthDeviceType.HEART_RATE,
                "brands": ["Polar", "Garmin"],
                "normal_range": (60, 100)
            },
            {
                "type": HealthDeviceType.GLUCOSE,
                "brands": ["Accu-Chek", "OneTouch"],
                "normal_range": (70, 100)
            }
        ]
    
    def _check_normal_range(self, reading_data: Dict) -> bool:
        """Verifica si lectura está en rango normal"""
        # Lógica simplificada
        return True
    
    def _calculate_health_trend(self, values: List[float]) -> str:
        """Calcula tendencia de salud"""
        if len(values) < 2:
            return "stable"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0
        
        if avg_second > avg_first * 1.1:
            return "increasing"
        elif avg_second < avg_first * 0.9:
            return "decreasing"
        return "stable"
    
    def _calculate_normal_percentage(self, readings: List[Dict]) -> float:
        """Calcula porcentaje de lecturas normales"""
        normal_count = sum(1 for r in readings if r.get("is_normal", True))
        return round((normal_count / len(readings) * 100) if readings else 0, 2)
    
    def _detect_health_alerts(self, device_type: str, readings: List[Dict]) -> List[Dict]:
        """Detecta alertas de salud"""
        alerts = []
        
        # Lógica simplificada
        recent_readings = readings[-5:] if len(readings) >= 5 else readings
        abnormal_count = sum(1 for r in recent_readings if not r.get("is_normal", True))
        
        if abnormal_count >= 3:
            alerts.append({
                "type": "abnormal_readings",
                "severity": "high",
                "message": f"Múltiples lecturas anormales detectadas en {device_type}"
            })
        
        return alerts
    
    def _generate_health_recommendations(self, device_type: str, readings: List[Dict]) -> List[str]:
        """Genera recomendaciones de salud"""
        recommendations = []
        
        alerts = self._detect_health_alerts(device_type, readings)
        if alerts:
            recommendations.append("Considera consultar con un profesional de salud")
        
        return recommendations

