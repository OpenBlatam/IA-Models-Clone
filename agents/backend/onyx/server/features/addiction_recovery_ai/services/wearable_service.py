"""
Servicio de Integración con Wearables - Conecta con dispositivos portátiles
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class WearableType(str, Enum):
    """Tipos de wearables"""
    FITNESS_TRACKER = "fitness_tracker"
    SMARTWATCH = "smartwatch"
    HEART_RATE_MONITOR = "heart_rate_monitor"
    SLEEP_TRACKER = "sleep_tracker"
    BLOOD_PRESSURE = "blood_pressure"


class WearableService:
    """Servicio de integración con wearables y sensores"""
    
    def __init__(self):
        """Inicializa el servicio de wearables"""
        pass
    
    def register_device(
        self,
        user_id: str,
        device_type: str,
        device_name: str,
        device_id: str,
        manufacturer: Optional[str] = None
    ) -> Dict:
        """
        Registra un dispositivo wearable
        
        Args:
            user_id: ID del usuario
            device_type: Tipo de dispositivo
            device_name: Nombre del dispositivo
            device_id: ID único del dispositivo
            manufacturer: Fabricante (opcional)
        
        Returns:
            Dispositivo registrado
        """
        device = {
            "user_id": user_id,
            "device_type": device_type,
            "device_name": device_name,
            "device_id": device_id,
            "manufacturer": manufacturer,
            "registered_at": datetime.now().isoformat(),
            "active": True,
            "last_sync": datetime.now().isoformat()
        }
        
        return device
    
    def sync_device_data(
        self,
        user_id: str,
        device_id: str,
        data: Dict
    ) -> Dict:
        """
        Sincroniza datos del dispositivo
        
        Args:
            user_id: ID del usuario
            device_id: ID del dispositivo
            data: Datos del dispositivo
        
        Returns:
            Resultado de sincronización
        """
        return {
            "user_id": user_id,
            "device_id": device_id,
            "data_synced": True,
            "records_count": len(data.get("records", [])),
            "sync_time": datetime.now().isoformat(),
            "data": data
        }
    
    def get_health_metrics_from_wearable(
        self,
        user_id: str,
        device_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Obtiene métricas de salud del wearable
        
        Args:
            user_id: ID del usuario
            device_id: ID del dispositivo
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
        
        Returns:
            Métricas de salud
        """
        # En implementación real, esto se conectaría con la API del wearable
        return {
            "user_id": user_id,
            "device_id": device_id,
            "metrics": {
                "heart_rate": {
                    "average": 72,
                    "resting": 65,
                    "max": 180
                },
                "steps": {
                    "total": 8500,
                    "average_daily": 8500
                },
                "sleep": {
                    "hours": 7.5,
                    "quality": "good"
                },
                "calories": {
                    "burned": 2200,
                    "active": 500
                }
            },
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            }
        }
    
    def correlate_wearable_with_recovery(
        self,
        user_id: str,
        wearable_data: Dict,
        recovery_data: Dict
    ) -> Dict:
        """
        Correlaciona datos del wearable con datos de recuperación
        
        Args:
            user_id: ID del usuario
            wearable_data: Datos del wearable
            recovery_data: Datos de recuperación
        
        Returns:
            Análisis de correlación
        """
        correlations = []
        
        # Correlación entre ejercicio y cravings
        if wearable_data.get("steps") and recovery_data.get("cravings_level"):
            if wearable_data["steps"] > 10000:
                correlations.append({
                    "metric": "exercise_vs_cravings",
                    "correlation": "negative",
                    "insight": "Mayor actividad física se correlaciona con menores cravings"
                })
        
        # Correlación entre sueño y estado de ánimo
        if wearable_data.get("sleep") and recovery_data.get("mood"):
            sleep_hours = wearable_data["sleep"].get("hours", 0)
            if sleep_hours >= 7:
                correlations.append({
                    "metric": "sleep_vs_mood",
                    "correlation": "positive",
                    "insight": "Mejor sueño se correlaciona con mejor estado de ánimo"
                })
        
        return {
            "user_id": user_id,
            "correlations": correlations,
            "recommendations": self._generate_wearable_recommendations(correlations),
            "generated_at": datetime.now().isoformat()
        }
    
    def get_user_devices(self, user_id: str) -> List[Dict]:
        """
        Obtiene dispositivos del usuario
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de dispositivos
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def _generate_wearable_recommendations(self, correlations: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en correlaciones"""
        recommendations = []
        
        for corr in correlations:
            if corr.get("metric") == "exercise_vs_cravings" and corr.get("correlation") == "negative":
                recommendations.append("Aumenta tu actividad física para reducir cravings")
            
            if corr.get("metric") == "sleep_vs_mood" and corr.get("correlation") == "positive":
                recommendations.append("Prioriza 7-8 horas de sueño para mejorar tu estado de ánimo")
        
        return recommendations

