"""
Sistema de integración con wearables
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class WearableData:
    """Datos de wearable"""
    id: str
    user_id: str
    device_type: str  # "fitness_tracker", "smartwatch", "skin_sensor"
    device_id: str
    metric_type: str  # "heart_rate", "sleep", "uv_exposure", "hydration"
    value: float
    unit: str
    timestamp: str
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "device_type": self.device_type,
            "device_id": self.device_id,
            "metric_type": self.metric_type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "created_at": self.created_at
        }


@dataclass
class WearableInsight:
    """Insight de wearable"""
    user_id: str
    metric_type: str
    average_value: float
    trend: str  # "increasing", "decreasing", "stable"
    recommendation: str
    correlation_with_skin: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "metric_type": self.metric_type,
            "average_value": self.average_value,
            "trend": self.trend,
            "recommendation": self.recommendation,
            "correlation_with_skin": self.correlation_with_skin
        }


class WearableIntegration:
    """Sistema de integración con wearables"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.data: Dict[str, List[WearableData]] = {}  # user_id -> [data]
        self.devices: Dict[str, Dict] = {}  # device_id -> device_info
    
    def register_device(self, user_id: str, device_id: str, device_type: str,
                      device_name: str) -> Dict:
        """Registra un dispositivo wearable"""
        device_info = {
            "device_id": device_id,
            "user_id": user_id,
            "device_type": device_type,
            "device_name": device_name,
            "registered_at": datetime.now().isoformat()
        }
        
        self.devices[device_id] = device_info
        return device_info
    
    def sync_data(self, user_id: str, device_id: str, device_type: str,
                 metric_type: str, value: float, unit: str,
                 timestamp: str) -> WearableData:
        """Sincroniza datos de wearable"""
        data = WearableData(
            id=str(uuid.uuid4()),
            user_id=user_id,
            device_type=device_type,
            device_id=device_id,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=timestamp
        )
        
        if user_id not in self.data:
            self.data[user_id] = []
        
        self.data[user_id].append(data)
        return data
    
    def get_user_data(self, user_id: str, metric_type: Optional[str] = None,
                     days: int = 30) -> List[WearableData]:
        """Obtiene datos del usuario"""
        user_data = self.data.get(user_id, [])
        
        # Filtrar por tipo de métrica
        if metric_type:
            user_data = [d for d in user_data if d.metric_type == metric_type]
        
        # Filtrar por fecha
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        user_data = [
            d for d in user_data
            if datetime.fromisoformat(d.timestamp).timestamp() >= cutoff
        ]
        
        # Ordenar por timestamp
        user_data.sort(key=lambda x: x.timestamp, reverse=True)
        
        return user_data
    
    def generate_insights(self, user_id: str, metric_type: str) -> Optional[WearableInsight]:
        """Genera insights de datos de wearable"""
        user_data = self.get_user_data(user_id, metric_type, days=30)
        
        if len(user_data) < 3:
            return None
        
        # Calcular promedio
        values = [d.value for d in user_data]
        average = sum(values) / len(values)
        
        # Determinar tendencia
        recent_values = values[:7] if len(values) >= 7 else values
        older_values = values[-7:] if len(values) >= 14 else []
        
        if older_values:
            recent_avg = sum(recent_values) / len(recent_values)
            older_avg = sum(older_values) / len(older_values)
            
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Generar recomendación
        recommendation = self._generate_recommendation(metric_type, average, trend)
        
        return WearableInsight(
            user_id=user_id,
            metric_type=metric_type,
            average_value=average,
            trend=trend,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, metric_type: str, value: float,
                                trend: str) -> str:
        """Genera recomendación basada en métrica"""
        if metric_type == "uv_exposure":
            if value > 7:
                return "Exposición UV alta. Usa protección solar y evita el sol en horas pico."
            elif value < 2:
                return "Exposición UV baja. Considera suplementos de vitamina D."
            else:
                return "Exposición UV moderada. Mantén protección solar."
        
        elif metric_type == "sleep":
            if value < 6:
                return "Sueño insuficiente. El descanso afecta la salud de la piel."
            elif value > 9:
                return "Sueño adecuado. Continúa con tu rutina."
            else:
                return "Sueño moderado. Intenta dormir 7-8 horas."
        
        elif metric_type == "heart_rate":
            if value > 100:
                return "Frecuencia cardíaca elevada. El estrés puede afectar la piel."
            else:
                return "Frecuencia cardíaca normal."
        
        return "Monitorea esta métrica regularmente."






