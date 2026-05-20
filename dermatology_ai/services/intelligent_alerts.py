"""
Sistema de alertas inteligentes basadas en patrones
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid


@dataclass
class Alert:
    """Alerta"""
    id: str
    user_id: str
    alert_type: str  # "decline", "improvement", "anomaly", "milestone", "reminder"
    severity: str  # "low", "medium", "high", "critical"
    title: str
    message: str
    metric: Optional[str] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    action_required: bool = False
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "metric": self.metric,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "action_required": self.action_required,
            "created_at": self.created_at
        }


class IntelligentAlerts:
    """Sistema de alertas inteligentes"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.alerts: Dict[str, List[Alert]] = {}  # user_id -> [alerts]
        self.historical_data: Dict[str, List[Dict]] = {}  # user_id -> [data_points]
    
    def add_data_point(self, user_id: str, timestamp: str, metrics: Dict):
        """Agrega punto de datos"""
        data_point = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        if user_id not in self.historical_data:
            self.historical_data[user_id] = []
        
        self.historical_data[user_id].append(data_point)
        self.historical_data[user_id].sort(key=lambda x: x["timestamp"])
    
    def check_alerts(self, user_id: str) -> List[Alert]:
        """Verifica y genera alertas"""
        data_points = self.historical_data.get(user_id, [])
        
        if len(data_points) < 2:
            return []
        
        alerts = []
        recent_data = data_points[-10:]  # Últimos 10 puntos
        
        # Verificar cada métrica
        for metric_name in ["overall_score", "hydration_score", "texture_score"]:
            values = [d["metrics"].get(metric_name, 0) for d in recent_data]
            
            if not values or len(values) < 2:
                continue
            
            current_value = values[-1]
            previous_value = values[-2] if len(values) > 1 else current_value
            
            # Alerta por declive significativo
            if current_value < previous_value - 10:
                alerts.append(Alert(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    alert_type="decline",
                    severity="high" if current_value < previous_value - 15 else "medium",
                    title=f"Declive en {metric_name}",
                    message=f"{metric_name} ha disminuido de {previous_value:.1f} a {current_value:.1f}",
                    metric=metric_name,
                    threshold=previous_value - 10,
                    current_value=current_value,
                    action_required=True
                ))
            
            # Alerta por mejora significativa
            if current_value > previous_value + 10:
                alerts.append(Alert(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    alert_type="improvement",
                    severity="low",
                    title=f"Mejora en {metric_name}",
                    message=f"¡Excelente! {metric_name} ha mejorado de {previous_value:.1f} a {current_value:.1f}",
                    metric=metric_name,
                    threshold=previous_value + 10,
                    current_value=current_value,
                    action_required=False
                ))
            
            # Alerta por valor crítico
            if current_value < 40:
                alerts.append(Alert(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    alert_type="critical",
                    severity="critical",
                    title=f"Valor crítico en {metric_name}",
                    message=f"{metric_name} está en {current_value:.1f}, considera consultar con un dermatólogo",
                    metric=metric_name,
                    threshold=40,
                    current_value=current_value,
                    action_required=True
                ))
            
            # Alerta por milestone (valores altos)
            if current_value > 80 and previous_value <= 80:
                alerts.append(Alert(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    alert_type="milestone",
                    severity="low",
                    title=f"¡Milestone alcanzado!",
                    message=f"{metric_name} ha alcanzado {current_value:.1f} - ¡Excelente progreso!",
                    metric=metric_name,
                    threshold=80,
                    current_value=current_value,
                    action_required=False
                ))
        
        # Alerta por anomalía (valores fuera de rango esperado)
        for metric_name in ["overall_score", "hydration_score", "texture_score"]:
            values = [d["metrics"].get(metric_name, 0) for d in recent_data]
            if len(values) >= 3:
                avg = sum(values) / len(values)
                std = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
                current = values[-1]
                
                if abs(current - avg) > 2 * std and current < avg:
                    alerts.append(Alert(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        alert_type="anomaly",
                        severity="medium",
                        title=f"Anomalía detectada en {metric_name}",
                        message=f"{metric_name} está significativamente por debajo del promedio reciente",
                        metric=metric_name,
                        threshold=avg - 2 * std,
                        current_value=current,
                        action_required=True
                    ))
        
        # Guardar alertas
        if user_id not in self.alerts:
            self.alerts[user_id] = []
        
        self.alerts[user_id].extend(alerts)
        
        # Mantener solo últimas 100 alertas por usuario
        self.alerts[user_id] = self.alerts[user_id][-100:]
        
        return alerts
    
    def get_user_alerts(self, user_id: str, unread_only: bool = False) -> List[Alert]:
        """Obtiene alertas del usuario"""
        user_alerts = self.alerts.get(user_id, [])
        
        if unread_only:
            # En una implementación real, tendrías un campo "read"
            pass
        
        user_alerts.sort(key=lambda x: x.created_at, reverse=True)
        return user_alerts
    
    def get_critical_alerts(self, user_id: str) -> List[Alert]:
        """Obtiene alertas críticas"""
        user_alerts = self.alerts.get(user_id, [])
        critical = [a for a in user_alerts if a.severity == "critical"]
        critical.sort(key=lambda x: x.created_at, reverse=True)
        return critical
