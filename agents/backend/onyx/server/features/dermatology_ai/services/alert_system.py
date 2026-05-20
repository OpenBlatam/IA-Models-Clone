"""
Sistema de alertas y notificaciones
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class AlertLevel(str, Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alerta del sistema"""
    id: str
    user_id: Optional[str]
    level: AlertLevel
    title: str
    message: str
    timestamp: str
    acknowledged: bool = False
    metadata: Optional[Dict] = None


class AlertSystem:
    """Sistema de alertas y notificaciones"""
    
    def __init__(self):
        """Inicializa el sistema de alertas"""
        self.alerts: Dict[str, List[Alert]] = {}  # user_id -> [alerts]
    
    def check_analysis_alerts(self, analysis_result: Dict, 
                            user_id: Optional[str] = None) -> List[Alert]:
        """
        Verifica y genera alertas basadas en un análisis
        
        Args:
            analysis_result: Resultado del análisis
            user_id: ID del usuario
            
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        quality_scores = analysis_result.get("quality_scores", {})
        conditions = analysis_result.get("conditions", [])
        
        # Alerta por score general bajo
        overall_score = quality_scores.get("overall_score", 0)
        if overall_score < 40:
            alerts.append(self._create_alert(
                user_id=user_id,
                level=AlertLevel.CRITICAL,
                title="Score General Muy Bajo",
                message=f"Tu score general es {overall_score:.1f}/100. "
                       "Te recomendamos consultar con un dermatólogo.",
                metadata={"score": overall_score, "type": "low_score"}
            ))
        elif overall_score < 60:
            alerts.append(self._create_alert(
                user_id=user_id,
                level=AlertLevel.WARNING,
                title="Score General Bajo",
                message=f"Tu score general es {overall_score:.1f}/100. "
                       "Considera mejorar tu rutina de skincare.",
                metadata={"score": overall_score, "type": "low_score"}
            ))
        
        # Alerta por condiciones severas
        for condition in conditions:
            if condition.get("severity") == "severe":
                alerts.append(self._create_alert(
                    user_id=user_id,
                    level=AlertLevel.CRITICAL,
                    title=f"Condición Severa Detectada: {condition.get('name', 'Unknown').title()}",
                    message=f"Se detectó {condition.get('description', 'una condición')} "
                           f"con severidad alta. Consulta con un dermatólogo.",
                    metadata={"condition": condition, "type": "severe_condition"}
                ))
            elif condition.get("severity") == "moderate":
                alerts.append(self._create_alert(
                    user_id=user_id,
                    level=AlertLevel.WARNING,
                    title=f"Condición Moderada: {condition.get('name', 'Unknown').title()}",
                    message=f"Se detectó {condition.get('description', 'una condición')} "
                           f"con severidad moderada. Considera tratamiento específico.",
                    metadata={"condition": condition, "type": "moderate_condition"}
                ))
        
        # Alerta por hidratación muy baja
        hydration_score = quality_scores.get("hydration_score", 0)
        if hydration_score < 30:
            alerts.append(self._create_alert(
                user_id=user_id,
                level=AlertLevel.WARNING,
                title="Hidratación Muy Baja",
                message="Tu nivel de hidratación es muy bajo. "
                       "Aumenta el uso de productos hidratantes y bebe más agua.",
                metadata={"score": hydration_score, "type": "low_hydration"}
            ))
        
        # Alerta por múltiples condiciones
        if len(conditions) >= 3:
            alerts.append(self._create_alert(
                user_id=user_id,
                level=AlertLevel.WARNING,
                title="Múltiples Condiciones Detectadas",
                message=f"Se detectaron {len(conditions)} condiciones. "
                       "Considera una consulta dermatológica.",
                metadata={"condition_count": len(conditions), "type": "multiple_conditions"}
            ))
        
        # Guardar alertas
        for alert in alerts:
            self.add_alert(alert)
        
        return alerts
    
    def check_trend_alerts(self, user_id: str, trend_data: Dict) -> List[Alert]:
        """
        Verifica alertas basadas en tendencias
        
        Args:
            user_id: ID del usuario
            trend_data: Datos de tendencia
            
        Returns:
            Lista de alertas
        """
        alerts = []
        
        # Alerta por tendencia negativa
        if trend_data.get("direction") == "declining":
            percentage = trend_data.get("percentage_change", 0)
            if abs(percentage) > 10:
                alerts.append(self._create_alert(
                    user_id=user_id,
                    level=AlertLevel.WARNING,
                    title="Tendencia Negativa Detectada",
                    message=f"Tu score ha disminuido {abs(percentage):.1f}% recientemente. "
                           "Revisa tu rutina de skincare.",
                    metadata={"trend": trend_data, "type": "declining_trend"}
                ))
        
        return alerts
    
    def add_alert(self, alert: Alert):
        """Agrega una alerta al sistema"""
        user_id = alert.user_id or "system"
        if user_id not in self.alerts:
            self.alerts[user_id] = []
        self.alerts[user_id].append(alert)
    
    def get_user_alerts(self, user_id: str, unread_only: bool = False) -> List[Alert]:
        """
        Obtiene alertas de un usuario
        
        Args:
            user_id: ID del usuario
            unread_only: Solo alertas no leídas
            
        Returns:
            Lista de alertas
        """
        alerts = self.alerts.get(user_id, [])
        if unread_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, user_id: str, alert_id: str):
        """Marca una alerta como leída"""
        alerts = self.alerts.get(user_id, [])
        for alert in alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break
    
    def _create_alert(self, user_id: Optional[str], level: AlertLevel,
                     title: str, message: str,
                     metadata: Optional[Dict] = None) -> Alert:
        """Crea una nueva alerta"""
        import hashlib
        alert_id = hashlib.md5(
            f"{user_id}{title}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        return Alert(
            id=alert_id,
            user_id=user_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now().isoformat(),
            acknowledged=False,
            metadata=metadata or {}
        )
    
    def get_alert_summary(self, user_id: str) -> Dict:
        """
        Obtiene resumen de alertas de un usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Diccionario con resumen
        """
        alerts = self.get_user_alerts(user_id)
        unread = [a for a in alerts if not a.acknowledged]
        
        by_level = {
            "info": len([a for a in unread if a.level == AlertLevel.INFO]),
            "warning": len([a for a in unread if a.level == AlertLevel.WARNING]),
            "critical": len([a for a in unread if a.level == AlertLevel.CRITICAL])
        }
        
        return {
            "total_alerts": len(alerts),
            "unread_count": len(unread),
            "by_level": by_level,
            "has_critical": by_level["critical"] > 0
        }






