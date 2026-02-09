"""
Servicio de Alertas Inteligentes Avanzadas - Sistema completo de alertas
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class AlertPriority(str, Enum):
    """Prioridades de alerta"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    """Tipos de alerta"""
    RELAPSE_RISK = "relapse_risk"
    MEDICATION_MISSED = "medication_missed"
    APPOINTMENT_REMINDER = "appointment_reminder"
    MILESTONE = "milestone"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    HEALTH_ALERT = "health_alert"
    SUPPORT_NEEDED = "support_needed"


class AdvancedIntelligentAlertsService:
    """Servicio de alertas inteligentes avanzadas"""
    
    def __init__(self):
        """Inicializa el servicio de alertas"""
        pass
    
    def create_intelligent_alert(
        self,
        user_id: str,
        alert_type: str,
        context: Dict,
        priority: Optional[str] = None
    ) -> Dict:
        """
        Crea alerta inteligente
        
        Args:
            user_id: ID del usuario
            alert_type: Tipo de alerta
            context: Contexto de la alerta
            priority: Prioridad (opcional, se calcula si no se proporciona)
        
        Returns:
            Alerta creada
        """
        calculated_priority = priority or self._calculate_priority(alert_type, context)
        
        alert = {
            "id": f"alert_{datetime.now().timestamp()}",
            "user_id": user_id,
            "alert_type": alert_type,
            "priority": calculated_priority,
            "context": context,
            "message": self._generate_alert_message(alert_type, context),
            "action_required": calculated_priority in [AlertPriority.CRITICAL, AlertPriority.HIGH],
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        return alert
    
    def evaluate_alert_conditions(
        self,
        user_id: str,
        current_state: Dict,
        historical_data: List[Dict]
    ) -> List[Dict]:
        """
        Evalúa condiciones de alerta
        
        Args:
            user_id: ID del usuario
            current_state: Estado actual
            historical_data: Datos históricos
        
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        
        # Evaluar riesgo de recaída
        relapse_risk = current_state.get("relapse_risk", 0)
        if relapse_risk >= 0.7:
            alerts.append(self.create_intelligent_alert(
                user_id,
                AlertType.RELAPSE_RISK,
                {"risk_score": relapse_risk},
                AlertPriority.CRITICAL
            ))
        
        # Evaluar medicamentos perdidos
        missed_medications = current_state.get("missed_medications", 0)
        if missed_medications >= 2:
            alerts.append(self.create_intelligent_alert(
                user_id,
                AlertType.MEDICATION_MISSED,
                {"missed_count": missed_medications},
                AlertPriority.HIGH
            ))
        
        # Evaluar necesidad de apoyo
        support_level = current_state.get("support_level", 5)
        if support_level <= 3:
            alerts.append(self.create_intelligent_alert(
                user_id,
                AlertType.SUPPORT_NEEDED,
                {"support_level": support_level},
                AlertPriority.MEDIUM
            ))
        
        return alerts
    
    def get_user_alerts(
        self,
        user_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene alertas del usuario
        
        Args:
            user_id: ID del usuario
            status: Filtrar por estado (opcional)
            priority: Filtrar por prioridad (opcional)
        
        Returns:
            Lista de alertas
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
        acknowledgment: Dict
    ) -> Dict:
        """
        Reconoce alerta
        
        Args:
            alert_id: ID de la alerta
            user_id: ID del usuario
            acknowledgment: Información de reconocimiento
        
        Returns:
            Alerta reconocida
        """
        return {
            "alert_id": alert_id,
            "user_id": user_id,
            "acknowledged_at": datetime.now().isoformat(),
            "acknowledgment": acknowledgment,
            "status": "acknowledged"
        }
    
    def _calculate_priority(self, alert_type: str, context: Dict) -> str:
        """Calcula prioridad de alerta"""
        priority_map = {
            AlertType.RELAPSE_RISK: AlertPriority.CRITICAL,
            AlertType.MEDICATION_MISSED: AlertPriority.HIGH,
            AlertType.BEHAVIORAL_ANOMALY: AlertPriority.MEDIUM,
            AlertType.APPOINTMENT_REMINDER: AlertPriority.LOW,
            AlertType.MILESTONE: AlertPriority.INFO
        }
        
        base_priority = priority_map.get(alert_type, AlertPriority.MEDIUM)
        
        # Ajustar basado en contexto
        if context.get("severity") == "high":
            if base_priority == AlertPriority.MEDIUM:
                return AlertPriority.HIGH
            elif base_priority == AlertPriority.LOW:
                return AlertPriority.MEDIUM
        
        return base_priority
    
    def _generate_alert_message(self, alert_type: str, context: Dict) -> str:
        """Genera mensaje de alerta"""
        messages = {
            AlertType.RELAPSE_RISK: "⚠️ Alto riesgo de recaída detectado. Contacta tu sistema de apoyo.",
            AlertType.MEDICATION_MISSED: "Recuerda tomar tus medicamentos según lo prescrito.",
            AlertType.SUPPORT_NEEDED: "Tu nivel de apoyo está bajo. Considera contactar a alguien de confianza.",
            AlertType.MILESTONE: "¡Felicitaciones! Has alcanzado un hito importante.",
            AlertType.APPOINTMENT_REMINDER: "Tienes una cita programada pronto."
        }
        
        return messages.get(alert_type, "Nueva alerta disponible")

