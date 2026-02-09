"""
Servicio de Alertas Inteligentes - Sistema avanzado de alertas basado en IA
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class AlertSeverity(str, Enum):
    """Niveles de severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Tipos de alertas"""
    RELAPSE_RISK = "relapse_risk"
    MISSED_CHECK_IN = "missed_check_in"
    MEDICATION_MISSED = "medication_missed"
    EMOTIONAL_DISTRESS = "emotional_distress"
    ISOLATION = "isolation"
    SUCCESS_MILESTONE = "success_milestone"
    PATTERN_DETECTED = "pattern_detected"


class IntelligentAlertsService:
    """Servicio de alertas inteligentes"""
    
    def __init__(self):
        """Inicializa el servicio de alertas inteligentes"""
        self.alert_rules = self._load_alert_rules()
    
    def evaluate_alert_conditions(
        self,
        user_id: str,
        user_data: Dict,
        recent_activity: List[Dict]
    ) -> List[Dict]:
        """
        Evalúa condiciones para generar alertas
        
        Args:
            user_id: ID del usuario
            user_data: Datos del usuario
            recent_activity: Actividad reciente
        
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        
        # Evaluar cada regla de alerta
        for rule in self.alert_rules:
            if self._evaluate_rule(rule, user_data, recent_activity):
                alert = self._create_alert(user_id, rule, user_data)
                alerts.append(alert)
        
        return alerts
    
    def create_custom_alert(
        self,
        user_id: str,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        action_required: bool = False
    ) -> Dict:
        """
        Crea una alerta personalizada
        
        Args:
            user_id: ID del usuario
            alert_type: Tipo de alerta
            severity: Severidad
            title: Título
            message: Mensaje
            action_required: Si requiere acción
        
        Returns:
            Alerta creada
        """
        alert = {
            "id": f"alert_{datetime.now().timestamp()}",
            "user_id": user_id,
            "alert_type": alert_type,
            "severity": severity,
            "title": title,
            "message": message,
            "action_required": action_required,
            "created_at": datetime.now().isoformat(),
            "acknowledged": False,
            "acknowledged_at": None
        }
        
        return alert
    
    def get_active_alerts(
        self,
        user_id: str,
        severity: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene alertas activas del usuario
        
        Args:
            user_id: ID del usuario
            severity: Filtrar por severidad (opcional)
        
        Returns:
            Lista de alertas activas
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Marca una alerta como reconocida
        
        Args:
            alert_id: ID de la alerta
            user_id: ID del usuario
            notes: Notas adicionales (opcional)
        
        Returns:
            Alerta actualizada
        """
        return {
            "alert_id": alert_id,
            "user_id": user_id,
            "acknowledged": True,
            "acknowledged_at": datetime.now().isoformat(),
            "notes": notes
        }
    
    def _evaluate_rule(
        self,
        rule: Dict,
        user_data: Dict,
        recent_activity: List[Dict]
    ) -> bool:
        """Evalúa si una regla de alerta se cumple"""
        rule_type = rule.get("type")
        
        if rule_type == AlertType.MISSED_CHECK_IN:
            # Verificar si faltó check-in
            last_check_in = user_data.get("last_check_in")
            if last_check_in:
                last_date = datetime.fromisoformat(last_check_in.replace('Z', '+00:00'))
                days_since = (datetime.now() - last_date.replace(tzinfo=None)).days
                return days_since >= rule.get("threshold_days", 2)
        
        elif rule_type == AlertType.RELAPSE_RISK:
            # Verificar riesgo de recaída
            risk_score = user_data.get("relapse_risk_score", 0)
            return risk_score >= rule.get("threshold_score", 75)
        
        elif rule_type == AlertType.EMOTIONAL_DISTRESS:
            # Verificar angustia emocional
            sentiment = user_data.get("recent_sentiment", "neutral")
            return sentiment == "negative" and user_data.get("negative_days_streak", 0) >= 3
        
        return False
    
    def _create_alert(
        self,
        user_id: str,
        rule: Dict,
        user_data: Dict
    ) -> Dict:
        """Crea una alerta basada en una regla"""
        return {
            "id": f"alert_{datetime.now().timestamp()}",
            "user_id": user_id,
            "alert_type": rule.get("type"),
            "severity": rule.get("severity", AlertSeverity.WARNING),
            "title": rule.get("title", "Alerta"),
            "message": rule.get("message", ""),
            "action_required": rule.get("action_required", False),
            "recommended_actions": rule.get("recommended_actions", []),
            "created_at": datetime.now().isoformat(),
            "acknowledged": False
        }
    
    def _load_alert_rules(self) -> List[Dict]:
        """Carga reglas de alerta"""
        return [
            {
                "type": AlertType.MISSED_CHECK_IN,
                "severity": AlertSeverity.WARNING,
                "threshold_days": 2,
                "title": "Check-in Perdido",
                "message": "No has registrado un check-in en los últimos días",
                "action_required": False,
                "recommended_actions": ["Registrar check-in", "Contactar apoyo"]
            },
            {
                "type": AlertType.RELAPSE_RISK,
                "severity": AlertSeverity.HIGH,
                "threshold_score": 75,
                "title": "Alto Riesgo de Recaída",
                "message": "Se detectó un alto riesgo de recaída",
                "action_required": True,
                "recommended_actions": [
                    "Contactar sistema de apoyo inmediatamente",
                    "Revisar plan de emergencia",
                    "Usar estrategias de afrontamiento"
                ]
            },
            {
                "type": AlertType.EMOTIONAL_DISTRESS,
                "severity": AlertSeverity.WARNING,
                "title": "Angustia Emocional Detectada",
                "message": "Se detectaron múltiples días con sentimientos negativos",
                "action_required": False,
                "recommended_actions": [
                    "Contactar consejero o terapeuta",
                    "Practicar técnicas de relajación",
                    "Revisar estrategias de afrontamiento"
                ]
            },
            {
                "type": AlertType.ISOLATION,
                "severity": AlertSeverity.WARNING,
                "title": "Aislamiento Social Detectado",
                "message": "Se detectó un patrón de aislamiento",
                "action_required": False,
                "recommended_actions": [
                    "Contactar familiares o amigos",
                    "Asistir a grupos de apoyo",
                    "Programar actividades sociales"
                ]
            }
        ]

