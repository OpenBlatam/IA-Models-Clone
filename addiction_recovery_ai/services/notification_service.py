"""
Servicio de notificaciones - Envía recordatorios y alertas
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class NotificationType(str, Enum):
    """Tipos de notificaciones"""
    REMINDER = "reminder"
    MILESTONE = "milestone"
    RISK_ALERT = "risk_alert"
    MOTIVATION = "motivation"
    DAILY_CHECK_IN = "daily_check_in"
    WEEKLY_REVIEW = "weekly_review"


class NotificationService:
    """Servicio de notificaciones y recordatorios"""
    
    def __init__(self):
        """Inicializa el servicio de notificaciones"""
        pass
    
    def create_reminder(
        self,
        user_id: str,
        reminder_type: str,
        message: str,
        scheduled_time: datetime,
        repeat_daily: bool = False
    ) -> Dict:
        """
        Crea un recordatorio
        
        Args:
            user_id: ID del usuario
            reminder_type: Tipo de recordatorio
            message: Mensaje del recordatorio
            scheduled_time: Hora programada
            repeat_daily: Si se repite diariamente
        
        Returns:
            Recordatorio creado
        """
        reminder = {
            "user_id": user_id,
            "reminder_type": reminder_type,
            "message": message,
            "scheduled_time": scheduled_time.isoformat(),
            "repeat_daily": repeat_daily,
            "active": True,
            "created_at": datetime.now().isoformat()
        }
        
        return reminder
    
    def get_daily_reminders(self, user_id: str) -> List[Dict]:
        """Obtiene recordatorios diarios del usuario"""
        reminders = [
            {
                "type": "morning_check_in",
                "time": "09:00",
                "message": "¡Buenos días! ¿Cómo te sientes hoy?",
                "action": "Registrar estado emocional"
            },
            {
                "type": "evening_reflection",
                "time": "20:00",
                "message": "Tómate un momento para reflexionar sobre tu día",
                "action": "Completar entrada diaria"
            }
        ]
        
        return reminders
    
    def create_milestone_notification(
        self,
        user_id: str,
        milestone_days: int,
        message: str
    ) -> Dict:
        """Crea notificación de hito alcanzado"""
        return {
            "user_id": user_id,
            "type": NotificationType.MILESTONE,
            "milestone_days": milestone_days,
            "message": message,
            "created_at": datetime.now().isoformat(),
            "read": False
        }
    
    def create_risk_alert(
        self,
        user_id: str,
        risk_level: str,
        risk_score: float,
        recommendations: List[str]
    ) -> Dict:
        """Crea alerta de riesgo de recaída"""
        return {
            "user_id": user_id,
            "type": NotificationType.RISK_ALERT,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "message": f"⚠️ Alerta: Riesgo {risk_level} de recaída detectado",
            "recommendations": recommendations,
            "created_at": datetime.now().isoformat(),
            "read": False,
            "urgent": risk_level in ["alto", "crítico"]
        }
    
    def get_pending_notifications(self, user_id: str) -> List[Dict]:
        """Obtiene notificaciones pendientes del usuario"""
        # En implementación real, esto vendría de la base de datos
        return []
    
    def mark_notification_read(self, notification_id: str) -> bool:
        """Marca una notificación como leída"""
        # En implementación real, actualizaría la base de datos
        return True

