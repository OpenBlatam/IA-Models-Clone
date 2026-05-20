"""
Servicio de Notificaciones Push - Notificaciones en tiempo real
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class PushPlatform(str, Enum):
    """Plataformas de push"""
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"


class NotificationPriority(str, Enum):
    """Prioridades de notificación"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class PushNotificationService:
    """Servicio de notificaciones push"""
    
    def __init__(self):
        """Inicializa el servicio de push"""
        pass
    
    def register_device(
        self,
        user_id: str,
        device_token: str,
        platform: str,
        device_info: Optional[Dict] = None
    ) -> Dict:
        """
        Registra un dispositivo para recibir notificaciones push
        
        Args:
            user_id: ID del usuario
            device_token: Token del dispositivo
            platform: Plataforma (ios, android, web)
            device_info: Información adicional del dispositivo (opcional)
        
        Returns:
            Dispositivo registrado
        """
        device = {
            "user_id": user_id,
            "device_token": device_token,
            "platform": platform,
            "device_info": device_info or {},
            "registered_at": datetime.now().isoformat(),
            "active": True,
            "last_seen": datetime.now().isoformat()
        }
        
        return device
    
    def send_notification(
        self,
        user_id: str,
        title: str,
        body: str,
        priority: str = NotificationPriority.NORMAL,
        data: Optional[Dict] = None,
        scheduled_time: Optional[datetime] = None
    ) -> Dict:
        """
        Envía una notificación push
        
        Args:
            user_id: ID del usuario
            title: Título de la notificación
            body: Cuerpo de la notificación
            priority: Prioridad
            data: Datos adicionales (opcional)
            scheduled_time: Hora programada (opcional, inmediato si no se especifica)
        
        Returns:
            Notificación enviada
        """
        notification = {
            "user_id": user_id,
            "title": title,
            "body": body,
            "priority": priority,
            "data": data or {},
            "scheduled_time": scheduled_time.isoformat() if scheduled_time else None,
            "sent_at": datetime.now().isoformat() if not scheduled_time else None,
            "status": "sent" if not scheduled_time else "scheduled"
        }
        
        return notification
    
    def send_reminder_notification(
        self,
        user_id: str,
        reminder_type: str,
        message: str
    ) -> Dict:
        """
        Envía notificación de recordatorio
        
        Args:
            user_id: ID del usuario
            reminder_type: Tipo de recordatorio
            message: Mensaje
        
        Returns:
            Notificación enviada
        """
        titles = {
            "check_in": "Check-in Diario",
            "medication": "Recordatorio de Medicamento",
            "exercise": "Hora de Ejercicio",
            "support": "Recordatorio de Apoyo"
        }
        
        title = titles.get(reminder_type, "Recordatorio")
        
        return self.send_notification(
            user_id=user_id,
            title=title,
            body=message,
            priority=NotificationPriority.NORMAL,
            data={"type": "reminder", "reminder_type": reminder_type}
        )
    
    def send_milestone_notification(
        self,
        user_id: str,
        milestone_days: int,
        message: str
    ) -> Dict:
        """
        Envía notificación de hito alcanzado
        
        Args:
            user_id: ID del usuario
            milestone_days: Días del hito
            message: Mensaje de celebración
        
        Returns:
            Notificación enviada
        """
        return self.send_notification(
            user_id=user_id,
            title=f"¡{milestone_days} días de sobriedad!",
            body=message,
            priority=NotificationPriority.HIGH,
            data={"type": "milestone", "days": milestone_days}
        )
    
    def send_risk_alert(
        self,
        user_id: str,
        risk_level: str,
        message: str,
        recommendations: List[str]
    ) -> Dict:
        """
        Envía alerta de riesgo
        
        Args:
            user_id: ID del usuario
            risk_level: Nivel de riesgo
            message: Mensaje de alerta
            recommendations: Recomendaciones
        
        Returns:
            Notificación enviada
        """
        priority = NotificationPriority.URGENT if risk_level in ["alto", "crítico"] else NotificationPriority.HIGH
        
        return self.send_notification(
            user_id=user_id,
            title=f"⚠️ Alerta: Riesgo {risk_level}",
            body=message,
            priority=priority,
            data={
                "type": "risk_alert",
                "risk_level": risk_level,
                "recommendations": recommendations
            }
        )
    
    def schedule_recurring_notification(
        self,
        user_id: str,
        title: str,
        body: str,
        schedule: Dict,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Programa notificación recurrente
        
        Args:
            user_id: ID del usuario
            title: Título
            body: Cuerpo
            schedule: Horario (daily, weekly, etc.)
            end_date: Fecha de fin (opcional)
        
        Returns:
            Notificación programada
        """
        return {
            "user_id": user_id,
            "title": title,
            "body": body,
            "schedule": schedule,
            "end_date": end_date.isoformat() if end_date else None,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
    
    def get_notification_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Obtiene historial de notificaciones
        
        Args:
            user_id: ID del usuario
            limit: Límite de resultados
        
        Returns:
            Historial de notificaciones
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def mark_notification_read(
        self,
        notification_id: str
    ) -> Dict:
        """
        Marca una notificación como leída
        
        Args:
            notification_id: ID de la notificación
        
        Returns:
            Resultado
        """
        return {
            "notification_id": notification_id,
            "read_at": datetime.now().isoformat(),
            "status": "read"
        }

