"""
Sistema de notificaciones
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from sqlalchemy import Column, String, Text, DateTime, JSON, Boolean, Integer
from sqlalchemy.orm import Session

from ..db.base import Base, get_db_session

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Tipos de notificaciones"""
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    IDENTITY_CREATED = "identity_created"
    CONTENT_GENERATED = "content_generated"
    VERSION_CREATED = "version_created"
    SYSTEM_ALERT = "system_alert"


class NotificationModel(Base):
    """Modelo de notificación en BD"""
    __tablename__ = "notifications"
    
    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)  # Opcional para multi-user
    notification_type = Column(String(50), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSON, nullable=True)
    read = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    read_at = Column(DateTime, nullable=True)


@dataclass
class Notification:
    """Notificación"""
    notification_id: str
    notification_type: NotificationType
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    read: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    read_at: Optional[datetime] = None


class NotificationService:
    """Servicio de notificaciones"""
    
    def __init__(self):
        self._init_table()
    
    def _init_table(self):
        """Inicializa tabla de notificaciones"""
        from ..db.base import init_db
        init_db()
    
    def create_notification(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Crea una notificación
        
        Args:
            notification_type: Tipo de notificación
            title: Título
            message: Mensaje
            data: Datos adicionales
            user_id: ID de usuario (opcional)
            
        Returns:
            ID de la notificación
        """
        notification_id = str(uuid.uuid4())
        
        with get_db_session() as db:
            notification = NotificationModel(
                id=notification_id,
                user_id=user_id,
                notification_type=notification_type.value,
                title=title,
                message=message,
                data=data,
                read=False,
                created_at=datetime.utcnow()
            )
            db.add(notification)
            db.commit()
        
        logger.info(f"Notificación creada: {notification_id} ({notification_type.value})")
        return notification_id
    
    def get_notifications(
        self,
        user_id: Optional[str] = None,
        unread_only: bool = False,
        notification_type: Optional[NotificationType] = None,
        limit: int = 50
    ) -> List[Notification]:
        """Obtiene notificaciones"""
        with get_db_session() as db:
            query = db.query(NotificationModel)
            
            if user_id:
                query = query.filter(NotificationModel.user_id == user_id)
            
            if unread_only:
                query = query.filter(NotificationModel.read == False)
            
            if notification_type:
                query = query.filter(NotificationModel.notification_type == notification_type.value)
            
            results = query.order_by(
                NotificationModel.created_at.desc()
            ).limit(limit).all()
            
            return [
                Notification(
                    notification_id=n.id,
                    notification_type=NotificationType(n.notification_type),
                    title=n.title,
                    message=n.message,
                    data=n.data,
                    read=n.read,
                    created_at=n.created_at,
                    read_at=n.read_at
                )
                for n in results
            ]
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Marca notificación como leída"""
        with get_db_session() as db:
            notification = db.query(NotificationModel).filter_by(id=notification_id).first()
            if not notification:
                return False
            
            notification.read = True
            notification.read_at = datetime.utcnow()
            db.commit()
            
            return True
    
    def mark_all_as_read(self, user_id: Optional[str] = None) -> int:
        """Marca todas las notificaciones como leídas"""
        with get_db_session() as db:
            query = db.query(NotificationModel).filter_by(read=False)
            
            if user_id:
                query = query.filter_by(user_id=user_id)
            
            count = query.update({
                "read": True,
                "read_at": datetime.utcnow()
            })
            db.commit()
            
            return count
    
    def get_unread_count(self, user_id: Optional[str] = None) -> int:
        """Obtiene conteo de no leídas"""
        with get_db_session() as db:
            query = db.query(NotificationModel).filter_by(read=False)
            
            if user_id:
                query = query.filter_by(user_id=user_id)
            
            return query.count()
    
    def delete_notification(self, notification_id: str) -> bool:
        """Elimina notificación"""
        with get_db_session() as db:
            notification = db.query(NotificationModel).filter_by(id=notification_id).first()
            if not notification:
                return False
            
            db.delete(notification)
            db.commit()
            return True


# Singleton global
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Obtiene instancia singleton del servicio de notificaciones"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service




