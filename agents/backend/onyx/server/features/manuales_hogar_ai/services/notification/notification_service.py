"""
Notification Service
===================

Servicio para gestionar notificaciones de usuarios.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, update

from ...core.base.service_base import BaseService
from ...database.models import Notification


class NotificationService(BaseService):
    """Servicio para gestionar notificaciones."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def create_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        message: str,
        manual_id: Optional[int] = None
    ) -> Notification:
        """
        Crear notificación.
        
        Args:
            user_id: ID del usuario
            notification_type: Tipo de notificación
            title: Título
            message: Mensaje
            manual_id: ID del manual relacionado (opcional)
        
        Returns:
            Notificación creada
        """
        try:
            notification = Notification(
                user_id=user_id,
                manual_id=manual_id,
                type=notification_type,
                title=title,
                message=message
            )
            
            self.db.add(notification)
            await self.db.commit()
            await self.db.refresh(notification)
            
            self.log_info(f"Notificación creada: Usuario {user_id}, Tipo: {notification_type}")
            return notification
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error creando notificación: {str(e)}")
            raise
    
    async def get_user_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 20,
        offset: int = 0
    ) -> List[Notification]:
        """
        Obtener notificaciones de usuario.
        
        Args:
            user_id: ID del usuario
            unread_only: Solo no leídas
            limit: Límite de resultados
            offset: Offset para paginación
        
        Returns:
            Lista de notificaciones
        """
        try:
            query = select(Notification).where(
                Notification.user_id == user_id
            )
            
            if unread_only:
                query = query.where(Notification.is_read == False)
            
            query = query.order_by(desc(Notification.created_at)).limit(limit).offset(offset)
            
            result = await self.db.execute(query)
            return list(result.scalars().all())
        
        except Exception as e:
            self.log_error(f"Error obteniendo notificaciones: {str(e)}")
            return []
    
    async def mark_as_read(
        self,
        notification_id: int,
        user_id: str
    ) -> bool:
        """
        Marcar notificación como leída.
        
        Args:
            notification_id: ID de la notificación
            user_id: ID del usuario
        
        Returns:
            True si se marcó exitosamente
        """
        try:
            result = await self.db.execute(
                select(Notification).where(
                    and_(
                        Notification.id == notification_id,
                        Notification.user_id == user_id
                    )
                )
            )
            notification = result.scalar_one_or_none()
            
            if not notification:
                return False
            
            notification.is_read = True
            notification.read_at = datetime.now()
            
            await self.db.commit()
            
            self.log_info(f"Notificación marcada como leída: {notification_id}")
            return True
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error marcando notificación: {str(e)}")
            return False
    
    async def mark_all_as_read(
        self,
        user_id: str
    ) -> int:
        """
        Marcar todas las notificaciones como leídas.
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Número de notificaciones marcadas
        """
        try:
            stmt = update(Notification).where(
                and_(
                    Notification.user_id == user_id,
                    Notification.is_read == False
                )
            ).values(
                is_read=True,
                read_at=datetime.now()
            )
            
            result = await self.db.execute(stmt)
            await self.db.commit()
            
            count = result.rowcount
            self.log_info(f"Marcadas {count} notificaciones como leídas: Usuario {user_id}")
            return count
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error marcando notificaciones: {str(e)}")
            return 0
    
    async def get_unread_count(
        self,
        user_id: str
    ) -> int:
        """
        Obtener contador de notificaciones no leídas.
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Número de notificaciones no leídas
        """
        try:
            result = await self.db.execute(
                select(func.count(Notification.id)).where(
                    and_(
                        Notification.user_id == user_id,
                        Notification.is_read == False
                    )
                )
            )
            return result.scalar() or 0
        except Exception as e:
            self.log_error(f"Error obteniendo contador: {str(e)}")
            return 0
    
    async def delete_notification(
        self,
        notification_id: int,
        user_id: str
    ) -> bool:
        """
        Eliminar notificación.
        
        Args:
            notification_id: ID de la notificación
            user_id: ID del usuario
        
        Returns:
            True si se eliminó exitosamente
        """
        try:
            result = await self.db.execute(
                select(Notification).where(
                    and_(
                        Notification.id == notification_id,
                        Notification.user_id == user_id
                    )
                )
            )
            notification = result.scalar_one_or_none()
            
            if not notification:
                return False
            
            await self.db.delete(notification)
            await self.db.commit()
            
            self.log_info(f"Notificación eliminada: {notification_id}")
            return True
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error eliminando notificación: {str(e)}")
            return False

