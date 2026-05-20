"""
Notifications endpoints
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class NotificationsRouter(BaseRouter):
    """Router for notifications endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/notifications", tags=["Notifications"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all notifications routes"""
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def get_notifications(
            user_id: str = Query(...),
            unread_only: bool = Query(False),
            limit: int = Query(50, ge=1, le=100)
        ):
            """Obtiene las notificaciones de un usuario"""
            notification_service = self.get_service("notification_service")
            notifications = notification_service.get_notifications(user_id, unread_only, limit)
            return self.list_response(notifications, key="notifications")
        
        @self.router.put("/{notification_id}/read", response_model=dict)
        @self.handle_exceptions
        async def mark_as_read(
            notification_id: str,
            user_id: str = Query(...)
        ):
            """Marca una notificación como leída"""
            notification_service = self.get_service("notification_service")
            success = notification_service.mark_as_read(notification_id, user_id)
            self.require_success(success, "Notificación no encontrada", status_code=404)
            return self.success_response(None, message="Notificación marcada como leída")
        
        @self.router.put("/read-all", response_model=dict)
        @self.handle_exceptions
        async def mark_all_as_read(user_id: str = Query(...)):
            """Marca todas las notificaciones como leídas"""
            notification_service = self.get_service("notification_service")
            notification_service.mark_all_as_read(user_id)
            return self.success_response(None, message="Todas las notificaciones marcadas como leídas")
        
        @self.router.delete("/{notification_id}", response_model=dict)
        @self.handle_exceptions
        async def delete_notification(
            notification_id: str,
            user_id: str = Query(...)
        ):
            """Elimina una notificación"""
            notification_service = self.get_service("notification_service")
            success = notification_service.delete_notification(notification_id, user_id)
            self.require_success(success, "Notificación no encontrada", status_code=404)
            return self.success_response(None, message="Notificación eliminada")
        
        @self.router.get("/stats", response_model=dict)
        @self.handle_exceptions
        async def get_notification_stats(user_id: str = Query(...)):
            """Obtiene estadísticas de notificaciones"""
            notification_service = self.get_service("notification_service")
            stats = notification_service.get_stats(user_id)
            return self.success_response({"stats": stats})


def get_notifications_router() -> NotificationsRouter:
    """Factory function to get notifications router"""
    return NotificationsRouter()

