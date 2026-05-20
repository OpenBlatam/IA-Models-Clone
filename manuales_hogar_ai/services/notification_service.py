"""
Servicio de Notificaciones (Legacy)
====================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar services.notification.notification_service.NotificationService
"""

from .notification.notification_service import NotificationService

__all__ = ["NotificationService"]

