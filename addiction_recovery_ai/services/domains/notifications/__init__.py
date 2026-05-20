"""
Notifications domain services
"""

from services.domains import register_service

try:
    from services.notification_service import NotificationService
    from services.intelligent_notifications_service import IntelligentNotificationsService
    from services.intelligent_reminders_service import IntelligentRemindersService
    from services.push_notification_service import PushNotificationService
    
    def register_services():
        register_service("notifications", "notification", NotificationService)
        register_service("notifications", "intelligent", IntelligentNotificationsService)
        register_service("notifications", "reminders", IntelligentRemindersService)
        register_service("notifications", "push", PushNotificationService)
except ImportError:
    pass



