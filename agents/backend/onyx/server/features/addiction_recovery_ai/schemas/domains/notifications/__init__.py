"""
Notifications domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.notifications import (
        NotificationResponse,
        NotificationsListResponse,
        ReminderResponse,
        RemindersListResponse
    )
    
    def register_schemas():
        register_schema("notifications", "NotificationResponse", NotificationResponse)
        register_schema("notifications", "NotificationsListResponse", NotificationsListResponse)
        register_schema("notifications", "ReminderResponse", ReminderResponse)
        register_schema("notifications", "RemindersListResponse", RemindersListResponse)
except ImportError:
    pass



