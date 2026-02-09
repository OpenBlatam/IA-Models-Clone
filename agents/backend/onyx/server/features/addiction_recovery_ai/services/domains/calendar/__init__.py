"""
Calendar domain services
"""

from services.domains import register_service

try:
    from services.calendar_service import CalendarService
    
    def register_services():
        register_service("calendar", "calendar", CalendarService)
except ImportError:
    pass



