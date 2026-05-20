"""
Calendar Integration Service - Integración con calendarios
===========================================================

Sistema para integrar con Google Calendar, Outlook, etc.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CalendarType(str):
    """Tipos de calendarios"""
    GOOGLE = "google"
    OUTLOOK = "outlook"
    APPLE = "apple"
    ICAL = "ical"


@dataclass
class CalendarEvent:
    """Evento de calendario"""
    id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    attendees: List[str] = field(default_factory=list)
    reminder_minutes: int = 15
    calendar_type: str = CalendarType.GOOGLE


class CalendarIntegrationService:
    """Servicio de integración con calendarios"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.events: Dict[str, List[CalendarEvent]] = {}  # user_id -> [events]
        logger.info("CalendarIntegrationService initialized")
    
    def create_event(
        self,
        user_id: str,
        title: str,
        description: str,
        start_time: datetime,
        end_time: datetime,
        location: Optional[str] = None,
        reminder_minutes: int = 15
    ) -> CalendarEvent:
        """Crear evento en calendario"""
        event = CalendarEvent(
            id=f"cal_event_{user_id}_{int(datetime.now().timestamp())}",
            title=title,
            description=description,
            start_time=start_time,
            end_time=end_time,
            location=location,
            reminder_minutes=reminder_minutes,
        )
        
        if user_id not in self.events:
            self.events[user_id] = []
        
        self.events[user_id].append(event)
        
        logger.info(f"Calendar event created for user {user_id}: {title}")
        return event
    
    def schedule_interview_reminder(
        self,
        user_id: str,
        interview_date: datetime,
        job_title: str,
        company: str,
        location: Optional[str] = None
    ) -> CalendarEvent:
        """Programar recordatorio de entrevista"""
        # Recordatorio 1 día antes
        reminder_time = interview_date - timedelta(days=1)
        
        return self.create_event(
            user_id=user_id,
            title=f"Recordatorio: Entrevista - {job_title} en {company}",
            description=f"Tu entrevista para {job_title} en {company} es mañana. ¡Prepárate!",
            start_time=reminder_time,
            end_time=reminder_time + timedelta(minutes=30),
            location=location,
            reminder_minutes=1440,  # 1 día antes
        )
    
    def get_upcoming_events(self, user_id: str, days: int = 7) -> List[CalendarEvent]:
        """Obtener eventos próximos"""
        if user_id not in self.events:
            return []
        
        now = datetime.now()
        end_date = now + timedelta(days=days)
        
        upcoming = [
            event for event in self.events[user_id]
            if now <= event.start_time <= end_date
        ]
        
        upcoming.sort(key=lambda x: x.start_time)
        return upcoming
    
    def export_to_ical(self, user_id: str) -> str:
        """Exportar eventos a formato iCal"""
        events = self.events.get(user_id, [])
        
        ical_content = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//AI Job Helper//EN\n"
        
        for event in events:
            ical_content += f"""
BEGIN:VEVENT
UID:{event.id}
DTSTART:{event.start_time.strftime('%Y%m%dT%H%M%S')}
DTEND:{event.end_time.strftime('%Y%m%dT%H%M%S')}
SUMMARY:{event.title}
DESCRIPTION:{event.description}
END:VEVENT
"""
        
        ical_content += "END:VCALENDAR\n"
        return ical_content




