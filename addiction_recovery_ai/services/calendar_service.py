"""
Servicio de calendario - Gestión de recordatorios y eventos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class EventType(str, Enum):
    """Tipos de eventos"""
    REMINDER = "reminder"
    MILESTONE = "milestone"
    COACHING = "coaching"
    SUPPORT_GROUP = "support_group"
    CHECK_IN = "check_in"
    CUSTOM = "custom"


class CalendarService:
    """Servicio de calendario y recordatorios"""
    
    def __init__(self):
        """Inicializa el servicio de calendario"""
        pass
    
    def create_event(
        self,
        user_id: str,
        event_type: str,
        title: str,
        description: str,
        scheduled_time: datetime,
        repeat_daily: bool = False,
        repeat_weekly: bool = False,
        reminder_minutes: int = 15
    ) -> Dict:
        """
        Crea un evento en el calendario
        
        Args:
            user_id: ID del usuario
            event_type: Tipo de evento
            title: Título del evento
            description: Descripción
            scheduled_time: Hora programada
            repeat_daily: Si se repite diariamente
            repeat_weekly: Si se repite semanalmente
            reminder_minutes: Minutos antes para recordatorio
        
        Returns:
            Evento creado
        """
        event = {
            "user_id": user_id,
            "event_type": event_type,
            "title": title,
            "description": description,
            "scheduled_time": scheduled_time.isoformat(),
            "repeat_daily": repeat_daily,
            "repeat_weekly": repeat_weekly,
            "reminder_minutes": reminder_minutes,
            "reminder_time": (scheduled_time - timedelta(minutes=reminder_minutes)).isoformat(),
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        return event
    
    def get_upcoming_events(
        self,
        user_id: str,
        days_ahead: int = 7
    ) -> List[Dict]:
        """
        Obtiene eventos próximos
        
        Args:
            user_id: ID del usuario
            days_ahead: Días hacia adelante para buscar
        
        Returns:
            Lista de eventos próximos
        """
        # En implementación real, esto vendría de la base de datos
        # Por ahora, generamos eventos de ejemplo basados en el plan de recuperación
        upcoming = []
        
        today = datetime.now()
        for i in range(days_ahead):
            date = today + timedelta(days=i)
            
            # Evento de check-in diario
            if i == 0:  # Hoy
                upcoming.append({
                    "date": date.isoformat(),
                    "title": "Check-in Diario",
                    "description": "Registra tu estado emocional y progreso",
                    "time": "09:00",
                    "type": EventType.CHECK_IN
                })
            
            # Evento de reflexión nocturna
            if i == 0:  # Hoy
                upcoming.append({
                    "date": date.isoformat(),
                    "title": "Reflexión Nocturna",
                    "description": "Tómate un momento para reflexionar sobre tu día",
                    "time": "20:00",
                    "type": EventType.REMINDER
                })
        
        return sorted(upcoming, key=lambda x: x.get("date", ""))
    
    def create_daily_reminders(self, user_id: str) -> List[Dict]:
        """
        Crea recordatorios diarios automáticos
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de recordatorios creados
        """
        reminders = []
        today = datetime.now()
        
        # Check-in matutino
        morning_time = today.replace(hour=9, minute=0, second=0, microsecond=0)
        reminders.append(self.create_event(
            user_id=user_id,
            event_type=EventType.CHECK_IN,
            title="Check-in Matutino",
            description="Registra cómo te sientes al despertar",
            scheduled_time=morning_time,
            repeat_daily=True,
            reminder_minutes=0
        ))
        
        # Reflexión nocturna
        evening_time = today.replace(hour=20, minute=0, second=0, microsecond=0)
        reminders.append(self.create_event(
            user_id=user_id,
            event_type=EventType.REMINDER,
            title="Reflexión Nocturna",
            description="Reflexiona sobre tu día y completa tu entrada diaria",
            scheduled_time=evening_time,
            repeat_daily=True,
            reminder_minutes=15
        ))
        
        return reminders
    
    def get_events_for_date(
        self,
        user_id: str,
        target_date: datetime
    ) -> List[Dict]:
        """
        Obtiene eventos para una fecha específica
        
        Args:
            user_id: ID del usuario
            target_date: Fecha objetivo
        
        Returns:
            Lista de eventos para esa fecha
        """
        # En implementación real, esto consultaría la base de datos
        events = []
        
        # Verificar si hay hitos próximos
        # Esto se calcularía basado en el progreso del usuario
        
        return events
    
    def delete_event(self, event_id: str) -> bool:
        """
        Elimina un evento
        
        Args:
            event_id: ID del evento
        
        Returns:
            True si se eliminó exitosamente
        """
        # En implementación real, esto actualizaría la base de datos
        return True
    
    def update_event(self, event_id: str, updates: Dict) -> Dict:
        """
        Actualiza un evento
        
        Args:
            event_id: ID del evento
            updates: Diccionario con campos a actualizar
        
        Returns:
            Evento actualizado
        """
        # En implementación real, esto actualizaría la base de datos
        return {
            "event_id": event_id,
            **updates,
            "updated_at": datetime.now().isoformat()
        }

