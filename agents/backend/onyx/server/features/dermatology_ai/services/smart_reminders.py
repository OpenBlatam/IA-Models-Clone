"""
Sistema de recordatorios inteligentes
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid


class ReminderType(str, Enum):
    """Tipo de recordatorio"""
    ROUTINE = "routine"
    PRODUCT = "product"
    ANALYSIS = "analysis"
    APPOINTMENT = "appointment"
    CUSTOM = "custom"


@dataclass
class SmartReminder:
    """Recordatorio inteligente"""
    id: str
    user_id: str
    type: ReminderType
    title: str
    message: str
    scheduled_time: str
    frequency: str  # "once", "daily", "weekly"
    enabled: bool = True
    completed: bool = False
    completed_at: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "scheduled_time": self.scheduled_time,
            "frequency": self.frequency,
            "enabled": self.enabled,
            "completed": self.completed,
            "completed_at": self.completed_at,
            "created_at": self.created_at
        }


class SmartReminderSystem:
    """Sistema de recordatorios inteligentes"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reminders: Dict[str, List[SmartReminder]] = {}  # user_id -> [reminders]
    
    def create_reminder(self, user_id: str, type: ReminderType, title: str,
                       message: str, scheduled_time: str,
                       frequency: str = "once") -> SmartReminder:
        """Crea un recordatorio"""
        reminder = SmartReminder(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=type,
            title=title,
            message=message,
            scheduled_time=scheduled_time,
            frequency=frequency
        )
        
        if user_id not in self.reminders:
            self.reminders[user_id] = []
        
        self.reminders[user_id].append(reminder)
        return reminder
    
    def create_routine_reminder(self, user_id: str, routine_name: str,
                                time: str, morning: bool = True) -> SmartReminder:
        """Crea recordatorio de rutina"""
        reminder_type = ReminderType.ROUTINE
        title = f"Rutina {'Matutina' if morning else 'Nocturna'}"
        message = f"Es hora de tu rutina: {routine_name}"
        
        return self.create_reminder(
            user_id=user_id,
            type=reminder_type,
            title=title,
            message=message,
            scheduled_time=time,
            frequency="daily"
        )
    
    def create_analysis_reminder(self, user_id: str, days_interval: int = 7) -> SmartReminder:
        """Crea recordatorio de análisis"""
        next_date = datetime.now() + timedelta(days=days_interval)
        
        return self.create_reminder(
            user_id=user_id,
            type=ReminderType.ANALYSIS,
            title="Análisis de Piel",
            message="Es momento de realizar un nuevo análisis de tu piel",
            scheduled_time=next_date.isoformat(),
            frequency="weekly"
        )
    
    def get_pending_reminders(self, user_id: str) -> List[SmartReminder]:
        """Obtiene recordatorios pendientes"""
        user_reminders = self.reminders.get(user_id, [])
        
        now = datetime.now()
        pending = []
        
        for reminder in user_reminders:
            if not reminder.enabled or reminder.completed:
                continue
            
            scheduled = datetime.fromisoformat(reminder.scheduled_time)
            
            # Verificar si es hora del recordatorio
            if scheduled <= now:
                pending.append(reminder)
        
        # Ordenar por tiempo programado
        pending.sort(key=lambda x: x.scheduled_time)
        
        return pending
    
    def mark_completed(self, user_id: str, reminder_id: str) -> bool:
        """Marca recordatorio como completado"""
        user_reminders = self.reminders.get(user_id, [])
        
        for reminder in user_reminders:
            if reminder.id == reminder_id:
                reminder.completed = True
                reminder.completed_at = datetime.now().isoformat()
                
                # Si es recurrente, crear próximo recordatorio
                if reminder.frequency == "daily":
                    next_time = datetime.fromisoformat(reminder.scheduled_time) + timedelta(days=1)
                    reminder.scheduled_time = next_time.isoformat()
                    reminder.completed = False
                    reminder.completed_at = None
                elif reminder.frequency == "weekly":
                    next_time = datetime.fromisoformat(reminder.scheduled_time) + timedelta(days=7)
                    reminder.scheduled_time = next_time.isoformat()
                    reminder.completed = False
                    reminder.completed_at = None
                
                return True
        
        return False
    
    def get_user_reminders(self, user_id: str, include_completed: bool = False) -> List[SmartReminder]:
        """Obtiene todos los recordatorios del usuario"""
        user_reminders = self.reminders.get(user_id, [])
        
        if not include_completed:
            user_reminders = [r for r in user_reminders if not r.completed]
        
        user_reminders.sort(key=lambda x: x.scheduled_time)
        return user_reminders
    
    def disable_reminder(self, user_id: str, reminder_id: str) -> bool:
        """Desactiva un recordatorio"""
        user_reminders = self.reminders.get(user_id, [])
        
        for reminder in user_reminders:
            if reminder.id == reminder_id:
                reminder.enabled = False
                return True
        
        return False






