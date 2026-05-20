"""
Reminders Service - Sistema de recordatorios avanzado
======================================================

Sistema de recordatorios inteligentes y programados.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ReminderType(str, Enum):
    """Tipos de recordatorios"""
    STEP = "step"
    APPLICATION = "application"
    INTERVIEW = "interview"
    SKILL_LEARNING = "skill_learning"
    NETWORKING = "networking"
    FOLLOW_UP = "follow_up"
    CUSTOM = "custom"


class ReminderStatus(str, Enum):
    """Estado del recordatorio"""
    PENDING = "pending"
    SENT = "sent"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Reminder:
    """Recordatorio"""
    id: str
    user_id: str
    reminder_type: ReminderType
    title: str
    message: str
    due_date: datetime
    status: ReminderStatus = ReminderStatus.PENDING
    recurring: bool = False
    recurring_interval_days: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class RemindersService:
    """Servicio de recordatorios"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.reminders: Dict[str, List[Reminder]] = {}  # user_id -> [reminders]
        logger.info("RemindersService initialized")
    
    def create_reminder(
        self,
        user_id: str,
        reminder_type: ReminderType,
        title: str,
        message: str,
        due_date: datetime,
        recurring: bool = False,
        recurring_interval_days: Optional[int] = None
    ) -> Reminder:
        """Crear recordatorio"""
        reminder = Reminder(
            id=f"reminder_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            reminder_type=reminder_type,
            title=title,
            message=message,
            due_date=due_date,
            recurring=recurring,
            recurring_interval_days=recurring_interval_days,
        )
        
        if user_id not in self.reminders:
            self.reminders[user_id] = []
        
        self.reminders[user_id].append(reminder)
        
        logger.info(f"Reminder created for user {user_id}: {title}")
        return reminder
    
    def get_due_reminders(self, user_id: Optional[str] = None) -> List[Reminder]:
        """Obtener recordatorios vencidos"""
        now = datetime.now()
        due_reminders = []
        
        users_to_check = [user_id] if user_id else self.reminders.keys()
        
        for uid in users_to_check:
            if uid in self.reminders:
                for reminder in self.reminders[uid]:
                    if reminder.status == ReminderStatus.PENDING and reminder.due_date <= now:
                        due_reminders.append(reminder)
        
        return due_reminders
    
    def mark_as_sent(self, reminder_id: str, user_id: str) -> bool:
        """Marcar recordatorio como enviado"""
        if user_id not in self.reminders:
            return False
        
        for reminder in self.reminders[user_id]:
            if reminder.id == reminder_id:
                reminder.status = ReminderStatus.SENT
                reminder.sent_at = datetime.now()
                
                # Si es recurrente, crear próximo recordatorio
                if reminder.recurring and reminder.recurring_interval_days:
                    next_due = reminder.due_date + timedelta(days=reminder.recurring_interval_days)
                    self.create_reminder(
                        user_id=user_id,
                        reminder_type=reminder.reminder_type,
                        title=reminder.title,
                        message=reminder.message,
                        due_date=next_due,
                        recurring=True,
                        recurring_interval_days=reminder.recurring_interval_days
                    )
                
                return True
        
        return False
    
    def complete_reminder(self, reminder_id: str, user_id: str) -> bool:
        """Completar recordatorio"""
        if user_id not in self.reminders:
            return False
        
        for reminder in self.reminders[user_id]:
            if reminder.id == reminder_id:
                reminder.status = ReminderStatus.COMPLETED
                reminder.completed_at = datetime.now()
                return True
        
        return False
    
    def get_user_reminders(
        self,
        user_id: str,
        status: Optional[ReminderStatus] = None
    ) -> List[Reminder]:
        """Obtener recordatorios del usuario"""
        reminders = self.reminders.get(user_id, [])
        
        if status:
            reminders = [r for r in reminders if r.status == status]
        
        reminders.sort(key=lambda x: x.due_date)
        return reminders
    
    def create_interview_reminder(
        self,
        user_id: str,
        interview_date: datetime,
        job_title: str,
        company: str
    ) -> Reminder:
        """Crear recordatorio de entrevista"""
        # Recordatorio 1 día antes
        reminder_date = interview_date - timedelta(days=1)
        
        return self.create_reminder(
            user_id=user_id,
            reminder_type=ReminderType.INTERVIEW,
            title=f"Entrevista mañana: {job_title}",
            message=f"Tu entrevista para {job_title} en {company} es mañana. ¡Prepárate!",
            due_date=reminder_date,
        )
    
    def create_follow_up_reminder(
        self,
        user_id: str,
        application_date: datetime,
        company: str
    ) -> Reminder:
        """Crear recordatorio de follow-up"""
        # Follow-up 1 semana después
        reminder_date = application_date + timedelta(days=7)
        
        return self.create_reminder(
            user_id=user_id,
            reminder_type=ReminderType.FOLLOW_UP,
            title=f"Follow-up: {company}",
            message=f"Es hora de hacer follow-up con {company} sobre tu aplicación.",
            due_date=reminder_date,
        )




