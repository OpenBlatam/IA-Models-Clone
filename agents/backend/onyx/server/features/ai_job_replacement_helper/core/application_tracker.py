"""
Application Tracker Service - Seguimiento de aplicaciones
=========================================================

Sistema para rastrear el estado de todas las aplicaciones de trabajo.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ApplicationStatus(str, Enum):
    """Estado de la aplicación"""
    DRAFT = "draft"
    APPLIED = "applied"
    VIEWED = "viewed"
    IN_REVIEW = "in_review"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    INTERVIEW_COMPLETED = "interview_completed"
    OFFER_RECEIVED = "offer_received"
    OFFER_ACCEPTED = "offer_accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


@dataclass
class Application:
    """Aplicación de trabajo"""
    id: str
    user_id: str
    job_id: str
    job_title: str
    company: str
    platform: str
    status: ApplicationStatus
    applied_date: datetime
    cover_letter: Optional[str] = None
    notes: Optional[str] = None
    salary_expectation: Optional[str] = None
    next_action_date: Optional[datetime] = None
    next_action: Optional[str] = None
    interview_dates: List[datetime] = field(default_factory=list)
    offer_details: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.now)


class ApplicationTrackerService:
    """Servicio de seguimiento de aplicaciones"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.applications: Dict[str, List[Application]] = {}  # user_id -> [applications]
        logger.info("ApplicationTrackerService initialized")
    
    def create_application(
        self,
        user_id: str,
        job_id: str,
        job_title: str,
        company: str,
        platform: str,
        cover_letter: Optional[str] = None,
        status: ApplicationStatus = ApplicationStatus.APPLIED
    ) -> Application:
        """Crear nueva aplicación"""
        application = Application(
            id=f"app_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            job_id=job_id,
            job_title=job_title,
            company=company,
            platform=platform,
            status=status,
            applied_date=datetime.now(),
            cover_letter=cover_letter,
        )
        
        if user_id not in self.applications:
            self.applications[user_id] = []
        
        self.applications[user_id].append(application)
        
        logger.info(f"Application created: {application.id}")
        return application
    
    def update_application_status(
        self,
        user_id: str,
        application_id: str,
        new_status: ApplicationStatus,
        notes: Optional[str] = None
    ) -> Application:
        """Actualizar estado de aplicación"""
        application = self._get_application(user_id, application_id)
        if not application:
            raise ValueError(f"Application {application_id} not found")
        
        application.status = new_status
        application.updated_at = datetime.now()
        
        if notes:
            application.notes = notes
        
        # Establecer próxima acción según estado
        self._set_next_action(application, new_status)
        
        return application
    
    def add_interview(
        self,
        user_id: str,
        application_id: str,
        interview_date: datetime,
        interview_type: Optional[str] = None
    ) -> Application:
        """Agregar fecha de entrevista"""
        application = self._get_application(user_id, application_id)
        if not application:
            raise ValueError(f"Application {application_id} not found")
        
        application.interview_dates.append(interview_date)
        application.status = ApplicationStatus.INTERVIEW_SCHEDULED
        application.next_action_date = interview_date
        application.next_action = f"Interview: {interview_type or 'General'}"
        
        return application
    
    def add_offer(
        self,
        user_id: str,
        application_id: str,
        offer_details: Dict[str, Any]
    ) -> Application:
        """Agregar oferta recibida"""
        application = self._get_application(user_id, application_id)
        if not application:
            raise ValueError(f"Application {application_id} not found")
        
        application.status = ApplicationStatus.OFFER_RECEIVED
        application.offer_details = offer_details
        application.next_action = "Review and respond to offer"
        application.next_action_date = datetime.now() + timedelta(days=7)  # Típicamente 7 días para responder
        
        return application
    
    def reject_application(
        self,
        user_id: str,
        application_id: str,
        reason: Optional[str] = None
    ) -> Application:
        """Marcar aplicación como rechazada"""
        application = self._get_application(user_id, application_id)
        if not application:
            raise ValueError(f"Application {application_id} not found")
        
        application.status = ApplicationStatus.REJECTED
        application.rejection_reason = reason
        application.next_action = None
        application.next_action_date = None
        
        return application
    
    def get_user_applications(
        self,
        user_id: str,
        status: Optional[ApplicationStatus] = None
    ) -> List[Application]:
        """Obtener aplicaciones del usuario"""
        applications = self.applications.get(user_id, [])
        
        if status:
            applications = [a for a in applications if a.status == status]
        
        # Ordenar por fecha (más recientes primero)
        applications.sort(key=lambda x: x.applied_date, reverse=True)
        
        return applications
    
    def get_application_statistics(self, user_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de aplicaciones"""
        applications = self.applications.get(user_id, [])
        
        if not applications:
            return {
                "total": 0,
                "by_status": {},
                "success_rate": 0.0,
                "average_response_time": None,
            }
        
        # Contar por estado
        by_status = {}
        for status in ApplicationStatus:
            count = sum(1 for a in applications if a.status == status)
            if count > 0:
                by_status[status.value] = count
        
        # Calcular tasa de éxito (ofertas / total aplicadas)
        total_applied = sum(
            1 for a in applications
            if a.status in [
                ApplicationStatus.APPLIED,
                ApplicationStatus.IN_REVIEW,
                ApplicationStatus.INTERVIEW_SCHEDULED,
                ApplicationStatus.INTERVIEW_COMPLETED,
                ApplicationStatus.OFFER_RECEIVED,
                ApplicationStatus.OFFER_ACCEPTED,
                ApplicationStatus.REJECTED,
            ]
        )
        
        offers_received = sum(
            1 for a in applications
            if a.status == ApplicationStatus.OFFER_RECEIVED
        )
        
        success_rate = (offers_received / total_applied * 100) if total_applied > 0 else 0.0
        
        return {
            "total": len(applications),
            "by_status": by_status,
            "success_rate": round(success_rate, 2),
            "offers_received": offers_received,
            "interviews_scheduled": sum(
                1 for a in applications
                if a.status == ApplicationStatus.INTERVIEW_SCHEDULED
            ),
            "pending_response": sum(
                1 for a in applications
                if a.status in [
                    ApplicationStatus.APPLIED,
                    ApplicationStatus.VIEWED,
                    ApplicationStatus.IN_REVIEW,
                ]
            ),
        }
    
    def get_upcoming_actions(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener próximas acciones pendientes"""
        applications = self.applications.get(user_id, [])
        
        upcoming = []
        for app in applications:
            if app.next_action_date and app.next_action:
                upcoming.append({
                    "application_id": app.id,
                    "job_title": app.job_title,
                    "company": app.company,
                    "action": app.next_action,
                    "due_date": app.next_action_date.isoformat(),
                    "days_until": (app.next_action_date.date() - datetime.now().date()).days,
                })
        
        # Ordenar por fecha
        upcoming.sort(key=lambda x: x["due_date"])
        
        return upcoming
    
    def _get_application(self, user_id: str, application_id: str) -> Optional[Application]:
        """Obtener aplicación por ID"""
        applications = self.applications.get(user_id, [])
        return next((a for a in applications if a.id == application_id), None)
    
    def _set_next_action(self, application: Application, status: ApplicationStatus):
        """Establecer próxima acción según estado"""
        if status == ApplicationStatus.APPLIED:
            application.next_action = "Wait for response"
            application.next_action_date = datetime.now() + timedelta(days=7)
        elif status == ApplicationStatus.IN_REVIEW:
            application.next_action = "Follow up if no response"
            application.next_action_date = datetime.now() + timedelta(days=14)
        elif status == ApplicationStatus.INTERVIEW_SCHEDULED:
            application.next_action = "Prepare for interview"
            application.next_action_date = application.interview_dates[0] if application.interview_dates else None
        elif status == ApplicationStatus.OFFER_RECEIVED:
            application.next_action = "Review and respond to offer"
            application.next_action_date = datetime.now() + timedelta(days=7)
        else:
            application.next_action = None
            application.next_action_date = None

