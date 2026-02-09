"""
Sistema de consultas con expertos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


class ConsultationStatus(str, Enum):
    """Estado de consulta"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExpertConsultation:
    """Consulta con experto"""
    id: str
    user_id: str
    expert_id: str
    expert_name: str
    consultation_type: str  # "video", "chat", "photo_review"
    status: ConsultationStatus
    scheduled_time: Optional[str] = None
    duration_minutes: int = 30
    notes: Optional[str] = None
    recommendations: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "expert_id": self.expert_id,
            "expert_name": self.expert_name,
            "consultation_type": self.consultation_type,
            "status": self.status.value,
            "scheduled_time": self.scheduled_time,
            "duration_minutes": self.duration_minutes,
            "notes": self.notes,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class ExpertConsultationSystem:
    """Sistema de consultas con expertos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.consultations: Dict[str, List[ExpertConsultation]] = {}  # user_id -> [consultations]
        self.experts: Dict[str, Dict] = {
            "expert1": {"name": "Dr. Sarah Johnson", "specialty": "Dermatology", "rating": 4.8},
            "expert2": {"name": "Dr. Michael Chen", "specialty": "Skincare", "rating": 4.9},
            "expert3": {"name": "Dr. Emily Rodriguez", "specialty": "Cosmetic Dermatology", "rating": 4.7}
        }
    
    def request_consultation(self, user_id: str, expert_id: str,
                             consultation_type: str,
                             scheduled_time: Optional[str] = None) -> ExpertConsultation:
        """Solicita una consulta"""
        expert = self.experts.get(expert_id, {"name": "Unknown Expert"})
        
        consultation = ExpertConsultation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            expert_id=expert_id,
            expert_name=expert["name"],
            consultation_type=consultation_type,
            status=ConsultationStatus.SCHEDULED if scheduled_time else ConsultationStatus.PENDING,
            scheduled_time=scheduled_time
        )
        
        if user_id not in self.consultations:
            self.consultations[user_id] = []
        
        self.consultations[user_id].append(consultation)
        return consultation
    
    def get_available_experts(self) -> List[Dict]:
        """Obtiene expertos disponibles"""
        return [
            {"id": eid, **info}
            for eid, info in self.experts.items()
        ]
    
    def get_user_consultations(self, user_id: str,
                               status: Optional[ConsultationStatus] = None) -> List[ExpertConsultation]:
        """Obtiene consultas del usuario"""
        user_consultations = self.consultations.get(user_id, [])
        
        if status:
            user_consultations = [c for c in user_consultations if c.status == status]
        
        user_consultations.sort(key=lambda x: x.created_at, reverse=True)
        return user_consultations
    
    def update_consultation(self, user_id: str, consultation_id: str,
                           notes: Optional[str] = None,
                           recommendations: Optional[List[str]] = None,
                           status: Optional[ConsultationStatus] = None) -> bool:
        """Actualiza una consulta"""
        user_consultations = self.consultations.get(user_id, [])
        
        for consultation in user_consultations:
            if consultation.id == consultation_id:
                if notes is not None:
                    consultation.notes = notes
                if recommendations is not None:
                    consultation.recommendations = recommendations
                if status is not None:
                    consultation.status = status
                return True
        
        return False
    
    def cancel_consultation(self, user_id: str, consultation_id: str) -> bool:
        """Cancela una consulta"""
        return self.update_consultation(
            user_id, consultation_id, status=ConsultationStatus.CANCELLED
        )






