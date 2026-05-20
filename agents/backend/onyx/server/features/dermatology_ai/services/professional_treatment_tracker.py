"""
Sistema de seguimiento de tratamientos profesionales
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class ProfessionalTreatment:
    """Tratamiento profesional"""
    id: str
    user_id: str
    treatment_name: str
    treatment_type: str  # "facial", "peel", "laser", "microneedling", "injection"
    provider_name: str
    provider_license: Optional[str] = None
    treatment_date: str
    cost: Optional[float] = None
    duration_minutes: Optional[int] = None
    before_photos: List[str] = None
    after_photos: List[str] = None
    notes: Optional[str] = None
    follow_up_date: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.before_photos is None:
            self.before_photos = []
        if self.after_photos is None:
            self.after_photos = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "treatment_name": self.treatment_name,
            "treatment_type": self.treatment_type,
            "provider_name": self.provider_name,
            "provider_license": self.provider_license,
            "treatment_date": self.treatment_date,
            "cost": self.cost,
            "duration_minutes": self.duration_minutes,
            "before_photos": self.before_photos,
            "after_photos": self.after_photos,
            "notes": self.notes,
            "follow_up_date": self.follow_up_date,
            "created_at": self.created_at
        }


@dataclass
class TreatmentSeries:
    """Serie de tratamientos"""
    id: str
    user_id: str
    treatment_type: str
    treatments: List[ProfessionalTreatment]
    total_sessions: int
    completed_sessions: int
    next_session_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "treatment_type": self.treatment_type,
            "treatments": [t.to_dict() for t in self.treatments],
            "total_sessions": self.total_sessions,
            "completed_sessions": self.completed_sessions,
            "next_session_date": self.next_session_date
        }


class ProfessionalTreatmentTracker:
    """Sistema de seguimiento de tratamientos profesionales"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.treatments: Dict[str, List[ProfessionalTreatment]] = {}  # user_id -> [treatments]
        self.series: Dict[str, List[TreatmentSeries]] = {}  # user_id -> [series]
    
    def add_treatment(self, user_id: str, treatment_name: str, treatment_type: str,
                     provider_name: str, treatment_date: str,
                     provider_license: Optional[str] = None,
                     cost: Optional[float] = None,
                     duration_minutes: Optional[int] = None,
                     before_photos: Optional[List[str]] = None,
                     after_photos: Optional[List[str]] = None,
                     notes: Optional[str] = None,
                     follow_up_date: Optional[str] = None) -> ProfessionalTreatment:
        """Agrega tratamiento profesional"""
        treatment = ProfessionalTreatment(
            id=str(uuid.uuid4()),
            user_id=user_id,
            treatment_name=treatment_name,
            treatment_type=treatment_type,
            provider_name=provider_name,
            provider_license=provider_license,
            treatment_date=treatment_date,
            cost=cost,
            duration_minutes=duration_minutes,
            before_photos=before_photos or [],
            after_photos=after_photos or [],
            notes=notes,
            follow_up_date=follow_up_date
        )
        
        if user_id not in self.treatments:
            self.treatments[user_id] = []
        
        self.treatments[user_id].append(treatment)
        return treatment
    
    def create_treatment_series(self, user_id: str, treatment_type: str,
                               total_sessions: int) -> TreatmentSeries:
        """Crea serie de tratamientos"""
        series = TreatmentSeries(
            id=str(uuid.uuid4()),
            user_id=user_id,
            treatment_type=treatment_type,
            treatments=[],
            total_sessions=total_sessions,
            completed_sessions=0
        )
        
        if user_id not in self.series:
            self.series[user_id] = []
        
        self.series[user_id].append(series)
        return series
    
    def add_treatment_to_series(self, series_id: str, treatment: ProfessionalTreatment) -> bool:
        """Agrega tratamiento a serie"""
        for user_series in self.series.values():
            for series in user_series:
                if series.id == series_id:
                    series.treatments.append(treatment)
                    series.completed_sessions = len(series.treatments)
                    return True
        return False
    
    def get_user_treatments(self, user_id: str, treatment_type: Optional[str] = None) -> List[ProfessionalTreatment]:
        """Obtiene tratamientos del usuario"""
        user_treatments = self.treatments.get(user_id, [])
        
        if treatment_type:
            user_treatments = [t for t in user_treatments if t.treatment_type == treatment_type]
        
        user_treatments.sort(key=lambda x: x.treatment_date, reverse=True)
        return user_treatments
    
    def get_treatment_statistics(self, user_id: str) -> Dict:
        """Obtiene estadísticas de tratamientos"""
        user_treatments = self.treatments.get(user_id, [])
        
        total_treatments = len(user_treatments)
        total_cost = sum(t.cost for t in user_treatments if t.cost)
        
        by_type = {}
        for treatment in user_treatments:
            by_type[treatment.treatment_type] = by_type.get(treatment.treatment_type, 0) + 1
        
        return {
            "total_treatments": total_treatments,
            "total_cost": total_cost,
            "by_type": by_type,
            "average_cost": total_cost / total_treatments if total_treatments > 0 else 0.0
        }






