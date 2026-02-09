"""
Sistema de seguimiento de tratamientos médicos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class MedicalTreatment:
    """Tratamiento médico"""
    id: str
    user_id: str
    treatment_name: str
    treatment_type: str  # "prescription", "procedure", "therapy"
    doctor_name: Optional[str] = None
    start_date: str
    end_date: Optional[str] = None
    frequency: str  # "daily", "weekly", "as_needed"
    dosage: Optional[str] = None
    instructions: List[str] = None
    side_effects: List[str] = None
    status: str = "active"  # "active", "completed", "discontinued"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.instructions is None:
            self.instructions = []
        if self.side_effects is None:
            self.side_effects = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "treatment_name": self.treatment_name,
            "treatment_type": self.treatment_type,
            "doctor_name": self.doctor_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "frequency": self.frequency,
            "dosage": self.dosage,
            "instructions": self.instructions,
            "side_effects": self.side_effects,
            "status": self.status,
            "created_at": self.created_at
        }


@dataclass
class TreatmentProgress:
    """Progreso del tratamiento"""
    treatment_id: str
    date: str
    adherence: bool  # Si siguió el tratamiento
    notes: Optional[str] = None
    side_effects_reported: List[str] = None
    effectiveness_rating: Optional[int] = None  # 1-5
    
    def __post_init__(self):
        if self.side_effects_reported is None:
            self.side_effects_reported = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "treatment_id": self.treatment_id,
            "date": self.date,
            "adherence": self.adherence,
            "notes": self.notes,
            "side_effects_reported": self.side_effects_reported,
            "effectiveness_rating": self.effectiveness_rating
        }


class MedicalTreatmentTracker:
    """Sistema de seguimiento de tratamientos médicos"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.treatments: Dict[str, List[MedicalTreatment]] = {}  # user_id -> [treatments]
        self.progress: Dict[str, List[TreatmentProgress]] = {}  # treatment_id -> [progress]
    
    def add_treatment(self, user_id: str, treatment_name: str, treatment_type: str,
                     start_date: str, frequency: str,
                     doctor_name: Optional[str] = None,
                     end_date: Optional[str] = None,
                     dosage: Optional[str] = None,
                     instructions: Optional[List[str]] = None) -> MedicalTreatment:
        """Agrega un tratamiento"""
        treatment = MedicalTreatment(
            id=str(uuid.uuid4()),
            user_id=user_id,
            treatment_name=treatment_name,
            treatment_type=treatment_type,
            doctor_name=doctor_name,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            dosage=dosage,
            instructions=instructions or []
        )
        
        if user_id not in self.treatments:
            self.treatments[user_id] = []
        
        self.treatments[user_id].append(treatment)
        return treatment
    
    def record_progress(self, treatment_id: str, date: str, adherence: bool,
                       notes: Optional[str] = None,
                       side_effects: Optional[List[str]] = None,
                       effectiveness_rating: Optional[int] = None) -> TreatmentProgress:
        """Registra progreso del tratamiento"""
        progress = TreatmentProgress(
            treatment_id=treatment_id,
            date=date,
            adherence=adherence,
            notes=notes,
            side_effects_reported=side_effects or [],
            effectiveness_rating=effectiveness_rating
        )
        
        if treatment_id not in self.progress:
            self.progress[treatment_id] = []
        
        self.progress[treatment_id].append(progress)
        return progress
    
    def get_user_treatments(self, user_id: str, status: Optional[str] = None) -> List[MedicalTreatment]:
        """Obtiene tratamientos del usuario"""
        user_treatments = self.treatments.get(user_id, [])
        
        if status:
            user_treatments = [t for t in user_treatments if t.status == status]
        
        return user_treatments
    
    def get_treatment_progress(self, treatment_id: str) -> List[TreatmentProgress]:
        """Obtiene progreso de un tratamiento"""
        return self.progress.get(treatment_id, [])
    
    def get_adherence_rate(self, treatment_id: str) -> float:
        """Calcula tasa de adherencia"""
        progress_records = self.progress.get(treatment_id, [])
        
        if not progress_records:
            return 0.0
        
        adhered = sum(1 for p in progress_records if p.adherence)
        return (adhered / len(progress_records)) * 100.0






