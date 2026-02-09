"""
Sistema de seguimiento de suplementos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class Supplement:
    """Suplemento"""
    id: str
    user_id: str
    supplement_name: str
    supplement_type: str  # "vitamin", "mineral", "herbal", "probiotic", "collagen", "other"
    dosage: str
    frequency: str  # "daily", "weekly", "as_needed"
    start_date: str
    end_date: Optional[str] = None
    active: bool = True
    purpose: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "supplement_name": self.supplement_name,
            "supplement_type": self.supplement_type,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "active": self.active,
            "purpose": self.purpose,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class SupplementReport:
    """Reporte de suplementos"""
    user_id: str
    active_supplements: List[Supplement]
    total_supplements: int
    by_type: Dict[str, int]
    recommendations: List[str]
    potential_interactions: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "active_supplements": [s.to_dict() for s in self.active_supplements],
            "total_supplements": self.total_supplements,
            "by_type": self.by_type,
            "recommendations": self.recommendations,
            "potential_interactions": self.potential_interactions
        }


class SupplementTracker:
    """Sistema de seguimiento de suplementos"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.supplements: Dict[str, List[Supplement]] = {}  # user_id -> [supplements]
    
    def add_supplement(self, user_id: str, supplement_name: str, supplement_type: str,
                      dosage: str, frequency: str, start_date: str,
                      purpose: Optional[str] = None, notes: Optional[str] = None) -> Supplement:
        """Agrega suplemento"""
        supplement = Supplement(
            id=str(uuid.uuid4()),
            user_id=user_id,
            supplement_name=supplement_name,
            supplement_type=supplement_type,
            dosage=dosage,
            frequency=frequency,
            start_date=start_date,
            purpose=purpose,
            notes=notes
        )
        
        if user_id not in self.supplements:
            self.supplements[user_id] = []
        
        self.supplements[user_id].append(supplement)
        return supplement
    
    def deactivate_supplement(self, supplement_id: str, user_id: str, end_date: str) -> bool:
        """Desactiva suplemento"""
        user_supplements = self.supplements.get(user_id, [])
        
        for supplement in user_supplements:
            if supplement.id == supplement_id:
                supplement.active = False
                supplement.end_date = end_date
                return True
        
        return False
    
    def get_user_supplements(self, user_id: str, active_only: bool = False) -> List[Supplement]:
        """Obtiene suplementos del usuario"""
        user_supplements = self.supplements.get(user_id, [])
        
        if active_only:
            user_supplements = [s for s in user_supplements if s.active]
        
        user_supplements.sort(key=lambda x: x.start_date, reverse=True)
        return user_supplements
    
    def generate_supplement_report(self, user_id: str) -> SupplementReport:
        """Genera reporte de suplementos"""
        user_supplements = self.supplements.get(user_id, [])
        active_supplements = [s for s in user_supplements if s.active]
        
        # Por tipo
        by_type = {}
        for supplement in user_supplements:
            by_type[supplement.supplement_type] = by_type.get(supplement.supplement_type, 0) + 1
        
        # Recomendaciones
        recommendations = []
        
        if len(active_supplements) == 0:
            recommendations.append("No tienes suplementos activos. Considera vitamina D y colágeno para la piel.")
        elif len(active_supplements) > 5:
            recommendations.append("Tienes muchos suplementos activos. Consulta con un médico sobre posibles interacciones.")
        
        # Verificar tipos importantes para la piel
        skin_beneficial = ["vitamin", "collagen", "probiotic"]
        has_skin_beneficial = any(s.supplement_type in skin_beneficial for s in active_supplements)
        
        if not has_skin_beneficial:
            recommendations.append("Considera suplementos beneficiosos para la piel como colágeno, vitamina C, o probióticos")
        
        # Interacciones potenciales
        potential_interactions = []
        
        if len(active_supplements) > 3:
            potential_interactions.append("Múltiples suplementos pueden tener interacciones. Consulta con un profesional.")
        
        return SupplementReport(
            user_id=user_id,
            active_supplements=active_supplements,
            total_supplements=len(user_supplements),
            by_type=by_type,
            recommendations=recommendations,
            potential_interactions=potential_interactions
        )






