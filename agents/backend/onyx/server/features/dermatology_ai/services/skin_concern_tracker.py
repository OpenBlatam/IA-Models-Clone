"""
Sistema de seguimiento de preocupaciones de la piel
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class SkinConcern:
    """Preocupación de la piel"""
    id: str
    user_id: str
    concern_type: str  # "acne", "wrinkles", "dark_spots", "dryness", "oiliness", "sensitivity", "redness"
    severity: str  # "mild", "moderate", "severe"
    location: Optional[str] = None  # "forehead", "cheeks", "chin", "nose", "all_over"
    first_noticed: str = None
    current_status: str = "active"  # "active", "improving", "resolved"
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.first_noticed is None:
            self.first_noticed = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "concern_type": self.concern_type,
            "severity": self.severity,
            "location": self.location,
            "first_noticed": self.first_noticed,
            "current_status": self.current_status,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class ConcernAnalysis:
    """Análisis de preocupaciones"""
    user_id: str
    active_concerns: List[SkinConcern]
    improving_concerns: List[SkinConcern]
    resolved_concerns: List[SkinConcern]
    primary_concern: Optional[str] = None
    recommendations: List[str] = None
    progress_summary: Dict = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.progress_summary is None:
            self.progress_summary = {}
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "active_concerns": [c.to_dict() for c in self.active_concerns],
            "improving_concerns": [c.to_dict() for c in self.improving_concerns],
            "resolved_concerns": [c.to_dict() for c in self.resolved_concerns],
            "primary_concern": self.primary_concern,
            "recommendations": self.recommendations,
            "progress_summary": self.progress_summary
        }


class SkinConcernTracker:
    """Sistema de seguimiento de preocupaciones de la piel"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.concerns: Dict[str, List[SkinConcern]] = {}
    
    def add_concern(self, user_id: str, concern_type: str, severity: str,
                   location: Optional[str] = None,
                   notes: Optional[str] = None) -> SkinConcern:
        """Agrega preocupación"""
        concern = SkinConcern(
            id=str(uuid.uuid4()),
            user_id=user_id,
            concern_type=concern_type,
            severity=severity,
            location=location,
            notes=notes
        )
        
        if user_id not in self.concerns:
            self.concerns[user_id] = []
        
        self.concerns[user_id].append(concern)
        return concern
    
    def update_concern_status(self, concern_id: str, user_id: str,
                             new_status: str) -> bool:
        """Actualiza estado de preocupación"""
        user_concerns = self.concerns.get(user_id, [])
        
        for concern in user_concerns:
            if concern.id == concern_id:
                concern.current_status = new_status
                return True
        
        return False
    
    def analyze_concerns(self, user_id: str) -> ConcernAnalysis:
        """Analiza preocupaciones"""
        user_concerns = self.concerns.get(user_id, [])
        
        if not user_concerns:
            return ConcernAnalysis(
                user_id=user_id,
                active_concerns=[],
                improving_concerns=[],
                resolved_concerns=[],
                recommendations=["No hay preocupaciones registradas"]
            )
        
        active = [c for c in user_concerns if c.current_status == "active"]
        improving = [c for c in user_concerns if c.current_status == "improving"]
        resolved = [c for c in user_concerns if c.current_status == "resolved"]
        
        # Preocupación principal (la más severa activa)
        primary_concern = None
        if active:
            severity_order = {"severe": 3, "moderate": 2, "mild": 1}
            primary = max(active, key=lambda c: severity_order.get(c.severity, 0))
            primary_concern = primary.concern_type
        
        # Recomendaciones
        recommendations = []
        
        if len(active) > 3:
            recommendations.append("Tienes múltiples preocupaciones activas. Enfócate en una a la vez")
        
        if primary_concern:
            if primary_concern == "acne":
                recommendations.append("Para acné: usa productos con ácido salicílico o benzoyl peroxide")
            elif primary_concern == "wrinkles":
                recommendations.append("Para arrugas: considera retinol o péptidos")
            elif primary_concern == "dark_spots":
                recommendations.append("Para manchas: usa productos con vitamina C o niacinamida")
            elif primary_concern == "dryness":
                recommendations.append("Para sequedad: hidratación intensa con ceramidas y ácido hialurónico")
        
        if len(resolved) > 0:
            recommendations.append(f"¡Excelente! Has resuelto {len(resolved)} preocupación(es)")
        
        # Resumen de progreso
        progress_summary = {
            "total_concerns": len(user_concerns),
            "active_count": len(active),
            "improving_count": len(improving),
            "resolved_count": len(resolved),
            "resolution_rate": len(resolved) / len(user_concerns) if user_concerns else 0.0
        }
        
        return ConcernAnalysis(
            user_id=user_id,
            active_concerns=active,
            improving_concerns=improving,
            resolved_concerns=resolved,
            primary_concern=primary_concern,
            recommendations=recommendations,
            progress_summary=progress_summary
        )






