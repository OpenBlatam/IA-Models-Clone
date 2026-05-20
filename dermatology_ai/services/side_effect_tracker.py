"""
Sistema de seguimiento de efectos secundarios
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class SideEffectRecord:
    """Registro de efecto secundario"""
    id: str
    user_id: str
    record_date: str
    product_name: str
    product_category: str
    side_effect_type: str  # "irritation", "redness", "dryness", "breakout", "allergic_reaction", "other"
    severity: str  # "mild", "moderate", "severe"
    location: Optional[str] = None
    duration_hours: Optional[int] = None
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
            "record_date": self.record_date,
            "product_name": self.product_name,
            "product_category": self.product_category,
            "side_effect_type": self.side_effect_type,
            "severity": self.severity,
            "location": self.location,
            "duration_hours": self.duration_hours,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class SideEffectAnalysis:
    """Análisis de efectos secundarios"""
    user_id: str
    total_records: int
    most_common_type: Optional[str] = None
    most_problematic_product: Optional[str] = None
    severity_distribution: Dict[str, int] = None
    side_effects_by_category: Dict[str, int] = None
    recommendations: List[str] = None
    days_analyzed: int = 0
    
    def __post_init__(self):
        if self.severity_distribution is None:
            self.severity_distribution = {}
        if self.side_effects_by_category is None:
            self.side_effects_by_category = {}
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "total_records": self.total_records,
            "most_common_type": self.most_common_type,
            "most_problematic_product": self.most_problematic_product,
            "severity_distribution": self.severity_distribution,
            "side_effects_by_category": self.side_effects_by_category,
            "recommendations": self.recommendations,
            "days_analyzed": self.days_analyzed
        }


class SideEffectTracker:
    """Sistema de seguimiento de efectos secundarios"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.records: Dict[str, List[SideEffectRecord]] = {}
    
    def add_side_effect(self, user_id: str, record_date: str, product_name: str,
                       product_category: str, side_effect_type: str, severity: str,
                       location: Optional[str] = None, duration_hours: Optional[int] = None,
                       notes: Optional[str] = None) -> SideEffectRecord:
        """Agrega registro de efecto secundario"""
        record = SideEffectRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            record_date=record_date,
            product_name=product_name,
            product_category=product_category,
            side_effect_type=side_effect_type,
            severity=severity,
            location=location,
            duration_hours=duration_hours,
            notes=notes
        )
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        self.records[user_id].append(record)
        return record
    
    def analyze_side_effects(self, user_id: str, days: int = 90) -> SideEffectAnalysis:
        """Analiza efectos secundarios"""
        user_records = self.records.get(user_id, [])
        
        if not user_records:
            return SideEffectAnalysis(
                user_id=user_id,
                total_records=0,
                recommendations=["No hay registros de efectos secundarios"]
            )
        
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_records = [
            r for r in user_records
            if datetime.fromisoformat(r.record_date).date() >= cutoff
        ]
        
        if not recent_records:
            return SideEffectAnalysis(
                user_id=user_id,
                total_records=0,
                recommendations=["No hay registros recientes"]
            )
        
        # Análisis de tipos
        type_counts = {}
        severity_counts = {}
        category_counts = {}
        product_counts = {}
        
        for record in recent_records:
            type_counts[record.side_effect_type] = type_counts.get(record.side_effect_type, 0) + 1
            severity_counts[record.severity] = severity_counts.get(record.severity, 0) + 1
            category_counts[record.product_category] = category_counts.get(record.product_category, 0) + 1
            product_counts[record.product_name] = product_counts.get(record.product_name, 0) + 1
        
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        most_problematic_product = max(product_counts.items(), key=lambda x: x[1])[0] if product_counts else None
        
        # Recomendaciones
        recommendations = []
        
        if most_common_type:
            recommendations.append(f"Efecto secundario más común: {most_common_type}")
            
            if most_common_type == "irritation":
                recommendations.append("Considera productos más suaves y sin fragancia")
            elif most_common_type == "dryness":
                recommendations.append("Aumenta hidratación y reduce frecuencia de uso")
            elif most_common_type == "breakout":
                recommendations.append("Producto puede ser comedogénico. Considera alternativas")
            elif most_common_type == "allergic_reaction":
                recommendations.append("ADVERTENCIA: Consulta con dermatólogo. Puede ser alergia")
        
        if most_problematic_product:
            recommendations.append(f"Producto más problemático: {most_problematic_product}")
            recommendations.append("Considera descontinuar o reducir frecuencia de uso")
        
        severe_count = severity_counts.get("severe", 0)
        if severe_count > 0:
            recommendations.append("ADVERTENCIA: Efectos secundarios severos detectados. Consulta con médico")
        
        if not recommendations:
            recommendations.append("No se detectaron patrones problemáticos")
        
        return SideEffectAnalysis(
            user_id=user_id,
            total_records=len(recent_records),
            most_common_type=most_common_type,
            most_problematic_product=most_problematic_product,
            severity_distribution=severity_counts,
            side_effects_by_category=category_counts,
            recommendations=recommendations,
            days_analyzed=days
        )
