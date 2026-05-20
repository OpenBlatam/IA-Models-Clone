"""
Sistema de seguimiento de estrés
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class StressRecord:
    """Registro de estrés"""
    id: str
    user_id: str
    stress_date: str
    stress_level: int  # 1-10 scale
    stress_source: Optional[str] = None  # "work", "personal", "health", "financial", "other"
    physical_symptoms: List[str] = None
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.physical_symptoms is None:
            self.physical_symptoms = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "stress_date": self.stress_date,
            "stress_level": self.stress_level,
            "stress_source": self.stress_source,
            "physical_symptoms": self.physical_symptoms,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class StressAnalysis:
    """Análisis de estrés"""
    user_id: str
    average_stress_level: float
    stress_trend: str  # "increasing", "decreasing", "stable"
    primary_stress_sources: List[str]
    skin_impact: Dict
    recommendations: List[str]
    days_analyzed: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "average_stress_level": self.average_stress_level,
            "stress_trend": self.stress_trend,
            "primary_stress_sources": self.primary_stress_sources,
            "skin_impact": self.skin_impact,
            "recommendations": self.recommendations,
            "days_analyzed": self.days_analyzed
        }


class StressTracker:
    """Sistema de seguimiento de estrés"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.records: Dict[str, List[StressRecord]] = {}  # user_id -> [records]
    
    def add_stress_record(self, user_id: str, stress_date: str, stress_level: int,
                         stress_source: Optional[str] = None,
                         physical_symptoms: Optional[List[str]] = None,
                         notes: Optional[str] = None) -> StressRecord:
        """Agrega registro de estrés"""
        record = StressRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            stress_date=stress_date,
            stress_level=stress_level,
            stress_source=stress_source,
            physical_symptoms=physical_symptoms or [],
            notes=notes
        )
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        self.records[user_id].append(record)
        return record
    
    def analyze_stress(self, user_id: str, days: int = 30) -> StressAnalysis:
        """Analiza estrés"""
        user_records = self.records.get(user_id, [])
        
        if not user_records:
            return StressAnalysis(
                user_id=user_id,
                average_stress_level=0.0,
                stress_trend="unknown",
                primary_stress_sources=[],
                skin_impact={},
                recommendations=["Agrega registros de estrés para análisis"],
                days_analyzed=0
            )
        
        # Filtrar por días
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_records = [
            r for r in user_records
            if datetime.fromisoformat(r.stress_date).date() >= cutoff
        ]
        
        if not recent_records:
            return StressAnalysis(
                user_id=user_id,
                average_stress_level=0.0,
                stress_trend="unknown",
                primary_stress_sources=[],
                skin_impact={},
                recommendations=["No hay registros recientes"],
                days_analyzed=0
            )
        
        # Calcular promedio
        avg_stress = statistics.mean([r.stress_level for r in recent_records])
        
        # Determinar tendencia
        if len(recent_records) >= 2:
            first_half = recent_records[:len(recent_records)//2]
            second_half = recent_records[len(recent_records)//2:]
            
            first_avg = statistics.mean([r.stress_level for r in first_half])
            second_avg = statistics.mean([r.stress_level for r in second_half])
            
            if second_avg > first_avg + 1:
                trend = "increasing"
            elif second_avg < first_avg - 1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Fuentes principales de estrés
        source_counts = {}
        for record in recent_records:
            if record.stress_source:
                source_counts[record.stress_source] = source_counts.get(record.stress_source, 0) + 1
        
        primary_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        primary_stress_sources = [source for source, _ in primary_sources]
        
        # Impacto en la piel
        skin_impact = {
            "breakout_risk": "high" if avg_stress > 7 else "medium" if avg_stress > 5 else "low",
            "sensitivity_risk": "high" if avg_stress > 7 else "medium" if avg_stress > 5 else "low",
            "healing_speed": "slow" if avg_stress > 7 else "normal" if avg_stress > 5 else "optimal"
        }
        
        # Recomendaciones
        recommendations = []
        
        if avg_stress > 7:
            recommendations.append("Nivel de estrés alto detectado. Prioriza técnicas de relajación")
            recommendations.append("El estrés alto puede causar brotes y sensibilidad")
            recommendations.append("Considera productos calmantes y antiinflamatorios")
        elif avg_stress > 5:
            recommendations.append("Nivel de estrés moderado. Practica manejo de estrés")
            recommendations.append("Considera productos con ingredientes calmantes")
        
        if trend == "increasing":
            recommendations.append("El estrés está aumentando. Busca apoyo profesional si es necesario")
        
        if "work" in primary_stress_sources:
            recommendations.append("Estrés relacionado con trabajo. Considera límites saludables")
        
        return StressAnalysis(
            user_id=user_id,
            average_stress_level=avg_stress,
            stress_trend=trend,
            primary_stress_sources=primary_stress_sources,
            skin_impact=skin_impact,
            recommendations=recommendations,
            days_analyzed=len(recent_records)
        )






