"""
Sistema de análisis de múltiples condiciones
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class ConditionAnalysis:
    """Análisis de condición"""
    condition_name: str
    severity: str  # "mild", "moderate", "severe"
    confidence: float
    affected_areas: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "condition_name": self.condition_name,
            "severity": self.severity,
            "confidence": self.confidence,
            "affected_areas": self.affected_areas,
            "recommendations": self.recommendations
        }


@dataclass
class MultiConditionReport:
    """Reporte de múltiples condiciones"""
    id: str
    user_id: str
    image_url: str
    conditions: List[ConditionAnalysis]
    primary_concern: str
    overall_severity: str
    treatment_priority: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_url": self.image_url,
            "conditions": [c.to_dict() for c in self.conditions],
            "primary_concern": self.primary_concern,
            "overall_severity": self.overall_severity,
            "treatment_priority": self.treatment_priority,
            "created_at": self.created_at
        }


class MultiConditionAnalyzer:
    """Sistema de análisis de múltiples condiciones"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.reports: Dict[str, List[MultiConditionReport]] = {}  # user_id -> [reports]
    
    def analyze_conditions(self, user_id: str, image_url: str,
                          analysis_data: Dict) -> MultiConditionReport:
        """Analiza múltiples condiciones"""
        conditions = []
        
        # Analizar diferentes condiciones basadas en scores
        scores = analysis_data.get("quality_scores", {})
        detected_conditions = analysis_data.get("conditions", [])
        
        # Análisis de acné
        sebum_level = scores.get("sebum_level", 0)
        if sebum_level > 0.7:
            conditions.append(ConditionAnalysis(
                condition_name="Acné",
                severity="moderate" if sebum_level > 0.8 else "mild",
                confidence=0.75,
                affected_areas=["zona T", "mejillas"],
                recommendations=["Usa productos con ácido salicílico", "Limpieza suave dos veces al día"]
            ))
        
        # Análisis de deshidratación
        hydration = scores.get("hydration_score", 0)
        if hydration < 40:
            conditions.append(ConditionAnalysis(
                condition_name="Deshidratación",
                severity="severe" if hydration < 30 else "moderate",
                confidence=0.85,
                affected_areas=["toda la cara"],
                recommendations=["Hidratación intensa", "Productos con ácido hialurónico", "Evita productos astringentes"]
            ))
        
        # Análisis de hiperpigmentación
        if detected_conditions:
            for cond in detected_conditions:
                if "spot" in cond.lower() or "pigment" in cond.lower():
                    conditions.append(ConditionAnalysis(
                        condition_name="Hiperpigmentación",
                        severity="moderate",
                        confidence=0.70,
                        affected_areas=["mejillas", "frente"],
                        recommendations=["Productos con vitamina C", "Protección solar diaria", "Considera tratamiento profesional"]
                    ))
        
        # Determinar preocupación principal
        if conditions:
            primary_concern = max(conditions, key=lambda c: 3 if c.severity == "severe" else 2 if c.severity == "moderate" else 1).condition_name
        else:
            primary_concern = "Ninguna condición detectada"
        
        # Severidad general
        severe_count = sum(1 for c in conditions if c.severity == "severe")
        if severe_count > 0:
            overall_severity = "severe"
        elif len(conditions) > 2:
            overall_severity = "moderate"
        elif len(conditions) > 0:
            overall_severity = "mild"
        else:
            overall_severity = "none"
        
        # Prioridad de tratamiento
        treatment_priority = []
        for condition in sorted(conditions, key=lambda c: 3 if c.severity == "severe" else 2 if c.severity == "moderate" else 1, reverse=True):
            treatment_priority.append(condition.condition_name)
        
        report = MultiConditionReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            image_url=image_url,
            conditions=conditions,
            primary_concern=primary_concern,
            overall_severity=overall_severity,
            treatment_priority=treatment_priority
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def get_user_reports(self, user_id: str) -> List[MultiConditionReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])






