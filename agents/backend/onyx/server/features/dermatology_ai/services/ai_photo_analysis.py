"""
Sistema de análisis avanzado de fotos con IA
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class AIPhotoAnalysis:
    """Análisis de foto con IA"""
    id: str
    user_id: str
    image_url: str
    analysis_type: str  # "skin_condition", "wrinkles", "pores", "spots", "texture"
    confidence_score: float
    detected_features: List[Dict]
    recommendations: List[str]
    severity_level: str  # "low", "medium", "high"
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
            "analysis_type": self.analysis_type,
            "confidence_score": self.confidence_score,
            "detected_features": self.detected_features,
            "recommendations": self.recommendations,
            "severity_level": self.severity_level,
            "created_at": self.created_at
        }


class AIPhotoAnalysisSystem:
    """Sistema de análisis avanzado de fotos con IA"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.analyses: Dict[str, List[AIPhotoAnalysis]] = {}  # user_id -> [analyses]
    
    def analyze_photo(self, user_id: str, image_url: str,
                     analysis_type: str) -> AIPhotoAnalysis:
        """Analiza foto con IA"""
        # Simulación de análisis con IA
        # En producción, esto usaría modelos de ML reales
        
        detected_features = []
        recommendations = []
        confidence_score = 0.85
        severity_level = "medium"
        
        if analysis_type == "skin_condition":
            detected_features = [
                {"feature": "acne", "count": 3, "severity": "mild"},
                {"feature": "dark_spots", "count": 5, "severity": "moderate"}
            ]
            recommendations = [
                "Usa productos con ácido salicílico para acné",
                "Considera tratamiento para manchas oscuras"
            ]
            severity_level = "medium"
        
        elif analysis_type == "wrinkles":
            detected_features = [
                {"feature": "fine_lines", "count": 8, "location": "forehead"},
                {"feature": "deep_wrinkles", "count": 2, "location": "around_eyes"}
            ]
            recommendations = [
                "Usa productos con retinol",
                "Considera cremas anti-envejecimiento"
            ]
            severity_level = "low"
        
        elif analysis_type == "pores":
            detected_features = [
                {"feature": "large_pores", "count": 12, "location": "nose"},
                {"feature": "clogged_pores", "count": 5, "location": "cheeks"}
            ]
            recommendations = [
                "Usa exfoliante químico",
                "Considera limpieza profunda"
            ]
            severity_level = "medium"
        
        elif analysis_type == "texture":
            detected_features = [
                {"feature": "roughness", "level": 0.6},
                {"feature": "smoothness", "level": 0.4}
            ]
            recommendations = [
                "Usa productos exfoliantes",
                "Hidrata regularmente"
            ]
            severity_level = "medium"
        
        analysis = AIPhotoAnalysis(
            id=str(uuid.uuid4()),
            user_id=user_id,
            image_url=image_url,
            analysis_type=analysis_type,
            confidence_score=confidence_score,
            detected_features=detected_features,
            recommendations=recommendations,
            severity_level=severity_level
        )
        
        if user_id not in self.analyses:
            self.analyses[user_id] = []
        
        self.analyses[user_id].append(analysis)
        return analysis
    
    def get_user_analyses(self, user_id: str, analysis_type: Optional[str] = None) -> List[AIPhotoAnalysis]:
        """Obtiene análisis del usuario"""
        user_analyses = self.analyses.get(user_id, [])
        
        if analysis_type:
            user_analyses = [a for a in user_analyses if a.analysis_type == analysis_type]
        
        user_analyses.sort(key=lambda x: x.created_at, reverse=True)
        return user_analyses
    
    def compare_analyses(self, analysis1_id: str, analysis2_id: str,
                        user_id: str) -> Dict:
        """Compara dos análisis"""
        user_analyses = self.analyses.get(user_id, [])
        
        analysis1 = next((a for a in user_analyses if a.id == analysis1_id), None)
        analysis2 = next((a for a in user_analyses if a.id == analysis2_id), None)
        
        if not analysis1 or not analysis2:
            raise ValueError("One or both analyses not found")
        
        # Comparar características
        improvements = []
        deteriorations = []
        
        # Lógica simplificada de comparación
        if analysis1.severity_level != analysis2.severity_level:
            if self._severity_to_num(analysis2.severity_level) < self._severity_to_num(analysis1.severity_level):
                improvements.append("Mejora general en condición de piel")
            else:
                deteriorations.append("Empeoramiento en condición de piel")
        
        return {
            "analysis1": analysis1.to_dict(),
            "analysis2": analysis2.to_dict(),
            "improvements": improvements,
            "deteriorations": deteriorations,
            "overall_change": "improved" if improvements else "deteriorated" if deteriorations else "stable"
        }
    
    def _severity_to_num(self, severity: str) -> int:
        """Convierte severidad a número"""
        mapping = {"low": 1, "medium": 2, "high": 3}
        return mapping.get(severity, 2)






