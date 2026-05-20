"""
Sistema de análisis de fotos con múltiples ángulos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class AngleAnalysis:
    """Análisis de un ángulo específico"""
    angle: str  # "front", "left", "right", "top", "bottom"
    image_url: str
    analysis_data: Dict
    specific_concerns: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "angle": self.angle,
            "image_url": self.image_url,
            "analysis_data": self.analysis_data,
            "specific_concerns": self.specific_concerns,
            "recommendations": self.recommendations
        }


@dataclass
class MultiAngleReport:
    """Reporte de análisis multiángulo"""
    id: str
    user_id: str
    analyses: List[AngleAnalysis]
    overall_assessment: Dict
    comprehensive_recommendations: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "analyses": [a.to_dict() for a in self.analyses],
            "overall_assessment": self.overall_assessment,
            "comprehensive_recommendations": self.comprehensive_recommendations,
            "created_at": self.created_at
        }


class MultiAngleAnalysis:
    """Sistema de análisis multiángulo"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[MultiAngleReport]] = {}  # user_id -> [reports]
    
    def analyze_multiple_angles(self, user_id: str, angle_images: Dict[str, str]) -> MultiAngleReport:
        """Analiza múltiples ángulos"""
        analyses = []
        
        for angle, image_url in angle_images.items():
            # Simulación de análisis por ángulo
            # En producción usaría análisis real de cada imagen
            
            analysis_data = {
                "quality_scores": {
                    "overall_score": 75.0,
                    "texture_score": 70.0,
                    "hydration_score": 65.0
                }
            }
            
            specific_concerns = []
            recommendations = []
            
            if angle == "front":
                specific_concerns.append("Simetría facial")
                recommendations.append("Enfócate en productos para uniformidad")
            elif angle == "left" or angle == "right":
                specific_concerns.append("Perfil facial")
                recommendations.append("Considera productos para contorno")
            elif angle == "top":
                specific_concerns.append("Zona T")
                recommendations.append("Atención especial a frente y nariz")
            
            analysis = AngleAnalysis(
                angle=angle,
                image_url=image_url,
                analysis_data=analysis_data,
                specific_concerns=specific_concerns,
                recommendations=recommendations
            )
            analyses.append(analysis)
        
        # Evaluación general
        overall_scores = [a.analysis_data.get("quality_scores", {}).get("overall_score", 0) for a in analyses]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        overall_assessment = {
            "average_score": avg_score,
            "angles_analyzed": len(analyses),
            "consistency": "high" if max(overall_scores) - min(overall_scores) < 10 else "medium"
        }
        
        # Recomendaciones comprensivas
        comprehensive_recommendations = [
            "Análisis completo realizado desde múltiples ángulos",
            "Considera todos los ángulos al aplicar productos",
            "Monitorea cambios en cada área específica"
        ]
        
        report = MultiAngleReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            analyses=analyses,
            overall_assessment=overall_assessment,
            comprehensive_recommendations=comprehensive_recommendations
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def get_user_reports(self, user_id: str) -> List[MultiAngleReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])






