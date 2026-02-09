"""
Sistema de análisis de fotos con diferentes resoluciones
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class ResolutionData:
    """Análisis con resolución específica"""
    resolution: str  # "low", "medium", "high", "ultra_high"
    width: int
    height: int
    image_url: str
    analysis_quality: str  # "poor", "fair", "good", "excellent"
    detectable_features: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "image_url": self.image_url,
            "analysis_quality": self.analysis_quality,
            "detectable_features": self.detectable_features,
            "recommendations": self.recommendations
        }


@dataclass
class ResolutionReport:
    """Reporte de análisis de resolución"""
    id: str
    user_id: str
    analyses: List[ResolutionData]
    optimal_resolution: str
    quality_comparison: Dict
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
            "optimal_resolution": self.optimal_resolution,
            "quality_comparison": self.quality_comparison,
            "created_at": self.created_at
        }


class ResolutionAnalysisSystem:
    """Sistema de análisis con diferentes resoluciones"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[ResolutionReport]] = {}  # user_id -> [reports]
    
    def analyze_with_resolution(self, user_id: str, resolution_data: List[Dict]) -> ResolutionReport:
        """Analiza con diferentes resoluciones"""
        analyses = []
        
        for res_data in resolution_data:
            width = res_data.get("width", 640)
            height = res_data.get("height", 480)
            image_url = res_data.get("image_url", "")
            
            # Determinar resolución
            total_pixels = width * height
            if total_pixels < 300000:
                resolution = "low"
                analysis_quality = "poor"
            elif total_pixels < 1000000:
                resolution = "medium"
                analysis_quality = "fair"
            elif total_pixels < 5000000:
                resolution = "high"
                analysis_quality = "good"
            else:
                resolution = "ultra_high"
                analysis_quality = "excellent"
            
            # Características detectables según resolución
            detectable_features = self._get_detectable_features(resolution)
            
            # Recomendaciones
            recommendations = self._generate_resolution_recommendations(resolution, analysis_quality)
            
            analysis = ResolutionData(
                resolution=resolution,
                width=width,
                height=height,
                image_url=image_url,
                analysis_quality=analysis_quality,
                detectable_features=detectable_features,
                recommendations=recommendations
            )
            analyses.append(analysis)
        
        # Determinar resolución óptima
        optimal_resolution = "high"  # Por defecto
        best_analysis = max(analyses, key=lambda a: self._get_quality_score(a.analysis_quality))
        if best_analysis:
            optimal_resolution = best_analysis.resolution
        
        # Comparación de calidad
        quality_comparison = {
            "best_quality": best_analysis.analysis_quality if best_analysis else "fair",
            "recommended_minimum": "high",
            "note": "Resoluciones más altas permiten análisis más detallados"
        }
        
        report = ResolutionReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            analyses=analyses,
            optimal_resolution=optimal_resolution,
            quality_comparison=quality_comparison
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _get_detectable_features(self, resolution: str) -> List[str]:
        """Obtiene características detectables según resolución"""
        features = {
            "low": ["Tono general", "Textura básica"],
            "medium": ["Tono general", "Textura básica", "Poros grandes"],
            "high": ["Tono general", "Textura detallada", "Poros", "Arrugas visibles", "Manchas"],
            "ultra_high": ["Tono general", "Textura detallada", "Poros", "Arrugas finas", "Manchas", "Micro-textura"]
        }
        return features.get(resolution, features["medium"])
    
    def _generate_resolution_recommendations(self, resolution: str, quality: str) -> List[str]:
        """Genera recomendaciones según resolución"""
        recommendations = []
        
        if quality in ["poor", "fair"]:
            recommendations.append("Usa una resolución más alta para mejor análisis")
            recommendations.append("Mínimo recomendado: 1920x1080 (Full HD)")
        
        if resolution == "ultra_high":
            recommendations.append("Excelente resolución para análisis detallado")
        
        return recommendations
    
    def _get_quality_score(self, quality: str) -> int:
        """Obtiene score numérico de calidad"""
        scores = {"poor": 1, "fair": 2, "good": 3, "excellent": 4}
        return scores.get(quality, 2)
    
    def get_user_reports(self, user_id: str) -> List[ResolutionReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])

