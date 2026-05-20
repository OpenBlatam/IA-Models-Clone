"""
Sistema de análisis de fotos con diferentes distancias
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class DistanceAnalysis:
    """Análisis con distancia específica"""
    distance_type: str  # "close_up", "medium", "far", "macro"
    distance_cm: Optional[float] = None
    image_url: str = ""
    analysis_quality: str = "good"
    detectable_features: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.detectable_features is None:
            self.detectable_features = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "distance_type": self.distance_type,
            "distance_cm": self.distance_cm,
            "image_url": self.image_url,
            "analysis_quality": self.analysis_quality,
            "detectable_features": self.detectable_features,
            "recommendations": self.recommendations
        }


@dataclass
class DistanceReport:
    """Reporte de análisis de distancia"""
    id: str
    user_id: str
    analyses: List[DistanceAnalysis]
    optimal_distance: str
    distance_comparison: Dict
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
            "optimal_distance": self.optimal_distance,
            "distance_comparison": self.distance_comparison,
            "created_at": self.created_at
        }


class DistanceAnalysisSystem:
    """Sistema de análisis con diferentes distancias"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[DistanceReport]] = {}
    
    def analyze_with_distance(self, user_id: str, distance_data: List[Dict]) -> DistanceReport:
        """Analiza con diferentes distancias"""
        analyses = []
        
        for dist_data in distance_data:
            distance_type = dist_data.get("distance_type", "medium")
            distance_cm = dist_data.get("distance_cm")
            image_url = dist_data.get("image_url", "")
            
            # Calidad según distancia
            analysis_quality = self._evaluate_distance_quality(distance_type)
            
            # Características detectables
            detectable_features = self._get_detectable_features(distance_type)
            
            # Recomendaciones
            recommendations = self._generate_distance_recommendations(distance_type)
            
            analysis = DistanceAnalysis(
                distance_type=distance_type,
                distance_cm=distance_cm,
                image_url=image_url,
                analysis_quality=analysis_quality,
                detectable_features=detectable_features,
                recommendations=recommendations
            )
            analyses.append(analysis)
        
        # Determinar distancia óptima
        optimal_distance = "close_up"  # Por defecto
        
        # Comparación de distancias
        distance_comparison = {
            "best_for_detail": "macro",
            "best_for_overall": "close_up",
            "note": "Diferentes distancias revelan diferentes características"
        }
        
        report = DistanceReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            analyses=analyses,
            optimal_distance=optimal_distance,
            distance_comparison=distance_comparison
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _evaluate_distance_quality(self, distance_type: str) -> str:
        """Evalúa calidad según distancia"""
        quality_map = {
            "macro": "excellent",
            "close_up": "excellent",
            "medium": "good",
            "far": "fair"
        }
        return quality_map.get(distance_type, "good")
    
    def _get_detectable_features(self, distance_type: str) -> List[str]:
        """Obtiene características detectables"""
        features = {
            "macro": ["Poros individuales", "Textura microscópica", "Estructura celular", "Micro-arrugas"],
            "close_up": ["Poros", "Textura detallada", "Arrugas", "Manchas", "Imperfecciones"],
            "medium": ["Tono general", "Textura básica", "Manchas grandes", "Arrugas visibles"],
            "far": ["Tono general", "Simetría facial", "Contorno general"]
        }
        return features.get(distance_type, features["medium"])
    
    def _generate_distance_recommendations(self, distance_type: str) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        if distance_type == "far":
            recommendations.append("Usa distancia más cercana para análisis detallado")
        
        if distance_type == "macro":
            recommendations.append("Excelente para análisis de textura y poros")
        
        if distance_type == "close_up":
            recommendations.append("Distancia ideal para análisis general detallado")
        
        return recommendations


