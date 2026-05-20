"""
Sistema de análisis de fotos con diferentes condiciones de iluminación natural
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class NaturalLightingData:
    """Datos de iluminación natural"""
    time_of_day: str  # "morning", "midday", "afternoon", "evening", "golden_hour"
    weather_condition: str  # "sunny", "cloudy", "overcast", "shade"
    location: Optional[str] = None
    image_url: str = ""
    analysis_quality: str = "good"
    visible_features: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.visible_features is None:
            self.visible_features = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "time_of_day": self.time_of_day,
            "weather_condition": self.weather_condition,
            "location": self.location,
            "image_url": self.image_url,
            "analysis_quality": self.analysis_quality,
            "visible_features": self.visible_features,
            "recommendations": self.recommendations
        }


@dataclass
class NaturalLightingReport:
    """Reporte de iluminación natural"""
    id: str
    user_id: str
    analyses: List[NaturalLightingData]
    optimal_conditions: Dict
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
            "optimal_conditions": self.optimal_conditions,
            "created_at": self.created_at
        }


class NaturalLightingAnalysis:
    """Sistema de análisis con iluminación natural"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[NaturalLightingReport]] = {}
    
    def analyze_with_natural_lighting(self, user_id: str, lighting_data: List[Dict]) -> NaturalLightingReport:
        """Analiza con diferentes condiciones de iluminación natural"""
        analyses = []
        
        for light_data in lighting_data:
            time_of_day = light_data.get("time_of_day", "midday")
            weather = light_data.get("weather_condition", "sunny")
            location = light_data.get("location")
            image_url = light_data.get("image_url", "")
            
            # Calidad según condiciones
            analysis_quality = self._evaluate_lighting_quality(time_of_day, weather)
            
            # Características visibles
            visible_features = self._get_visible_features(time_of_day, weather)
            
            # Recomendaciones
            recommendations = self._generate_lighting_recommendations(time_of_day, weather)
            
            analysis = NaturalLightingData(
                time_of_day=time_of_day,
                weather_condition=weather,
                location=location,
                image_url=image_url,
                analysis_quality=analysis_quality,
                visible_features=visible_features,
                recommendations=recommendations
            )
            analyses.append(analysis)
        
        # Condiciones óptimas
        optimal_conditions = {
            "best_time": "golden_hour",
            "best_weather": "cloudy",
            "note": "Luz difusa y suave es ideal para análisis preciso"
        }
        
        report = NaturalLightingReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            analyses=analyses,
            optimal_conditions=optimal_conditions
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _evaluate_lighting_quality(self, time_of_day: str, weather: str) -> str:
        """Evalúa calidad de iluminación"""
        if weather == "cloudy" and time_of_day in ["morning", "afternoon"]:
            return "excellent"
        elif weather == "overcast":
            return "excellent"
        elif time_of_day == "golden_hour":
            return "good"
        elif weather == "sunny" and time_of_day == "midday":
            return "fair"
        else:
            return "good"
    
    def _get_visible_features(self, time_of_day: str, weather: str) -> List[str]:
        """Obtiene características visibles"""
        features = []
        
        if weather == "cloudy" or weather == "overcast":
            features.extend(["Tono uniforme", "Textura detallada", "Poros", "Manchas"])
        elif time_of_day == "golden_hour":
            features.extend(["Tono cálido", "Textura general"])
        else:
            features.extend(["Tono general", "Textura básica"])
        
        return features
    
    def _generate_lighting_recommendations(self, time_of_day: str, weather: str) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        if weather == "sunny" and time_of_day == "midday":
            recommendations.append("Luz solar directa puede crear sombras. Usa sombra o luz difusa")
        
        if weather == "cloudy":
            recommendations.append("Excelente condición para análisis - luz difusa y uniforme")
        
        if time_of_day == "golden_hour":
            recommendations.append("Luz cálida puede afectar tono. Mejor usar luz neutra")
        
        return recommendations


