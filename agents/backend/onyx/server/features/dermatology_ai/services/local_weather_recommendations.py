"""
Sistema de recomendaciones basadas en clima local
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WeatherProfile:
    """Perfil de clima local"""
    user_id: str
    location: str
    climate_type: str  # "tropical", "temperate", "arid", "continental", "polar"
    average_humidity: Optional[float] = None
    average_temperature: Optional[float] = None
    uv_index_range: Optional[str] = None  # "low", "medium", "high", "very_high"
    seasonal_variations: bool = True
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "location": self.location,
            "climate_type": self.climate_type,
            "average_humidity": self.average_humidity,
            "average_temperature": self.average_temperature,
            "uv_index_range": self.uv_index_range,
            "seasonal_variations": self.seasonal_variations
        }


@dataclass
class WeatherRecommendation:
    """Recomendación basada en clima"""
    climate_factor: str
    impact: str
    recommendations: List[str]
    product_suggestions: List[Dict]
    priority: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "climate_factor": self.climate_factor,
            "impact": self.impact,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "priority": self.priority
        }


class LocalWeatherRecommendations:
    """Sistema de recomendaciones basadas en clima local"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, WeatherProfile] = {}
    
    def create_weather_profile(self, user_id: str, location: str, climate_type: str,
                              average_humidity: Optional[float] = None,
                              average_temperature: Optional[float] = None,
                              uv_index_range: Optional[str] = None,
                              seasonal_variations: bool = True) -> WeatherProfile:
        """Crea perfil de clima"""
        profile = WeatherProfile(
            user_id=user_id,
            location=location,
            climate_type=climate_type,
            average_humidity=average_humidity,
            average_temperature=average_temperature,
            uv_index_range=uv_index_range,
            seasonal_variations=seasonal_variations
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def get_weather_recommendations(self, user_id: str) -> List[WeatherRecommendation]:
        """Obtiene recomendaciones basadas en clima"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            return []
        
        recommendations = []
        
        # Recomendaciones para clima tropical
        if profile.climate_type == "tropical":
            recommendations.append(WeatherRecommendation(
                climate_factor="Clima Tropical",
                impact="high",
                recommendations=[
                    "Alta humedad puede aumentar producción de sebo",
                    "Protección solar extrema necesaria",
                    "Productos ligeros y no comedogénicos",
                    "Limpieza frecuente importante"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "spf_50_water_resistant", "priority": 1},
                    {"category": "cleanser", "type": "oil_free", "reason": "Control de oleosidad"}
                ],
                priority=1
            ))
        
        # Recomendaciones para clima árido
        if profile.climate_type == "arid":
            recommendations.append(WeatherRecommendation(
                climate_factor="Clima Árido",
                impact="high",
                recommendations=[
                    "Baja humedad deshidrata la piel",
                    "Hidratación intensa necesaria",
                    "Protección contra viento y polvo",
                    "Productos oclusivos recomendados"
                ],
                product_suggestions=[
                    {"category": "moisturizer", "type": "rich_occlusive", "priority": 1},
                    {"category": "serum", "type": "hyaluronic_acid", "reason": "Hidratación profunda"}
                ],
                priority=1
            ))
        
        # Recomendaciones para clima templado
        if profile.climate_type == "temperate":
            recommendations.append(WeatherRecommendation(
                climate_factor="Clima Templado",
                impact="medium",
                recommendations=[
                    "Condiciones generalmente favorables",
                    "Ajusta rutina según estación",
                    "Protección solar moderada",
                    "Hidratación balanceada"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "spf_30", "priority": 2}
                ],
                priority=2
            ))
        
        # Recomendaciones basadas en humedad
        if profile.average_humidity is not None:
            if profile.average_humidity < 30:
                recommendations.append(WeatherRecommendation(
                    climate_factor="Baja Humedad",
                    impact="high",
                    recommendations=[
                        "Humedad baja deshidrata la piel",
                        "Usa humectantes oclusivos",
                        "Considera humidificador",
                        "Evita productos astringentes"
                    ],
                    product_suggestions=[
                        {"category": "moisturizer", "type": "barrier_repair", "priority": 1}
                    ],
                    priority=1
                ))
            elif profile.average_humidity > 70:
                recommendations.append(WeatherRecommendation(
                    climate_factor="Alta Humedad",
                    impact="medium",
                    recommendations=[
                        "Alta humedad puede aumentar oleosidad",
                        "Productos ligeros recomendados",
                        "Limpieza regular importante"
                    ],
                    product_suggestions=[
                        {"category": "cleanser", "type": "gentle_cleansing", "priority": 2}
                    ],
                    priority=2
                ))
        
        # Recomendaciones basadas en UV
        if profile.uv_index_range in ["high", "very_high"]:
            recommendations.append(WeatherRecommendation(
                climate_factor="Alto Índice UV",
                impact="high",
                recommendations=[
                    "Protección solar extrema necesaria",
                    "Reaplica SPF cada 2 horas",
                    "Evita exposición solar directa en horas pico",
                    "Productos con antioxidantes para protección adicional"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "spf_50_broad_spectrum", "priority": 1},
                    {"category": "serum", "type": "antioxidant", "reason": "Protección adicional"}
                ],
                priority=1
            ))
        
        # Recomendaciones basadas en temperatura
        if profile.average_temperature is not None:
            if profile.average_temperature > 30:
                recommendations.append(WeatherRecommendation(
                    climate_factor="Temperatura Alta",
                    impact="medium",
                    recommendations=[
                        "Calor puede aumentar producción de sebo",
                        "Productos refrescantes recomendados",
                        "Hidratación ligera",
                        "Limpieza frecuente"
                    ],
                    product_suggestions=[
                        {"category": "moisturizer", "type": "lightweight_gel", "priority": 2}
                    ],
                    priority=2
                ))
            elif profile.average_temperature < 5:
                recommendations.append(WeatherRecommendation(
                    climate_factor="Temperatura Baja",
                    impact="high",
                    recommendations=[
                        "Frío puede deshidratar y irritar",
                        "Protección de barrera importante",
                        "Productos ricos y reparadores",
                        "Evita limpieza excesiva"
                    ],
                    product_suggestions=[
                        {"category": "moisturizer", "type": "rich_repair", "priority": 1},
                        {"category": "cleanser", "type": "gentle_creamy", "reason": "No deshidratar"}
                    ],
                    priority=1
                ))
        
        recommendations.sort(key=lambda x: x.priority)
        return recommendations


