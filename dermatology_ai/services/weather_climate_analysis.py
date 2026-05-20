"""
Sistema de análisis de clima y su impacto en la piel
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WeatherData:
    """Datos del clima"""
    location: str
    temperature: float
    humidity: float
    uv_index: float
    wind_speed: float
    air_quality: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "location": self.location,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "uv_index": self.uv_index,
            "wind_speed": self.wind_speed,
            "air_quality": self.air_quality,
            "timestamp": self.timestamp
        }


@dataclass
class ClimateRecommendation:
    """Recomendación basada en clima"""
    location: str
    weather_conditions: WeatherData
    skin_impact: str  # "positive", "negative", "neutral"
    recommendations: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "location": self.location,
            "weather_conditions": self.weather_conditions.to_dict(),
            "skin_impact": self.skin_impact,
            "recommendations": self.recommendations,
            "warnings": self.warnings
        }


class WeatherClimateAnalysis:
    """Sistema de análisis de clima"""
    
    def __init__(self):
        """Inicializa el sistema"""
        pass
    
    def analyze_weather_impact(self, weather_data: WeatherData,
                              user_skin_type: str) -> ClimateRecommendation:
        """Analiza impacto del clima en la piel"""
        recommendations = []
        warnings = []
        impact = "neutral"
        
        # Análisis de temperatura
        if weather_data.temperature < 10:
            recommendations.append("Temperatura baja. Hidrata bien tu piel.")
            impact = "negative"
        elif weather_data.temperature > 30:
            recommendations.append("Temperatura alta. Usa protección solar y mantén hidratación.")
            impact = "negative"
        
        # Análisis de humedad
        if weather_data.humidity < 30:
            recommendations.append("Humedad baja. La piel puede secarse. Usa humectantes intensos.")
            if impact == "neutral":
                impact = "negative"
        elif weather_data.humidity > 70:
            recommendations.append("Humedad alta. Puede aumentar producción de sebo en piel grasa.")
            if user_skin_type == "oily":
                impact = "negative"
        
        # Análisis de UV
        if weather_data.uv_index >= 6:
            warnings.append(f"Índice UV alto ({weather_data.uv_index}). Protección solar esencial.")
            impact = "negative"
        elif weather_data.uv_index >= 3:
            recommendations.append(f"Índice UV moderado ({weather_data.uv_index}). Usa protección solar.")
        
        # Análisis de calidad del aire
        if weather_data.air_quality:
            if weather_data.air_quality > 100:
                warnings.append("Calidad del aire pobre. Contaminantes pueden dañar la piel.")
                recommendations.append("Limpia tu piel más frecuentemente y usa antioxidantes.")
                impact = "negative"
        
        # Recomendaciones específicas por tipo de piel
        if user_skin_type == "dry":
            if weather_data.humidity < 40:
                recommendations.append("Piel seca: Considera usar humidificador.")
        elif user_skin_type == "oily":
            if weather_data.humidity > 60:
                recommendations.append("Piel grasa: Usa productos oil-free en clima húmedo.")
        
        return ClimateRecommendation(
            location=weather_data.location,
            weather_conditions=weather_data,
            skin_impact=impact,
            recommendations=recommendations,
            warnings=warnings
        )
    
    def get_seasonal_recommendations(self, location: str, season: str,
                                    user_skin_type: str) -> List[str]:
        """Obtiene recomendaciones estacionales"""
        recommendations = []
        
        if season.lower() == "winter":
            recommendations.append("Invierno: Clima seco. Hidratación intensa necesaria.")
            recommendations.append("Usa cremas más ricas y evita duchas muy calientes.")
        elif season.lower() == "summer":
            recommendations.append("Verano: Mayor exposición UV. Protección solar diaria.")
            recommendations.append("Hidratación ligera y productos oil-free.")
        elif season.lower() == "spring":
            recommendations.append("Primavera: Transición. Ajusta tu rutina gradualmente.")
        elif season.lower() == "fall":
            recommendations.append("Otoño: Preparación para invierno. Fortalece la barrera cutánea.")
        
        return recommendations






