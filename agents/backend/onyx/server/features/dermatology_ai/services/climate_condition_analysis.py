"""
Sistema de análisis de fotos con diferentes condiciones climáticas
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class ClimateConditionData:
    """Análisis con condición climática específica"""
    condition_type: str  # "hot_dry", "hot_humid", "cold_dry", "cold_humid", "moderate"
    temperature: float
    humidity: float
    uv_index: float
    image_url: str
    analysis_data: Dict
    skin_impact: Dict
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "condition_type": self.condition_type,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "uv_index": self.uv_index,
            "image_url": self.image_url,
            "analysis_data": self.analysis_data,
            "skin_impact": self.skin_impact,
            "recommendations": self.recommendations
        }


@dataclass
class ClimateAnalysisReport:
    """Reporte de análisis climático"""
    id: str
    user_id: str
    analyses: List[ClimateConditionData]
    optimal_condition: str
    seasonal_recommendations: Dict
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
            "optimal_condition": self.optimal_condition,
            "seasonal_recommendations": self.seasonal_recommendations,
            "created_at": self.created_at
        }


class ClimateAnalysisSystem:
    """Sistema de análisis con diferentes condiciones climáticas"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[ClimateAnalysisReport]] = {}  # user_id -> [reports]
    
    def analyze_with_climate(self, user_id: str, climate_data: List[Dict]) -> ClimateAnalysisReport:
        """Analiza con diferentes condiciones climáticas"""
        analyses = []
        
        for climate in climate_data:
            condition_type = climate.get("condition_type", "moderate")
            temperature = climate.get("temperature", 20.0)
            humidity = climate.get("humidity", 50.0)
            uv_index = climate.get("uv_index", 3.0)
            image_url = climate.get("image_url", "")
            
            # Análisis de impacto en la piel
            skin_impact = {
                "hydration_impact": self._calculate_hydration_impact(temperature, humidity),
                "sensitivity_impact": self._calculate_sensitivity_impact(temperature, uv_index),
                "oiliness_impact": self._calculate_oiliness_impact(temperature, humidity)
            }
            
            analysis_data = {
                "quality_scores": {
                    "overall_score": 75.0,
                    "hydration_score": 65.0 + skin_impact["hydration_impact"],
                    "sensitivity_score": 70.0 - skin_impact["sensitivity_impact"]
                }
            }
            
            recommendations = self._generate_climate_recommendations(condition_type, temperature, humidity, uv_index)
            
            analysis = ClimateConditionData(
                condition_type=condition_type,
                temperature=temperature,
                humidity=humidity,
                uv_index=uv_index,
                image_url=image_url,
                analysis_data=analysis_data,
                skin_impact=skin_impact,
                recommendations=recommendations
            )
            analyses.append(analysis)
        
        # Determinar condición óptima
        optimal_condition = "moderate"  # Por defecto
        
        # Recomendaciones estacionales
        seasonal_recommendations = {
            "summer": "Protección solar intensa, hidratación ligera",
            "winter": "Hidratación profunda, protección contra el frío",
            "spring": "Transición de productos, renovación celular",
            "fall": "Preparación para invierno, reparación"
        }
        
        report = ClimateAnalysisReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            analyses=analyses,
            optimal_condition=optimal_condition,
            seasonal_recommendations=seasonal_recommendations
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _calculate_hydration_impact(self, temperature: float, humidity: float) -> float:
        """Calcula impacto en hidratación"""
        if humidity < 30:
            return -10.0  # Deshidratación
        elif humidity > 70:
            return 5.0  # Mejor hidratación
        return 0.0
    
    def _calculate_sensitivity_impact(self, temperature: float, uv_index: float) -> float:
        """Calcula impacto en sensibilidad"""
        impact = 0.0
        if temperature > 30:
            impact += 5.0
        if uv_index > 6:
            impact += 10.0
        return impact
    
    def _calculate_oiliness_impact(self, temperature: float, humidity: float) -> float:
        """Calcula impacto en oleosidad"""
        if temperature > 25 and humidity > 60:
            return 10.0  # Aumenta oleosidad
        return 0.0
    
    def _generate_climate_recommendations(self, condition_type: str, temperature: float,
                                         humidity: float, uv_index: float) -> List[str]:
        """Genera recomendaciones basadas en clima"""
        recommendations = []
        
        if condition_type == "hot_dry":
            recommendations.append("Hidratación intensa necesaria")
            recommendations.append("Protección solar máxima")
        elif condition_type == "hot_humid":
            recommendations.append("Productos ligeros y no comedogénicos")
            recommendations.append("Control de oleosidad")
        elif condition_type == "cold_dry":
            recommendations.append("Hidratación profunda y barrera protectora")
            recommendations.append("Evitar agua muy caliente")
        elif condition_type == "cold_humid":
            recommendations.append("Hidratación moderada")
            recommendations.append("Protección contra viento")
        
        if uv_index > 6:
            recommendations.append("SPF 50+ y reaplicar cada 2 horas")
        
        return recommendations
    
    def get_user_reports(self, user_id: str) -> List[ClimateAnalysisReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])

