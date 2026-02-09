"""
Location Analysis Service - Análisis de ubicación y tráfico
"""

import logging
from typing import Dict, Any, Optional
from ..core.models import StoreType
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class LocationAnalysisService:
    """Servicio para analizar ubicación"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
    
    async def analyze_location(
        self,
        location: str,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Analizar ubicación para el tipo de tienda"""
        
        if self.llm_service.client:
            try:
                return await self._analyze_with_llm(location, store_type)
            except Exception as e:
                logger.error(f"Error analizando ubicación con LLM: {e}")
                return self._generate_default_analysis(location, store_type)
        else:
            return self._generate_default_analysis(location, store_type)
    
    async def _analyze_with_llm(
        self,
        location: str,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Analizar ubicación usando LLM"""
        prompt = f"""Analiza la ubicación "{location}" para una tienda de tipo {store_type.value}.

Proporciona análisis de:
1. Tráfico de peatones y vehículos
2. Visibilidad y accesibilidad
3. Competencia cercana
4. Demografía del área
5. Factores positivos y negativos
6. Recomendaciones específicas

Responde en formato JSON con estas claves:
- traffic_analysis: object con pedestrian_traffic, vehicle_traffic, peak_hours
- visibility: object con score, factors
- accessibility: object con public_transport, parking, walkability
- demographics: object con target_audience_match, income_level, age_groups
- competition: object con nearby_competitors, market_saturation
- pros: array de strings
- cons: array de strings
- recommendations: array de strings
- overall_score: number (1-10)"""
        
        result = await self.llm_service.generate_structured(
            prompt=prompt,
            system_prompt="Eres un experto en análisis de ubicaciones comerciales y bienes raíces."
        )
        
        return result if result else self._generate_default_analysis(location, store_type)
    
    def _generate_default_analysis(
        self,
        location: str,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Generar análisis por defecto"""
        
        return {
            "location": location,
            "traffic_analysis": {
                "pedestrian_traffic": "Moderado a alto (verificar en persona)",
                "vehicle_traffic": "Moderado (verificar en persona)",
                "peak_hours": "Mañana y tarde (verificar localmente)"
            },
            "visibility": {
                "score": 7,
                "factors": [
                    "Verificar visibilidad desde calle principal",
                    "Considerar letrero visible",
                    "Evaluar obstáculos visuales"
                ]
            },
            "accessibility": {
                "public_transport": "Verificar conexión de transporte público",
                "parking": "Verificar disponibilidad de estacionamiento",
                "walkability": "Evaluar facilidad de acceso a pie"
            },
            "demographics": {
                "target_audience_match": "Verificar con datos locales",
                "income_level": "Verificar con datos del censo",
                "age_groups": "Verificar demografía local"
            },
            "competition": {
                "nearby_competitors": "Realizar investigación de campo",
                "market_saturation": "Evaluar competencia directa e indirecta"
            },
            "pros": [
                "Ubicación en área comercial",
                "Potencial de tráfico",
                "Accesibilidad"
            ],
            "cons": [
                "Necesita verificación en persona",
                "Competencia potencial",
                "Costos de alquiler a verificar"
            ],
            "recommendations": [
                "Visitar la ubicación en diferentes horarios",
                "Realizar conteo de tráfico",
                "Hablar con negocios vecinos",
                "Verificar datos demográficos oficiales",
                "Evaluar costos de alquiler vs. potencial"
            ],
            "overall_score": 7
        }




