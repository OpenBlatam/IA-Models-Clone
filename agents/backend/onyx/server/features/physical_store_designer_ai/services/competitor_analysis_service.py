"""
Competitor Analysis Service - Análisis de competencia
"""

import logging
from typing import List, Dict, Any, Optional
from ..core.models import StoreType
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class CompetitorAnalysisService:
    """Servicio para analizar competencia"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
    
    async def analyze_competitors(
        self,
        store_type: StoreType,
        location: Optional[str] = None,
        store_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analizar competencia en el área"""
        
        if self.llm_service.client:
            try:
                return await self._analyze_with_llm(store_type, location, store_name)
            except Exception as e:
                logger.error(f"Error analizando competencia con LLM: {e}")
                return self._generate_default_analysis(store_type, location)
        else:
            return self._generate_default_analysis(store_type, location)
    
    async def _analyze_with_llm(
        self,
        store_type: StoreType,
        location: Optional[str],
        store_name: Optional[str]
    ) -> Dict[str, Any]:
        """Analizar competencia usando LLM"""
        prompt = f"""Analiza la competencia para una tienda de tipo {store_type.value} en {location or "un área comercial"}.

Proporciona:
1. Principales competidores en el área
2. Fortalezas y debilidades de la competencia
3. Oportunidades de diferenciación
4. Amenazas del mercado
5. Recomendaciones estratégicas

Responde en formato JSON con estas claves:
- competitors: array de objetos con name, strengths, weaknesses
- opportunities: array de strings
- threats: array de strings
- recommendations: array de strings
- market_positioning: string"""
        
        result = await self.llm_service.generate_structured(
            prompt=prompt,
            system_prompt="Eres un experto en análisis de mercado y competencia para negocios físicos."
        )
        
        return result if result else self._generate_default_analysis(store_type, location)
    
    def _generate_default_analysis(
        self,
        store_type: StoreType,
        location: Optional[str]
    ) -> Dict[str, Any]:
        """Generar análisis por defecto"""
        
        analysis_templates = {
            StoreType.RESTAURANT: {
                "competitors": [
                    {
                        "name": "Restaurantes locales establecidos",
                        "strengths": ["Cliente fiel", "Ubicación conocida", "Menú probado"],
                        "weaknesses": ["Menú estático", "Precios altos", "Servicio lento"]
                    }
                ],
                "opportunities": [
                    "Menú innovador y saludable",
                    "Ambiente único y acogedor",
                    "Precios competitivos",
                    "Servicio rápido y eficiente",
                    "Opción de delivery/takeout"
                ],
                "threats": [
                    "Competencia establecida",
                    "Aumento de costos de ingredientes",
                    "Cambios en preferencias del consumidor"
                ],
                "recommendations": [
                    "Enfocarse en un nicho específico",
                    "Crear experiencia única",
                    "Marketing en redes sociales",
                    "Programa de fidelización"
                ],
                "market_positioning": "Posicionarse como opción innovadora y accesible"
            },
            StoreType.CAFE: {
                "competitors": [
                    {
                        "name": "Cadenas de café establecidas",
                        "strengths": ["Marca reconocida", "Ubicaciones estratégicas"],
                        "weaknesses": ["Precios altos", "Ambiente genérico"]
                    }
                ],
                "opportunities": [
                    "Café de especialidad",
                    "Ambiente acogedor para trabajo",
                    "Eventos y actividades",
                    "Productos locales",
                    "Precios competitivos"
                ],
                "threats": [
                    "Competencia de grandes cadenas",
                    "Cambios en hábitos de consumo"
                ],
                "recommendations": [
                    "Diferenciarse con calidad y ambiente",
                    "Crear comunidad local",
                    "Eventos regulares",
                    "Productos únicos"
                ],
                "market_positioning": "Cafetería local con ambiente único y productos de calidad"
            },
            StoreType.BOUTIQUE: {
                "competitors": [
                    {
                        "name": "Tiendas de moda establecidas",
                        "strengths": ["Marca reconocida", "Variedad"],
                        "weaknesses": ["Precios altos", "Atención impersonal"]
                    }
                ],
                "opportunities": [
                    "Moda única y exclusiva",
                    "Atención personalizada",
                    "Eventos de moda",
                    "Colaboraciones con diseñadores locales"
                ],
                "threats": [
                    "Competencia online",
                    "Cambios en tendencias"
                ],
                "recommendations": [
                    "Enfocarse en experiencia única",
                    "Servicio personalizado",
                    "Eventos exclusivos",
                    "Presencia online fuerte"
                ],
                "market_positioning": "Boutique exclusiva con atención personalizada"
            }
        }
        
        return analysis_templates.get(
            store_type,
            {
                "competitors": [],
                "opportunities": ["Diferenciación", "Calidad", "Servicio"],
                "threats": ["Competencia", "Cambios de mercado"],
                "recommendations": ["Enfocarse en calidad", "Marketing efectivo"],
                "market_positioning": "Posicionarse como opción de calidad"
            }
        )




