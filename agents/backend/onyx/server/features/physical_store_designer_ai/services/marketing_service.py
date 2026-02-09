"""
Marketing Service - Genera planes de marketing y ventas
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os

from ..core.models import StoreType, MarketingPlan

logger = logging.getLogger(__name__)


class MarketingService:
    """Servicio para generar planes de marketing y ventas"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
    
    async def generate_marketing_plan(
        self,
        store_type: StoreType,
        store_name: str,
        target_audience: Optional[str] = None,
        location: Optional[str] = None
    ) -> MarketingPlan:
        """Generar plan de marketing y ventas"""
        
        # Si hay API key, usar LLM para generar plan personalizado
        if self.client:
            try:
                return await self._generate_with_llm(store_type, store_name, target_audience, location)
            except Exception as e:
                logger.error(f"Error generando plan con LLM: {e}")
                return self._generate_default_plan(store_type, store_name, target_audience)
        else:
            return self._generate_default_plan(store_type, store_name, target_audience)
    
    async def _generate_with_llm(
        self,
        store_type: StoreType,
        store_name: str,
        target_audience: Optional[str],
        location: Optional[str]
    ) -> MarketingPlan:
        """Generar plan usando LLM"""
        import json
        
        prompt = f"""Genera un plan completo de marketing y ventas para una tienda llamada "{store_name}".

Tipo de tienda: {store_type.value}
Audiencia objetivo: {target_audience or "General"}
Ubicación: {location or "No especificada"}

El plan debe incluir:
1. Audiencia objetivo detallada
2. Estrategias de marketing (mínimo 5)
3. Tácticas de ventas (mínimo 5)
4. Estrategia de precios
5. Ideas de promoción (mínimo 5)
6. Plan de redes sociales (instagram, facebook, tiktok)
7. Estrategia de apertura

Responde SOLO con un objeto JSON válido con estas claves exactas:
- target_audience: string
- marketing_strategy: array de strings
- sales_tactics: array de strings
- pricing_strategy: string
- promotion_ideas: array de strings
- social_media_plan: object con claves instagram, facebook, tiktok
- opening_strategy: string"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en marketing y ventas para negocios físicos. Responde siempre en formato JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return MarketingPlan(
                target_audience=data.get("target_audience", target_audience or "Clientes locales"),
                marketing_strategy=data.get("marketing_strategy", []),
                sales_tactics=data.get("sales_tactics", []),
                pricing_strategy=data.get("pricing_strategy", "Estrategia competitiva"),
                promotion_ideas=data.get("promotion_ideas", []),
                social_media_plan=data.get("social_media_plan", {}),
                opening_strategy=data.get("opening_strategy", f"Gran apertura de {store_name}")
            )
        except Exception as e:
            logger.error(f"Error parseando respuesta LLM: {e}")
            return self._generate_default_plan(store_type, store_name, target_audience)
    
    def _generate_default_plan(
        self,
        store_type: StoreType,
        store_name: str,
        target_audience: Optional[str]
    ) -> MarketingPlan:
        """Generar plan por defecto basado en templates"""
        
        # Templates por tipo de tienda
        templates = {
            StoreType.RESTAURANT: {
                "marketing_strategy": [
                    "Marketing en redes sociales con fotos de platos",
                    "Colaboraciones con influencers locales",
                    "Programa de fidelización con descuentos",
                    "Eventos especiales y noches temáticas",
                    "Reseñas y recomendaciones en plataformas de comida"
                ],
                "sales_tactics": [
                    "Menú del día con precio especial",
                    "Happy hour con descuentos",
                    "Combos familiares",
                    "Reservas online con beneficios",
                    "Programa de recomendaciones"
                ],
                "pricing_strategy": "Estrategia de precios competitivos con opciones para diferentes presupuestos",
                "promotion_ideas": [
                    "Apertura con 50% de descuento el primer día",
                    "Descuento para grupos grandes",
                    "Promociones de cumpleaños",
                    "Menú degustación",
                    "Eventos de networking"
                ]
            },
            StoreType.CAFE: {
                "marketing_strategy": [
                    "Instagram con estética de café",
                    "Colaboraciones con artistas locales",
                    "Ambiente acogedor para trabajo remoto",
                    "Eventos culturales y música en vivo",
                    "Programa de tarjeta de puntos"
                ],
                "sales_tactics": [
                    "Combos de desayuno",
                    "Happy hour de café",
                    "Productos de temporada",
                    "Venta de productos para llevar",
                    "Eventos temáticos"
                ],
                "pricing_strategy": "Precios competitivos con opciones premium",
                "promotion_ideas": [
                    "Café gratis el primer día",
                    "Descuento para estudiantes",
                    "Programa de lealtad",
                    "Eventos de cata de café",
                    "Colaboraciones con negocios locales"
                ]
            },
            StoreType.BOUTIQUE: {
                "marketing_strategy": [
                    "Fashion shows locales",
                    "Colaboraciones con influencers de moda",
                    "Contenido visual en redes sociales",
                    "Eventos de lanzamiento de colecciones",
                    "Programa VIP con beneficios exclusivos"
                ],
                "sales_tactics": [
                    "Personal shopping",
                    "Descuentos por temporada",
                    "Programa de recomendaciones",
                    "Eventos privados para clientes VIP",
                    "Venta online complementaria"
                ],
                "pricing_strategy": "Precios premium con opciones accesibles",
                "promotion_ideas": [
                    "Descuento de apertura",
                    "Eventos de lanzamiento",
                    "Colaboraciones con diseñadores",
                    "Programa de fidelización",
                    "Descuentos por volumen"
                ]
            }
        }
        
        template = templates.get(store_type, templates[StoreType.RETAIL])
        
        return MarketingPlan(
            target_audience=target_audience or "Clientes locales y visitantes",
            marketing_strategy=template["marketing_strategy"],
            sales_tactics=template["sales_tactics"],
            pricing_strategy=template["pricing_strategy"],
            promotion_ideas=template["promotion_ideas"],
            social_media_plan={
                "instagram": "Contenido visual diario, stories, reels",
                "facebook": "Eventos y promociones",
                "tiktok": "Contenido viral y tendencias"
            },
            opening_strategy=(
                f"Gran apertura de {store_name} con eventos especiales, "
                "promociones de lanzamiento y estrategia de relaciones públicas local"
            )
        )

