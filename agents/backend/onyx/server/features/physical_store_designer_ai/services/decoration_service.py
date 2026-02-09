"""
Decoration Service - Genera planes de decoración del local
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os

from ..core.models import DecorationPlan, DesignStyle, StoreType

logger = logging.getLogger(__name__)


class DecorationService:
    """Servicio para generar planes de decoración"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
    
    def generate_decoration_plan(
        self,
        store_type: StoreType,
        style: DesignStyle,
        dimensions: Optional[Dict[str, float]] = None,
        budget_range: Optional[str] = None
    ) -> DecorationPlan:
        """Generar plan de decoración"""
        
        # Si hay API key, usar LLM para generar plan personalizado
        if self.client:
            try:
                return self._generate_with_llm(store_type, style, dimensions, budget_range)
            except Exception as e:
                logger.error(f"Error generando plan con LLM: {e}")
                return self._generate_default_plan(store_type, style, dimensions, budget_range)
        else:
            return self._generate_default_plan(store_type, style, dimensions, budget_range)
    
    def _generate_with_llm(
        self,
        store_type: StoreType,
        style: DesignStyle,
        dimensions: Optional[Dict[str, float]],
        budget_range: Optional[str]
    ) -> DecorationPlan:
        """Generar plan usando LLM"""
        # Por ahora usar plan por defecto, mejorar con LLM después
        return self._generate_default_plan(store_type, style, dimensions, budget_range)
    
    def _generate_default_plan(
        self,
        store_type: StoreType,
        style: DesignStyle,
        dimensions: Optional[Dict[str, float]],
        budget_range: Optional[str]
    ) -> DecorationPlan:
        """Generar plan por defecto basado en templates"""
        
        # Esquemas de color por estilo
        color_schemes = {
            DesignStyle.MODERN: {
                "primary": "#2C3E50",
                "secondary": "#3498DB",
                "accent": "#E74C3C",
                "neutral": "#ECF0F1"
            },
            DesignStyle.CLASSIC: {
                "primary": "#8B4513",
                "secondary": "#D2B48C",
                "accent": "#CD853F",
                "neutral": "#F5F5DC"
            },
            DesignStyle.MINIMALIST: {
                "primary": "#FFFFFF",
                "secondary": "#000000",
                "accent": "#808080",
                "neutral": "#F5F5F5"
            },
            DesignStyle.INDUSTRIAL: {
                "primary": "#2C2C2C",
                "secondary": "#7F8C8D",
                "accent": "#E67E22",
                "neutral": "#BDC3C7"
            },
            DesignStyle.RUSTIC: {
                "primary": "#8B4513",
                "secondary": "#A0522D",
                "accent": "#D2691E",
                "neutral": "#F5DEB3"
            },
            DesignStyle.LUXURY: {
                "primary": "#1A1A1A",
                "secondary": "#D4AF37",
                "accent": "#8B7355",
                "neutral": "#F5F5F5"
            },
            DesignStyle.ECO_FRIENDLY: {
                "primary": "#2E7D32",
                "secondary": "#66BB6A",
                "accent": "#81C784",
                "neutral": "#E8F5E9"
            },
            DesignStyle.VINTAGE: {
                "primary": "#8B4513",
                "secondary": "#D2691E",
                "accent": "#CD853F",
                "neutral": "#FFF8DC"
            }
        }
        
        # Plan de iluminación
        lighting_plan = {
            "natural": "Maximizar luz natural con ventanas grandes",
            "ambient": "Iluminación LED cálida para ambiente acogedor",
            "task": "Iluminación focalizada en áreas de trabajo",
            "accent": "Iluminación decorativa para destacar elementos"
        }
        
        # Recomendaciones de muebles según tipo de tienda
        furniture_recommendations = {
            StoreType.RESTAURANT: [
                {"item": "Mesas de comedor", "quantity": "Según capacidad", "style": style.value},
                {"item": "Sillas cómodas", "quantity": "2 por mesa", "style": style.value},
                {"item": "Barra o mostrador", "quantity": "1", "style": style.value},
                {"item": "Almacenamiento", "quantity": "Según necesidades", "style": style.value}
            ],
            StoreType.CAFE: [
                {"item": "Mesas pequeñas", "quantity": "8-12", "style": style.value},
                {"item": "Sillas y sofás", "quantity": "2-4 por mesa", "style": style.value},
                {"item": "Mostrador de servicio", "quantity": "1", "style": style.value},
                {"item": "Estanterías decorativas", "quantity": "2-3", "style": style.value}
            ],
            StoreType.BOUTIQUE: [
                {"item": "Percheros y estanterías", "quantity": "Según espacio", "style": style.value},
                {"item": "Espejos grandes", "quantity": "2-3", "style": style.value},
                {"item": "Mostrador de caja", "quantity": "1", "style": style.value},
                {"item": "Sofá o banco para probadores", "quantity": "1-2", "style": style.value}
            ],
            StoreType.RETAIL: [
                {"item": "Estanterías modulares", "quantity": "Según productos", "style": style.value},
                {"item": "Mostrador de atención", "quantity": "1-2", "style": style.value},
                {"item": "Vitrinas", "quantity": "2-4", "style": style.value},
                {"item": "Sistema de almacenamiento", "quantity": "Según inventario", "style": style.value}
            ]
        }
        
        # Elementos decorativos
        decoration_elements = {
            DesignStyle.MODERN: [
                "Líneas geométricas",
                "Plantas minimalistas",
                "Arte abstracto",
                "Espejos grandes",
                "Iluminación LED integrada"
            ],
            DesignStyle.CLASSIC: [
                "Molduras decorativas",
                "Cuadros clásicos",
                "Cortinas elegantes",
                "Alfombras",
                "Lámparas de estilo tradicional"
            ],
            DesignStyle.MINIMALIST: [
                "Espacios abiertos",
                "Plantas verdes",
                "Arte minimalista",
                "Superficies limpias",
                "Iluminación natural"
            ],
            DesignStyle.INDUSTRIAL: [
                "Tubos expuestos",
                "Ladrillo visto",
                "Metal y concreto",
                "Iluminación industrial",
                "Elementos vintage"
            ],
            DesignStyle.RUSTIC: [
                "Madera natural",
                "Piedra",
                "Textiles naturales",
                "Plantas",
                "Elementos artesanales"
            ],
            DesignStyle.LUXURY: [
                "Materiales premium",
                "Acabados refinados",
                "Iluminación sofisticada",
                "Arte exclusivo",
                "Detalles dorados o plateados"
            ],
            DesignStyle.ECO_FRIENDLY: [
                "Materiales reciclados",
                "Plantas vivas",
                "Iluminación LED eficiente",
                "Muebles sostenibles",
                "Elementos naturales"
            ],
            DesignStyle.VINTAGE: [
                "Muebles retro",
                "Carteles antiguos",
                "Iluminación vintage",
                "Textiles con patrones clásicos",
                "Elementos decorativos antiguos"
            ]
        }
        
        # Materiales recomendados
        materials = {
            DesignStyle.MODERN: ["Acero", "Vidrio", "Concreto pulido", "Madera clara"],
            DesignStyle.CLASSIC: ["Madera noble", "Mármol", "Textiles elegantes", "Metal dorado"],
            DesignStyle.MINIMALIST: ["Madera clara", "Vidrio", "Pintura blanca", "Metal cromado"],
            DesignStyle.INDUSTRIAL: ["Metal", "Concreto", "Ladrillo", "Madera recuperada"],
            DesignStyle.RUSTIC: ["Madera natural", "Piedra", "Textiles rústicos", "Hierro forjado"],
            DesignStyle.LUXURY: ["Mármol", "Madera exótica", "Metal precioso", "Textiles premium"],
            DesignStyle.ECO_FRIENDLY: ["Bambú", "Corcho", "Materiales reciclados", "Pintura ecológica"],
            DesignStyle.VINTAGE: ["Madera antigua", "Metal envejecido", "Textiles vintage", "Vidrio antiguo"]
        }
        
        # Estimación de presupuesto (simplificada)
        base_budget = {
            "low": {"furniture": 5000, "decoration": 2000, "lighting": 1500, "paint": 800},
            "medium": {"furniture": 15000, "decoration": 5000, "lighting": 3000, "paint": 2000},
            "high": {"furniture": 40000, "decoration": 15000, "lighting": 8000, "paint": 5000}
        }
        
        budget_key = "medium"
        if budget_range:
            budget_lower = budget_range.lower()
            if "bajo" in budget_lower or "low" in budget_lower:
                budget_key = "low"
            elif "alto" in budget_lower or "high" in budget_lower:
                budget_key = "high"
        
        return DecorationPlan(
            color_scheme=color_schemes.get(style, color_schemes[DesignStyle.MODERN]),
            lighting_plan=lighting_plan,
            furniture_recommendations=furniture_recommendations.get(
                store_type, 
                furniture_recommendations[StoreType.RETAIL]
            ),
            decoration_elements=decoration_elements.get(style, decoration_elements[DesignStyle.MODERN]),
            materials=materials.get(style, materials[DesignStyle.MODERN]),
            budget_estimate=base_budget[budget_key]
        )




