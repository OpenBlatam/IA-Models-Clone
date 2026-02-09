"""
Template Service - Sistema de templates predefinidos
"""

import logging
from typing import Dict, Any, List, Optional
from ..core.models import StoreType, DesignStyle

logger = logging.getLogger(__name__)


class TemplateService:
    """Servicio para templates predefinidos"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Cargar templates predefinidos"""
        return {
            "modern_cafe": {
                "name": "Café Moderno",
                "description": "Cafetería moderna con ambiente acogedor",
                "store_type": StoreType.CAFE.value,
                "style": DesignStyle.MODERN.value,
                "features": [
                    "Iluminación LED moderna",
                    "Mobiliario minimalista",
                    "Área de trabajo",
                    "Barra de café profesional"
                ],
                "target_audience": "Jóvenes profesionales y estudiantes",
                "budget_range": "medio"
            },
            "luxury_boutique": {
                "name": "Boutique de Lujo",
                "description": "Boutique elegante con productos premium",
                "store_type": StoreType.BOUTIQUE.value,
                "style": DesignStyle.LUXURY.value,
                "features": [
                    "Iluminación sofisticada",
                    "Probadores exclusivos",
                    "Mostrador de mármol",
                    "Ambiente exclusivo"
                ],
                "target_audience": "Clientes de alto poder adquisitivo",
                "budget_range": "alto"
            },
            "rustic_restaurant": {
                "name": "Restaurante Rústico",
                "description": "Restaurante con ambiente rústico y acogedor",
                "store_type": StoreType.RESTAURANT.value,
                "style": DesignStyle.RUSTIC.value,
                "features": [
                    "Madera natural",
                    "Iluminación cálida",
                    "Ambiente familiar",
                    "Cocina abierta"
                ],
                "target_audience": "Familias y grupos",
                "budget_range": "medio"
            },
            "industrial_retail": {
                "name": "Tienda Industrial",
                "description": "Tienda retail con estilo industrial",
                "store_type": StoreType.RETAIL.value,
                "style": DesignStyle.INDUSTRIAL.value,
                "features": [
                    "Materiales rústicos",
                    "Iluminación industrial",
                    "Espacios abiertos",
                    "Estanterías modulares"
                ],
                "target_audience": "Clientes jóvenes y urbanos",
                "budget_range": "medio"
            },
            "eco_friendly_cafe": {
                "name": "Café Ecológico",
                "description": "Cafetería sostenible y ecológica",
                "store_type": StoreType.CAFE.value,
                "style": DesignStyle.ECO_FRIENDLY.value,
                "features": [
                    "Materiales reciclados",
                    "Plantas vivas",
                    "Iluminación LED eficiente",
                    "Productos orgánicos"
                ],
                "target_audience": "Consumidores conscientes",
                "budget_range": "medio"
            },
            "minimalist_boutique": {
                "name": "Boutique Minimalista",
                "description": "Boutique con diseño minimalista",
                "store_type": StoreType.BOUTIQUE.value,
                "style": DesignStyle.MINIMALIST.value,
                "features": [
                    "Espacios abiertos",
                    "Líneas limpias",
                    "Colores neutros",
                    "Enfoque en productos"
                ],
                "target_audience": "Amantes del diseño minimalista",
                "budget_range": "medio"
            }
        }
    
    def get_templates(
        self,
        store_type: Optional[StoreType] = None,
        style: Optional[DesignStyle] = None
    ) -> List[Dict[str, Any]]:
        """Obtener templates filtrados"""
        templates = list(self.templates.values())
        
        if store_type:
            templates = [t for t in templates if t["store_type"] == store_type.value]
        
        if style:
            templates = [t for t in templates if t["style"] == style.value]
        
        return templates
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Obtener template específico"""
        return self.templates.get(template_id)
    
    def apply_template(
        self,
        template_id: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Aplicar template con personalizaciones"""
        template = self.get_template(template_id)
        
        if not template:
            raise ValueError(f"Template {template_id} no encontrado")
        
        result = template.copy()
        
        if customizations:
            result.update(customizations)
            result["customized"] = True
        else:
            result["customized"] = False
        
        return result
    
    def list_all_templates(self) -> List[Dict[str, Any]]:
        """Listar todos los templates"""
        return [
            {
                "id": template_id,
                **template
            }
            for template_id, template in self.templates.items()
        ]




