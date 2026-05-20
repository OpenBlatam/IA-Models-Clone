"""
Product Templates - Sistema de templates para productos comunes
================================================================
"""

import logging
from typing import Dict, List, Any, Optional
from ..models.schemas import ProductType

logger = logging.getLogger(__name__)


class ProductTemplateManager:
    """Gestor de templates para productos comunes"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Carga los templates de productos"""
        return {
            "licuadora_basica": {
                "name": "Licuadora Básica",
                "product_type": ProductType.LICUADORA,
                "description": "Licuadora estándar de uso doméstico",
                "materials": [
                    {"name": "Motor eléctrico 500W", "quantity": 1, "unit": "unidad"},
                    {"name": "Vaso de vidrio 1.5L", "quantity": 1, "unit": "unidad"},
                    {"name": "Base de plástico ABS", "quantity": 0.5, "unit": "kg"},
                    {"name": "Cuchillas de acero", "quantity": 1, "unit": "unidad"},
                    {"name": "Cables eléctricos", "quantity": 2, "unit": "metro"},
                    {"name": "Tornillos", "quantity": 20, "unit": "unidad"}
                ],
                "estimated_cost": 80.0,
                "difficulty": "media",
                "build_time": "2-3 horas"
            },
            "licuadora_premium": {
                "name": "Licuadora Premium",
                "product_type": ProductType.LICUADORA,
                "description": "Licuadora de alta gama con características avanzadas",
                "materials": [
                    {"name": "Motor eléctrico 1000W", "quantity": 1, "unit": "unidad"},
                    {"name": "Vaso de vidrio borosilicato 2L", "quantity": 1, "unit": "unidad"},
                    {"name": "Base de plástico ABS reforzado", "quantity": 0.8, "unit": "kg"},
                    {"name": "Cuchillas de acero inoxidable premium", "quantity": 1, "unit": "unidad"},
                    {"name": "Cables eléctricos de alta calidad", "quantity": 2.5, "unit": "metro"},
                    {"name": "Sistema de control de velocidad", "quantity": 1, "unit": "unidad"},
                    {"name": "Tornillos de acero inoxidable", "quantity": 25, "unit": "unidad"}
                ],
                "estimated_cost": 200.0,
                "difficulty": "alta",
                "build_time": "4-5 horas"
            },
            "estufa_4_quemadores": {
                "name": "Estufa 4 Quemadores",
                "product_type": ProductType.ESTUFA,
                "description": "Estufa de gas estándar con 4 quemadores",
                "materials": [
                    {"name": "Superficie de acero inoxidable", "quantity": 5, "unit": "kg"},
                    {"name": "Quemadores de gas", "quantity": 4, "unit": "unidad"},
                    {"name": "Válvulas de gas", "quantity": 4, "unit": "unidad"},
                    {"name": "Tubos de gas", "quantity": 2, "unit": "metro"},
                    {"name": "Perillas de control", "quantity": 4, "unit": "unidad"},
                    {"name": "Rejillas de soporte", "quantity": 4, "unit": "unidad"},
                    {"name": "Tornillos y fijaciones", "quantity": 40, "unit": "unidad"}
                ],
                "estimated_cost": 250.0,
                "difficulty": "alta",
                "build_time": "5-7 horas"
            },
            "maquina_corte_madera": {
                "name": "Máquina de Corte de Madera",
                "product_type": ProductType.MAQUINA,
                "description": "Máquina para cortar madera de forma precisa y segura",
                "materials": [
                    {"name": "Motor eléctrico 1500W", "quantity": 1, "unit": "unidad"},
                    {"name": "Hoja de sierra circular", "quantity": 1, "unit": "unidad"},
                    {"name": "Base de acero inoxidable", "quantity": 8, "unit": "kg"},
                    {"name": "Guía de corte ajustable", "quantity": 1, "unit": "unidad"},
                    {"name": "Sistema de seguridad", "quantity": 1, "unit": "unidad"},
                    {"name": "Cables y conectores", "quantity": 3, "unit": "metro"},
                    {"name": "Tornillos y fijaciones", "quantity": 50, "unit": "unidad"}
                ],
                "estimated_cost": 350.0,
                "difficulty": "muy_alta",
                "build_time": "8-10 horas"
            }
        }
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un template por ID"""
        return self.templates.get(template_id)
    
    def list_templates(self, product_type: Optional[ProductType] = None) -> List[Dict[str, Any]]:
        """Lista todos los templates disponibles"""
        templates = []
        for template_id, template in self.templates.items():
            if product_type is None or template["product_type"] == product_type:
                templates.append({
                    "id": template_id,
                    **template
                })
        return templates
    
    def create_from_template(self, template_id: str, customizations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Crea un prototipo desde un template con personalizaciones"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} no encontrado")
        
        # Copiar template
        prototype = template.copy()
        
        # Aplicar personalizaciones
        if customizations:
            if "materials" in customizations:
                prototype["materials"] = customizations["materials"]
            if "description" in customizations:
                prototype["description"] = customizations["description"]
            if "estimated_cost" in customizations:
                prototype["estimated_cost"] = customizations["estimated_cost"]
        
        return prototype




