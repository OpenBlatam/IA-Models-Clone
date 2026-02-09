"""
Recommendation Engine - Sistema de recomendaciones inteligente
===============================================================
"""

import logging
from typing import List, Dict, Any, Optional
from ..models.schemas import Material, BudgetOption

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Motor de recomendaciones para materiales y opciones"""
    
    def __init__(self):
        self.recommendation_rules = self._load_recommendation_rules()
    
    def _load_recommendation_rules(self) -> Dict[str, Any]:
        """Carga las reglas de recomendación"""
        return {
            "material_alternatives": {
                "acero_inoxidable": ["aluminio", "acero_carbon"],
                "plastico_abs": ["plastico_pet", "plastico_pla"],
                "vidrio": ["policarbonato", "acrilico"]
            },
            "budget_recommendations": {
                "bajo": {
                    "focus": "funcionalidad",
                    "priorities": ["durabilidad_basica", "precio"]
                },
                "medio": {
                    "focus": "balance",
                    "priorities": ["calidad", "precio", "durabilidad"]
                },
                "alto": {
                    "focus": "calidad",
                    "priorities": ["durabilidad", "acabados", "prestigio"]
                }
            }
        }
    
    def recommend_materials(self, materials: List[Material], budget: Optional[float] = None) -> List[Dict[str, Any]]:
        """Recomienda materiales alternativos o mejoras"""
        recommendations = []
        
        for material in materials:
            material_key = material.name.lower().replace(" ", "_")
            
            # Buscar alternativas
            alternatives = self.recommendation_rules["material_alternatives"].get(material_key, [])
            
            if alternatives:
                recommendations.append({
                    "material_original": material.name,
                    "alternativas": alternatives,
                    "razon": "Materiales similares con diferentes características"
                })
            
            # Recomendaciones según presupuesto
            if budget:
                if material.total_price > budget * 0.3:  # Si un material es más del 30% del presupuesto
                    recommendations.append({
                        "material": material.name,
                        "tipo": "optimizacion_presupuesto",
                        "sugerencia": f"Considera buscar alternativas más económicas para {material.name}",
                        "ahorro_potencial": f"${material.total_price * 0.2:.2f}"
                    })
        
        return recommendations
    
    def recommend_budget_option(self, budget_options: List[BudgetOption], 
                               user_budget: Optional[float] = None) -> Optional[BudgetOption]:
        """Recomienda la mejor opción de presupuesto"""
        if not budget_options:
            return None
        
        if user_budget:
            # Encontrar la opción más cercana al presupuesto sin excederlo
            suitable_options = [opt for opt in budget_options if opt.total_cost <= user_budget]
            if suitable_options:
                # Ordenar por calidad y elegir la mejor dentro del presupuesto
                suitable_options.sort(key=lambda x: (
                    {"premium": 4, "alta": 3, "estándar": 2, "básica": 1}.get(x.quality_level.lower(), 0),
                    -x.total_cost
                ), reverse=True)
                return suitable_options[0]
        
        # Si no hay presupuesto específico, recomendar la opción media
        for option in budget_options:
            if option.budget_level == "medio":
                return option
        
        # Si no hay opción media, devolver la primera
        return budget_options[0]
    
    def get_optimization_tips(self, materials: List[Material], 
                              product_type: Optional[Any] = None) -> List[str]:
        """Obtiene tips de optimización"""
        tips = []
        
        total_cost = sum(m.total_price for m in materials)
        
        # Tip según costo total
        if total_cost > 500:
            tips.append("Considera comprar materiales en grandes cantidades para obtener descuentos")
        
        # Tips según tipo de producto
        product_type_str = None
        if product_type:
            if hasattr(product_type, 'value'):
                product_type_str = product_type.value
            else:
                product_type_str = str(product_type).lower()
        
        if product_type_str == "licuadora":
            tips.append("Para licuadoras, prioriza la calidad del motor sobre otros componentes")
            tips.append("El vaso de vidrio es más durable pero más costoso que el plástico")
        
        elif product_type_str == "estufa":
            tips.append("Asegúrate de que las válvulas de gas cumplan con las normativas locales")
            tips.append("La superficie de acero inoxidable requiere mantenimiento regular")
        
        # Tips generales
        tips.extend([
            "Compara precios en múltiples fuentes antes de comprar",
            "Considera comprar materiales usados o reciclados para reducir costos",
            "Agrupa compras para reducir costos de envío",
            "Verifica la disponibilidad de materiales antes de comenzar el proyecto"
        ])
        
        return tips

