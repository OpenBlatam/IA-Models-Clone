"""
Cost Analyzer - Análisis detallado de costos
=============================================
"""

import logging
from typing import Dict, List, Any
from ..models.schemas import PrototypeResponse, Material, BudgetOption

logger = logging.getLogger(__name__)


class CostAnalyzer:
    """Analizador detallado de costos"""
    
    def analyze_costs(self, response: PrototypeResponse) -> Dict[str, Any]:
        """Realiza un análisis completo de costos"""
        materials = response.materials
        total_cost = response.total_cost_estimate
        
        # Análisis por categoría
        category_analysis = self._analyze_by_category(materials)
        
        # Análisis de costos por material
        material_analysis = self._analyze_materials(materials)
        
        # Proyecciones de costo
        cost_projections = self._project_costs(total_cost, materials)
        
        # Análisis de ahorros potenciales
        savings_analysis = self._analyze_savings(materials, response.budget_options)
        
        # Distribución de costos
        cost_distribution = self._calculate_distribution(materials, total_cost)
        
        return {
            "total_cost": total_cost,
            "category_breakdown": category_analysis,
            "material_analysis": material_analysis,
            "cost_projections": cost_projections,
            "savings_analysis": savings_analysis,
            "cost_distribution": cost_distribution,
            "recommendations": self._generate_cost_recommendations(materials, total_cost)
        }
    
    def _analyze_by_category(self, materials: List[Material]) -> Dict[str, Any]:
        """Analiza costos por categoría"""
        categories = {}
        
        for material in materials:
            category = material.category
            if category not in categories:
                categories[category] = {
                    "materials": [],
                    "total_cost": 0,
                    "count": 0
                }
            
            categories[category]["materials"].append({
                "name": material.name,
                "cost": material.total_price
            })
            categories[category]["total_cost"] += material.total_price
            categories[category]["count"] += 1
        
        # Calcular porcentajes
        total = sum(cat["total_cost"] for cat in categories.values())
        for category in categories:
            categories[category]["percentage"] = (
                (categories[category]["total_cost"] / total * 100) if total > 0 else 0
            )
        
        return categories
    
    def _analyze_materials(self, materials: List[Material]) -> Dict[str, Any]:
        """Analiza los materiales individuales"""
        sorted_materials = sorted(materials, key=lambda m: m.total_price, reverse=True)
        
        return {
            "most_expensive": [
                {
                    "name": m.name,
                    "cost": m.total_price,
                    "percentage": (m.total_price / sum(mat.total_price for mat in materials) * 100) if materials else 0
                }
                for m in sorted_materials[:3]
            ],
            "least_expensive": [
                {
                    "name": m.name,
                    "cost": m.total_price
                }
                for m in sorted_materials[-3:]
            ],
            "average_cost": sum(m.total_price for m in materials) / len(materials) if materials else 0,
            "cost_range": {
                "min": min(m.total_price for m in materials) if materials else 0,
                "max": max(m.total_price for m in materials) if materials else 0
            }
        }
    
    def _project_costs(self, total_cost: float, materials: List[Material]) -> Dict[str, Any]:
        """Proyecta costos en diferentes escenarios"""
        # Factor de imprevistos (10-20%)
        contingency_factor = 0.15
        
        # Factor de envío (5-10%)
        shipping_factor = 0.07
        
        # Factor de herramientas (si no se tienen)
        tools_factor = 0.05
        
        base_cost = total_cost
        with_contingency = base_cost * (1 + contingency_factor)
        with_shipping = with_contingency * (1 + shipping_factor)
        with_tools = with_shipping * (1 + tools_factor)
        
        return {
            "base_cost": base_cost,
            "with_contingency_15%": with_contingency,
            "with_shipping_7%": with_shipping,
            "with_tools_5%": with_tools,
            "recommended_budget": with_tools,
            "breakdown": {
                "materials": base_cost,
                "contingency": base_cost * contingency_factor,
                "shipping": with_contingency * shipping_factor,
                "tools": with_shipping * tools_factor
            }
        }
    
    def _analyze_savings(self, materials: List[Material], 
                        budget_options: List[BudgetOption]) -> Dict[str, Any]:
        """Analiza oportunidades de ahorro"""
        base_cost = sum(m.total_price for m in materials)
        
        savings_opportunities = []
        
        # Comparar con opciones de presupuesto
        for option in budget_options:
            if option.total_cost < base_cost:
                savings_opportunities.append({
                    "option": option.budget_level,
                    "savings": base_cost - option.total_cost,
                    "percentage": ((base_cost - option.total_cost) / base_cost * 100) if base_cost > 0 else 0,
                    "trade_offs": option.trade_offs
                })
        
        # Análisis de materiales con múltiples fuentes (oportunidad de comparar precios)
        materials_with_multiple_sources = [
            m for m in materials if len(m.sources) > 1
        ]
        
        potential_savings = sum(
            m.total_price * 0.1 for m in materials_with_multiple_sources
        )  # Asumir 10% de ahorro por comparar precios
        
        return {
            "base_cost": base_cost,
            "savings_opportunities": savings_opportunities,
            "potential_savings_from_comparison": potential_savings,
            "materials_with_multiple_sources": len(materials_with_multiple_sources),
            "recommendation": "Compara precios en múltiples fuentes para ahorrar hasta 10%"
        }
    
    def _calculate_distribution(self, materials: List[Material], 
                               total_cost: float) -> Dict[str, Any]:
        """Calcula la distribución de costos"""
        costs = [m.total_price for m in materials]
        
        # Dividir en terciles
        sorted_costs = sorted(costs, reverse=True)
        n = len(sorted_costs)
        
        top_third = sorted_costs[:n//3] if n >= 3 else sorted_costs
        middle_third = sorted_costs[n//3:2*n//3] if n >= 6 else []
        bottom_third = sorted_costs[2*n//3:] if n >= 9 else []
        
        return {
            "top_third": {
                "materials": len(top_third),
                "total_cost": sum(top_third),
                "percentage": (sum(top_third) / total_cost * 100) if total_cost > 0 else 0
            },
            "middle_third": {
                "materials": len(middle_third),
                "total_cost": sum(middle_third),
                "percentage": (sum(middle_third) / total_cost * 100) if total_cost > 0 else 0
            },
            "bottom_third": {
                "materials": len(bottom_third),
                "total_cost": sum(bottom_third),
                "percentage": (sum(bottom_third) / total_cost * 100) if total_cost > 0 else 0
            },
            "cost_concentration": "alta" if sum(top_third) / total_cost > 0.6 else "media" if sum(top_third) / total_cost > 0.4 else "baja"
        }
    
    def _generate_cost_recommendations(self, materials: List[Material], 
                                     total_cost: float) -> List[str]:
        """Genera recomendaciones de costos"""
        recommendations = []
        
        # Materiales más costosos
        sorted_materials = sorted(materials, key=lambda m: m.total_price, reverse=True)
        top_material = sorted_materials[0] if sorted_materials else None
        
        if top_material and top_material.total_price > total_cost * 0.3:
            recommendations.append(
                f"El material '{top_material.name}' representa más del 30% del costo total. "
                f"Considera buscar alternativas."
            )
        
        # Materiales con múltiples fuentes
        materials_with_sources = [m for m in materials if len(m.sources) > 1]
        if materials_with_sources:
            recommendations.append(
                f"Tienes {len(materials_with_sources)} materiales con múltiples fuentes. "
                f"Compara precios para ahorrar."
            )
        
        # Recomendación general
        if total_cost > 500:
            recommendations.append(
                "Para proyectos grandes, considera comprar materiales en grandes cantidades "
                "para obtener descuentos por volumen."
            )
        
        if not recommendations:
            recommendations.append("Los costos están bien distribuidos. No se requieren optimizaciones urgentes.")
        
        return recommendations




