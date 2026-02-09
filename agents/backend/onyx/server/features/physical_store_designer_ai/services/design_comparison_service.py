"""
Design Comparison Service - Comparación de diseños
"""

import logging
from typing import List, Dict, Any, Optional
from ..core.models import StoreDesign

logger = logging.getLogger(__name__)


class DesignComparisonService:
    """Servicio para comparar diseños"""
    
    def compare_designs(
        self,
        designs: List[StoreDesign],
        comparison_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comparar múltiples diseños"""
        
        if len(designs) < 2:
            raise ValueError("Se necesitan al menos 2 diseños para comparar")
        
        criteria = comparison_criteria or [
            "cost",
            "style",
            "marketing_potential",
            "financial_viability",
            "decoration_complexity"
        ]
        
        comparison = {
            "designs_count": len(designs),
            "designs": [self._get_design_summary(d) for d in designs],
            "comparison": {}
        }
        
        for criterion in criteria:
            comparison["comparison"][criterion] = self._compare_by_criterion(
                designs, criterion
            )
        
        comparison["recommendations"] = self._generate_comparison_recommendations(
            designs, comparison["comparison"]
        )
        
        return comparison
    
    def _get_design_summary(self, design: StoreDesign) -> Dict[str, Any]:
        """Obtener resumen de diseño"""
        total_cost = 0
        if design.decoration_plan and design.decoration_plan.budget_estimate:
            total_cost = sum(design.decoration_plan.budget_estimate.values())
        
        if design.financial_analysis:
            total_cost += design.financial_analysis.get("initial_investment", {}).get("total", 0)
        
        return {
            "store_id": design.store_id,
            "store_name": design.store_name,
            "store_type": design.store_type.value,
            "style": design.style.value,
            "total_cost": total_cost,
            "monthly_profit": design.financial_analysis.get("profitability", {}).get("monthly_profit", 0) if design.financial_analysis else 0,
            "break_even_months": design.financial_analysis.get("break_even", {}).get("months") if design.financial_analysis else None
        }
    
    def _compare_by_criterion(
        self,
        designs: List[StoreDesign],
        criterion: str
    ) -> Dict[str, Any]:
        """Comparar diseños por criterio específico"""
        
        if criterion == "cost":
            costs = []
            for design in designs:
                total = 0
                if design.decoration_plan and design.decoration_plan.budget_estimate:
                    total = sum(design.decoration_plan.budget_estimate.values())
                if design.financial_analysis:
                    total += design.financial_analysis.get("initial_investment", {}).get("total", 0)
                costs.append((design.store_id, total))
            
            costs.sort(key=lambda x: x[1])
            return {
                "cheapest": costs[0][0],
                "most_expensive": costs[-1][0],
                "cost_range": {
                    "min": costs[0][1],
                    "max": costs[-1][1],
                    "average": sum(c[1] for c in costs) / len(costs)
                },
                "ranking": [{"store_id": sid, "cost": cost} for sid, cost in costs]
            }
        
        elif criterion == "financial_viability":
            profits = []
            for design in designs:
                profit = 0
                if design.financial_analysis:
                    profit = design.financial_analysis.get("profitability", {}).get("monthly_profit", 0)
                profits.append((design.store_id, profit))
            
            profits.sort(key=lambda x: x[1], reverse=True)
            return {
                "most_profitable": profits[0][0] if profits[0][1] > 0 else None,
                "profit_range": {
                    "max": profits[0][1] if profits else 0,
                    "min": profits[-1][1] if profits else 0,
                    "average": sum(p[1] for p in profits) / len(profits) if profits else 0
                },
                "ranking": [{"store_id": sid, "monthly_profit": profit} for sid, profit in profits]
            }
        
        elif criterion == "style":
            styles = {}
            for design in designs:
                style = design.style.value
                if style not in styles:
                    styles[style] = []
                styles[style].append(design.store_id)
            
            return {
                "style_distribution": styles,
                "most_common_style": max(styles.items(), key=lambda x: len(x[1]))[0] if styles else None
            }
        
        elif criterion == "marketing_potential":
            marketing_scores = []
            for design in designs:
                score = 0
                if design.marketing_plan:
                    score += len(design.marketing_plan.marketing_strategy) * 2
                    score += len(design.marketing_plan.promotion_ideas)
                    score += len(design.marketing_plan.sales_tactics)
                marketing_scores.append((design.store_id, score))
            
            marketing_scores.sort(key=lambda x: x[1], reverse=True)
            return {
                "highest_potential": marketing_scores[0][0] if marketing_scores else None,
                "scores": {sid: score for sid, score in marketing_scores},
                "ranking": [{"store_id": sid, "score": score} for sid, score in marketing_scores]
            }
        
        elif criterion == "decoration_complexity":
            complexities = []
            for design in designs:
                complexity = 0
                if design.decoration_plan:
                    complexity += len(design.decoration_plan.furniture_recommendations)
                    complexity += len(design.decoration_plan.decoration_elements)
                    complexity += len(design.decoration_plan.materials)
                complexities.append((design.store_id, complexity))
            
            complexities.sort(key=lambda x: x[1], reverse=True)
            return {
                "most_complex": complexities[0][0] if complexities else None,
                "least_complex": complexities[-1][0] if complexities else None,
                "complexity_range": {
                    "max": complexities[0][1] if complexities else 0,
                    "min": complexities[-1][1] if complexities else 0,
                    "average": sum(c[1] for c in complexities) / len(complexities) if complexities else 0
                },
                "ranking": [{"store_id": sid, "complexity": comp} for sid, comp in complexities]
            }
        
        return {}
    
    def _generate_comparison_recommendations(
        self,
        designs: List[StoreDesign],
        comparison: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones basadas en comparación"""
        recommendations = []
        
        # Análisis de costo
        if "cost" in comparison:
            cost_comp = comparison["cost"]
            cheapest_id = cost_comp.get("cheapest")
            most_expensive_id = cost_comp.get("most_expensive")
            
            if cheapest_id != most_expensive_id:
                cheapest_design = next(d for d in designs if d.store_id == cheapest_id)
                recommendations.append(
                    f"El diseño más económico es '{cheapest_design.store_name}' "
                    f"(${cost_comp['cost_range']['min']:,.0f})"
                )
        
        # Análisis de rentabilidad
        if "financial_viability" in comparison:
            fin_comp = comparison["financial_viability"]
            most_profitable = fin_comp.get("most_profitable")
            
            if most_profitable:
                profitable_design = next(d for d in designs if d.store_id == most_profitable)
                recommendations.append(
                    f"El diseño más rentable es '{profitable_design.store_name}' "
                    f"con ganancia mensual de ${fin_comp['profit_range']['max']:,.0f}"
                )
        
        # Análisis de marketing
        if "marketing_potential" in comparison:
            marketing_comp = comparison["marketing_potential"]
            highest = marketing_comp.get("highest_potential")
            
            if highest:
                marketing_design = next(d for d in designs if d.store_id == highest)
                recommendations.append(
                    f"'{marketing_design.store_name}' tiene el mayor potencial de marketing "
                    f"con un score de {marketing_comp['scores'].get(highest, 0)}"
                )
        
        # Recomendación general
        if len(designs) > 2:
            recommendations.append(
                f"Considera combinar elementos de los mejores diseños en cada categoría"
            )
        
        return recommendations




