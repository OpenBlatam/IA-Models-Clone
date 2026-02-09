"""
Sistema de recomendaciones basadas en presupuesto
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BudgetRecommendation:
    """Recomendación basada en presupuesto"""
    product_id: str
    product_name: str
    category: str
    price: float
    priority: int  # 1-5
    value_score: float  # 0-100, relación calidad/precio
    reason: str
    alternatives: List[Dict] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "category": self.category,
            "price": self.price,
            "priority": self.priority,
            "value_score": self.value_score,
            "reason": self.reason,
            "alternatives": self.alternatives
        }


class BudgetBasedRecommendations:
    """Sistema de recomendaciones basadas en presupuesto"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.product_database: Dict[str, Dict] = {
            "product1": {"name": "Cleanser Budget", "category": "cleanser", "price": 10.0, "rating": 4.0},
            "product2": {"name": "Cleanser Premium", "category": "cleanser", "price": 35.0, "rating": 4.8},
            "product3": {"name": "Moisturizer Budget", "category": "moisturizer", "price": 12.0, "rating": 4.2},
            "product4": {"name": "Moisturizer Premium", "category": "moisturizer", "price": 45.0, "rating": 4.9},
            "product5": {"name": "Serum Budget", "category": "serum", "price": 15.0, "rating": 4.1},
            "product6": {"name": "Serum Premium", "category": "serum", "price": 60.0, "rating": 4.9}
        }
    
    def get_budget_recommendations(self, budget_limit: float, skin_needs: List[str],
                                  essential_only: bool = False) -> List[BudgetRecommendation]:
        """Obtiene recomendaciones basadas en presupuesto"""
        recommendations = []
        used_budget = 0.0
        
        # Categorías esenciales
        essential_categories = ["cleanser", "moisturizer", "sunscreen"]
        categories_to_cover = essential_categories if essential_only else ["cleanser", "moisturizer", "serum", "sunscreen"]
        
        for category in categories_to_cover:
            if used_budget >= budget_limit:
                break
            
            # Buscar productos en esta categoría
            category_products = [
                (pid, p) for pid, p in self.product_database.items()
                if p["category"] == category
            ]
            
            if not category_products:
                continue
            
            # Ordenar por value score (rating/price)
            category_products.sort(
                key=lambda x: x[1]["rating"] / max(x[1]["price"], 1),
                reverse=True
            )
            
            # Encontrar mejor producto que quepa en el presupuesto
            for product_id, product in category_products:
                remaining_budget = budget_limit - used_budget
                
                if product["price"] <= remaining_budget:
                    value_score = (product["rating"] / product["price"]) * 10
                    
                    # Buscar alternativas más baratas
                    alternatives = [
                        {"id": pid, "name": p["name"], "price": p["price"]}
                        for pid, p in category_products
                        if p["price"] < product["price"] and p["price"] <= remaining_budget
                    ][:2]
                    
                    recommendation = BudgetRecommendation(
                        product_id=product_id,
                        product_name=product["name"],
                        category=category,
                        price=product["price"],
                        priority=1 if category in essential_categories else 2,
                        value_score=value_score,
                        reason=f"Mejor relación calidad/precio en {category}",
                        alternatives=alternatives
                    )
                    
                    recommendations.append(recommendation)
                    used_budget += product["price"]
                    break
        
        # Ordenar por prioridad
        recommendations.sort(key=lambda x: (x.priority, -x.value_score))
        
        return recommendations
    
    def optimize_routine_for_budget(self, current_routine: List[Dict],
                                   budget_limit: float) -> Dict:
        """Optimiza rutina para presupuesto"""
        current_cost = sum(p.get("price", 0) for p in current_routine)
        
        if current_cost <= budget_limit:
            return {
                "optimization_needed": False,
                "current_cost": current_cost,
                "budget_limit": budget_limit,
                "savings": 0.0,
                "message": "Tu rutina actual está dentro del presupuesto"
            }
        
        # Calcular ahorro necesario
        savings_needed = current_cost - budget_limit
        
        # Sugerir productos más económicos
        optimized_routine = []
        optimized_cost = 0.0
        
        categories = set(p.get("category") for p in current_routine)
        
        for category in categories:
            category_products = [
                (pid, p) for pid, p in self.product_database.items()
                if p["category"] == category
            ]
            
            if category_products:
                # Encontrar producto más barato que aún sea bueno (rating > 4.0)
                affordable = [
                    (pid, p) for pid, p in category_products
                    if p["rating"] >= 4.0
                ]
                
                if affordable:
                    affordable.sort(key=lambda x: x[1]["price"])
                    best_affordable = affordable[0]
                    
                    optimized_routine.append({
                        "product_id": best_affordable[0],
                        "name": best_affordable[1]["name"],
                        "price": best_affordable[1]["price"],
                        "category": category
                    })
                    optimized_cost += best_affordable[1]["price"]
        
        savings = current_cost - optimized_cost
        
        return {
            "optimization_needed": True,
            "current_cost": current_cost,
            "optimized_cost": optimized_cost,
            "budget_limit": budget_limit,
            "savings": savings,
            "optimized_routine": optimized_routine,
            "message": f"Rutina optimizada. Ahorro: ${savings:.2f}"
        }

