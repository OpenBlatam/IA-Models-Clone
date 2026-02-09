"""
Sistema de recomendaciones basadas en presupuesto mensual
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BudgetProfile:
    """Perfil de presupuesto"""
    user_id: str
    monthly_budget: float
    currency: str  # "USD", "EUR", "MXN", etc.
    priority_areas: List[str]  # "anti_aging", "acne", "hydration", "brightening"
    willingness_to_splurge: str  # "low", "medium", "high"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "monthly_budget": self.monthly_budget,
            "currency": self.currency,
            "priority_areas": self.priority_areas,
            "willingness_to_splurge": self.willingness_to_splurge
        }


@dataclass
class BudgetRecommendation:
    """Recomendación basada en presupuesto"""
    product_category: str
    product_name: str
    price: float
    price_range: str  # "budget", "mid_range", "premium"
    value_score: float  # 0-1
    priority: int
    reasoning: str
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_category": self.product_category,
            "product_name": self.product_name,
            "price": self.price,
            "price_range": self.price_range,
            "value_score": self.value_score,
            "priority": self.priority,
            "reasoning": self.reasoning
        }


@dataclass
class BudgetRoutine:
    """Rutina basada en presupuesto"""
    user_id: str
    total_monthly_cost: float
    products: List[BudgetRecommendation]
    budget_allocation: Dict[str, float]
    savings_tips: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "total_monthly_cost": self.total_monthly_cost,
            "products": [p.to_dict() for p in self.products],
            "budget_allocation": self.budget_allocation,
            "savings_tips": self.savings_tips
        }


class MonthlyBudgetRecommendations:
    """Sistema de recomendaciones basadas en presupuesto mensual"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, BudgetProfile] = {}  # user_id -> profile
    
    def create_budget_profile(self, user_id: str, monthly_budget: float,
                            currency: str, priority_areas: List[str],
                            willingness_to_splurge: str) -> BudgetProfile:
        """Crea perfil de presupuesto"""
        profile = BudgetProfile(
            user_id=user_id,
            monthly_budget=monthly_budget,
            currency=currency,
            priority_areas=priority_areas,
            willingness_to_splurge=willingness_to_splurge
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def generate_budget_routine(self, user_id: str) -> BudgetRoutine:
        """Genera rutina basada en presupuesto"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            raise ValueError("Perfil de presupuesto no encontrado")
        
        products = []
        total_cost = 0.0
        
        # Productos básicos esenciales
        essential_products = [
            BudgetRecommendation(
                product_category="Cleanser",
                product_name="Gentle Cleanser",
                price=15.0,
                price_range="budget",
                value_score=0.9,
                priority=1,
                reasoning="Esencial para cualquier rutina"
            ),
            BudgetRecommendation(
                product_category="Moisturizer",
                product_name="Basic Moisturizer",
                price=20.0,
                price_range="budget",
                value_score=0.85,
                priority=1,
                reasoning="Hidratación básica necesaria"
            ),
            BudgetRecommendation(
                product_category="Sunscreen",
                product_name="SPF 30 Sunscreen",
                price=18.0,
                price_range="budget",
                value_score=0.95,
                priority=1,
                reasoning="Protección solar es crítica"
            )
        ]
        
        products.extend(essential_products)
        total_cost += sum(p.price for p in essential_products)
        
        # Productos según prioridades
        remaining_budget = profile.monthly_budget - total_cost
        
        if "anti_aging" in profile.priority_areas and remaining_budget > 30:
            products.append(BudgetRecommendation(
                product_category="Serum",
                product_name="Retinol Serum" if remaining_budget > 50 else "Vitamin C Serum",
                price=35.0 if remaining_budget > 50 else 25.0,
                price_range="mid_range",
                value_score=0.8,
                priority=2,
                reasoning="Anti-envejecimiento según presupuesto"
            ))
            total_cost += products[-1].price
            remaining_budget -= products[-1].price
        
        if "acne" in profile.priority_areas and remaining_budget > 20:
            products.append(BudgetRecommendation(
                product_category="Treatment",
                product_name="Salicylic Acid Treatment",
                price=22.0,
                price_range="budget",
                value_score=0.85,
                priority=2,
                reasoning="Tratamiento para acné"
            ))
            total_cost += products[-1].price
            remaining_budget -= products[-1].price
        
        # Ajustar según presupuesto
        if total_cost > profile.monthly_budget:
            # Priorizar productos esenciales
            products = [p for p in products if p.priority == 1]
            total_cost = sum(p.price for p in products)
        
        # Asignación de presupuesto
        budget_allocation = {
            "essentials": sum(p.price for p in products if p.priority == 1),
            "treatments": sum(p.price for p in products if p.priority == 2),
            "remaining": max(0, profile.monthly_budget - total_cost)
        }
        
        # Tips de ahorro
        savings_tips = []
        if total_cost > profile.monthly_budget * 0.8:
            savings_tips.append("Considera comprar productos en tamaños grandes para mejor valor")
            savings_tips.append("Busca ofertas y promociones")
        
        if remaining_budget > 0:
            savings_tips.append(f"Tienes ${remaining_budget:.2f} disponibles para productos adicionales")
        
        return BudgetRoutine(
            user_id=user_id,
            total_monthly_cost=total_cost,
            products=products,
            budget_allocation=budget_allocation,
            savings_tips=savings_tips
        )






