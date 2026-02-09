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
    currency: str = "USD"
    priority_areas: List[str] = None  # "acne", "anti_aging", "hydration", etc.
    preferred_brands: List[str] = None
    
    def __post_init__(self):
        if self.priority_areas is None:
            self.priority_areas = []
        if self.preferred_brands is None:
            self.preferred_brands = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "monthly_budget": self.monthly_budget,
            "currency": self.currency,
            "priority_areas": self.priority_areas,
            "preferred_brands": self.preferred_brands
        }


@dataclass
class BudgetProduct:
    """Producto con información de presupuesto"""
    product_id: str
    name: str
    category: str
    price: float
    size: str
    price_per_ml: float
    estimated_duration_days: int
    value_score: float  # 0.0 to 1.0
    priority: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "size": self.size,
            "price_per_ml": self.price_per_ml,
            "estimated_duration_days": self.estimated_duration_days,
            "value_score": self.value_score,
            "priority": self.priority
        }


@dataclass
class BudgetRoutine:
    """Rutina optimizada para presupuesto"""
    routine_id: str
    user_id: str
    total_monthly_cost: float
    products: List[BudgetProduct]
    savings_tips: List[str]
    alternative_products: List[BudgetProduct]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "routine_id": self.routine_id,
            "user_id": self.user_id,
            "total_monthly_cost": self.total_monthly_cost,
            "products": [p.to_dict() for p in self.products],
            "savings_tips": self.savings_tips,
            "alternative_products": [p.to_dict() for p in self.alternative_products],
            "created_at": self.created_at
        }


class BudgetRecommendations:
    """Sistema de recomendaciones basadas en presupuesto"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, BudgetProfile] = {}
        self.product_db: Dict[str, BudgetProduct] = {}
        self.routines: Dict[str, BudgetRoutine] = {}
    
    def create_budget_profile(self, user_id: str, monthly_budget: float,
                             currency: str = "USD", priority_areas: Optional[List[str]] = None,
                             preferred_brands: Optional[List[str]] = None) -> BudgetProfile:
        """Crea perfil de presupuesto"""
        profile = BudgetProfile(
            user_id=user_id,
            monthly_budget=monthly_budget,
            currency=currency,
            priority_areas=priority_areas or [],
            preferred_brands=preferred_brands or []
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def register_product(self, product_id: str, name: str, category: str,
                        price: float, size: str, estimated_duration_days: int,
                        value_score: float = 0.5) -> BudgetProduct:
        """Registra producto en la base de datos"""
        # Calcular precio por ml
        size_ml = self._parse_size_to_ml(size)
        price_per_ml = price / size_ml if size_ml > 0 else 0.0
        
        product = BudgetProduct(
            product_id=product_id,
            name=name,
            category=category,
            price=price,
            size=size,
            price_per_ml=price_per_ml,
            estimated_duration_days=estimated_duration_days,
            value_score=value_score,
            priority=1
        )
        
        self.product_db[product_id] = product
        return product
    
    def generate_budget_routine(self, user_id: str, required_categories: List[str]) -> BudgetRoutine:
        """Genera rutina optimizada para presupuesto"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            raise ValueError("Perfil de presupuesto no encontrado")
        
        selected_products = []
        total_cost = 0.0
        
        # Seleccionar productos por categoría
        for category in required_categories:
            category_products = [
                p for p in self.product_db.values()
                if p.category == category
            ]
            
            if not category_products:
                continue
            
            # Ordenar por valor (value_score / price_per_ml)
            category_products.sort(
                key=lambda p: (p.value_score / p.price_per_ml) if p.price_per_ml > 0 else 0,
                reverse=True
            )
            
            # Seleccionar el mejor producto que quepa en el presupuesto
            for product in category_products:
                monthly_cost = (product.price / product.estimated_duration_days) * 30
                
                if total_cost + monthly_cost <= profile.monthly_budget:
                    selected_products.append(product)
                    total_cost += monthly_cost
                    break
        
        # Si no se alcanzó el presupuesto, buscar alternativas más económicas
        alternative_products = []
        savings_tips = []
        
        if total_cost < profile.monthly_budget * 0.7:
            savings_tips.append("Tienes espacio en tu presupuesto para productos premium")
        elif total_cost > profile.monthly_budget:
            savings_tips.append("Considera productos más económicos o tamaños más grandes")
            
            # Buscar alternativas más baratas
            for product in selected_products:
                alternatives = [
                    p for p in self.product_db.values()
                    if p.category == product.category and p.price < product.price
                ]
                if alternatives:
                    alternative_products.extend(alternatives[:2])
        
        # Consejos de ahorro
        if not savings_tips:
            savings_tips.append("Tu rutina está optimizada para tu presupuesto")
        
        savings_tips.append("Compra tamaños más grandes cuando sea posible para mejor valor")
        savings_tips.append("Busca ofertas y promociones en productos esenciales")
        
        routine = BudgetRoutine(
            routine_id=str(uuid.uuid4()),
            user_id=user_id,
            total_monthly_cost=total_cost,
            products=selected_products,
            savings_tips=savings_tips,
            alternative_products=alternative_products[:5]
        )
        
        self.routines[routine.routine_id] = routine
        return routine
    
    def _parse_size_to_ml(self, size: str) -> float:
        """Parsea tamaño a ml"""
        size_lower = size.lower()
        
        # Intentar extraer número
        import re
        numbers = re.findall(r'\d+\.?\d*', size_lower)
        
        if not numbers:
            return 0.0
        
        value = float(numbers[0])
        
        # Convertir según unidad
        if 'ml' in size_lower or 'mililitro' in size_lower:
            return value
        elif 'l' in size_lower and 'ml' not in size_lower:
            return value * 1000
        elif 'oz' in size_lower or 'ounce' in size_lower:
            return value * 29.5735  # fl oz a ml
        elif 'fl' in size_lower:
            return value * 29.5735
        
        return value  # Asumir ml por defecto


