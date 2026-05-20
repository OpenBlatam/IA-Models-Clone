"""
Sistema de seguimiento de presupuesto
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid


@dataclass
class BudgetEntry:
    """Entrada de presupuesto"""
    id: str
    user_id: str
    product_name: str
    product_id: Optional[str] = None
    amount: float
    category: str  # "cleanser", "moisturizer", "serum", etc.
    purchase_date: str
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_name": self.product_name,
            "product_id": self.product_id,
            "amount": self.amount,
            "category": self.category,
            "purchase_date": self.purchase_date,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class BudgetSummary:
    """Resumen de presupuesto"""
    user_id: str
    period: str  # "month", "year", "all_time"
    total_spent: float
    budget_limit: Optional[float] = None
    remaining_budget: Optional[float] = None
    by_category: Dict[str, float] = None
    top_products: List[Dict] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.by_category is None:
            self.by_category = {}
        if self.top_products is None:
            self.top_products = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "period": self.period,
            "total_spent": self.total_spent,
            "budget_limit": self.budget_limit,
            "remaining_budget": self.remaining_budget,
            "by_category": self.by_category,
            "top_products": self.top_products,
            "created_at": self.created_at
        }


class BudgetTracker:
    """Sistema de seguimiento de presupuesto"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.entries: Dict[str, List[BudgetEntry]] = {}  # user_id -> [entries]
        self.budget_limits: Dict[str, Dict] = {}  # user_id -> {period: limit}
    
    def add_entry(self, user_id: str, product_name: str, amount: float,
                 category: str, purchase_date: str,
                 product_id: Optional[str] = None,
                 notes: Optional[str] = None) -> BudgetEntry:
        """Agrega entrada de presupuesto"""
        entry = BudgetEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            product_name=product_name,
            product_id=product_id,
            amount=amount,
            category=category,
            purchase_date=purchase_date,
            notes=notes
        )
        
        if user_id not in self.entries:
            self.entries[user_id] = []
        
        self.entries[user_id].append(entry)
        return entry
    
    def set_budget_limit(self, user_id: str, period: str, limit: float):
        """Establece límite de presupuesto"""
        if user_id not in self.budget_limits:
            self.budget_limits[user_id] = {}
        
        self.budget_limits[user_id][period] = limit
    
    def get_budget_summary(self, user_id: str, period: str = "month") -> BudgetSummary:
        """Obtiene resumen de presupuesto"""
        user_entries = self.entries.get(user_id, [])
        
        # Filtrar por período
        now = datetime.now()
        if period == "month":
            cutoff = now - timedelta(days=30)
        elif period == "year":
            cutoff = now - timedelta(days=365)
        else:  # all_time
            cutoff = datetime.min
        
        filtered_entries = [
            e for e in user_entries
            if datetime.fromisoformat(e.purchase_date) >= cutoff
        ]
        
        # Calcular total
        total_spent = sum(e.amount for e in filtered_entries)
        
        # Por categoría
        by_category = {}
        for entry in filtered_entries:
            by_category[entry.category] = by_category.get(entry.category, 0) + entry.amount
        
        # Top productos
        product_totals = {}
        for entry in filtered_entries:
            key = entry.product_name
            product_totals[key] = product_totals.get(key, 0) + entry.amount
        
        top_products = sorted(
            [{"name": k, "total": v} for k, v in product_totals.items()],
            key=lambda x: x["total"],
            reverse=True
        )[:5]
        
        # Límite de presupuesto
        budget_limit = None
        remaining_budget = None
        
        if user_id in self.budget_limits and period in self.budget_limits[user_id]:
            budget_limit = self.budget_limits[user_id][period]
            remaining_budget = budget_limit - total_spent
        
        return BudgetSummary(
            user_id=user_id,
            period=period,
            total_spent=total_spent,
            budget_limit=budget_limit,
            remaining_budget=remaining_budget,
            by_category=by_category,
            top_products=top_products
        )
    
    def get_user_entries(self, user_id: str, limit: int = 100) -> List[BudgetEntry]:
        """Obtiene entradas del usuario"""
        user_entries = self.entries.get(user_id, [])
        user_entries.sort(key=lambda x: x.purchase_date, reverse=True)
        return user_entries[:limit]






