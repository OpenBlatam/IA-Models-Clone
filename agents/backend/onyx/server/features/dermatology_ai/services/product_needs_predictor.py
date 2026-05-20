"""
Sistema de predicción de necesidades de productos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid


@dataclass
class ProductNeed:
    """Necesidad de producto"""
    id: str
    user_id: str
    product_category: str
    product_name: Optional[str] = None
    priority: int  # 1-5, 1 = más urgente
    reason: str
    estimated_cost: Optional[float] = None
    urgency_days: int = 30
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_category": self.product_category,
            "product_name": self.product_name,
            "priority": self.priority,
            "reason": self.reason,
            "estimated_cost": self.estimated_cost,
            "urgency_days": self.urgency_days,
            "created_at": self.created_at
        }


class ProductNeedsPredictor:
    """Sistema de predicción de necesidades de productos"""
    
    def __init__(self):
        """Inicializa el predictor"""
        self.needs: Dict[str, List[ProductNeed]] = {}  # user_id -> [needs]
    
    def predict_needs(self, user_id: str, current_products: List[Dict],
                     skin_analysis: Dict, usage_history: List[Dict]) -> List[ProductNeed]:
        """Predice necesidades de productos"""
        needs = []
        
        # Analizar productos actuales
        categories_covered = {p.get("category") for p in current_products if p.get("category")}
        
        # Verificar categorías esenciales
        essential_categories = ["cleanser", "moisturizer", "sunscreen"]
        missing_categories = [cat for cat in essential_categories if cat not in categories_covered]
        
        for category in missing_categories:
            need = ProductNeed(
                id=str(uuid.uuid4()),
                user_id=user_id,
                product_category=category,
                priority=1,
                reason=f"Categoría esencial faltante: {category}",
                urgency_days=7
            )
            needs.append(need)
        
        # Analizar análisis de piel
        scores = skin_analysis.get("quality_scores", {})
        hydration = scores.get("hydration_score", 0)
        
        if hydration < 50:
            need = ProductNeed(
                id=str(uuid.uuid4()),
                user_id=user_id,
                product_category="serum",
                product_name="Hidratante con Ácido Hialurónico",
                priority=1,
                reason=f"Hidratación baja ({hydration}%)",
                estimated_cost=25.0,
                urgency_days=14
            )
            needs.append(need)
        
        # Analizar uso de productos
        if usage_history:
            low_usage_products = [
                p for p in current_products
                if self._get_usage_frequency(p.get("id"), usage_history) < 0.3
            ]
            
            for product in low_usage_products[:2]:
                need = ProductNeed(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    product_category=product.get("category", "unknown"),
                    product_name=product.get("name"),
                    priority=3,
                    reason="Producto con bajo uso - considerar reemplazo",
                    urgency_days=30
                )
                needs.append(need)
        
        # Guardar necesidades
        if user_id not in self.needs:
            self.needs[user_id] = []
        self.needs[user_id].extend(needs)
        
        return needs
    
    def _get_usage_frequency(self, product_id: str, usage_history: List[Dict]) -> float:
        """Calcula frecuencia de uso"""
        product_usages = [u for u in usage_history if u.get("product_id") == product_id]
        if not product_usages:
            return 0.0
        
        # Calcular frecuencia en últimos 30 días
        cutoff = datetime.now() - timedelta(days=30)
        recent_usages = [
            u for u in product_usages
            if datetime.fromisoformat(u.get("date", datetime.now().isoformat())) >= cutoff
        ]
        
        return len(recent_usages) / 30.0
    
    def get_user_needs(self, user_id: str, priority: Optional[int] = None) -> List[ProductNeed]:
        """Obtiene necesidades del usuario"""
        user_needs = self.needs.get(user_id, [])
        
        if priority:
            user_needs = [n for n in user_needs if n.priority == priority]
        
        user_needs.sort(key=lambda x: (x.priority, x.urgency_days))
        return user_needs






