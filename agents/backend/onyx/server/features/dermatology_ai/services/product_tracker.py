"""
Sistema de seguimiento de productos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid


@dataclass
class ProductUsage:
    """Uso de producto"""
    id: str
    user_id: str
    product_id: str
    product_name: str
    usage_date: str
    frequency: str  # "daily", "weekly", "as_needed"
    effectiveness_rating: Optional[int] = None  # 1-5
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
            "product_id": self.product_id,
            "product_name": self.product_name,
            "usage_date": self.usage_date,
            "frequency": self.frequency,
            "effectiveness_rating": self.effectiveness_rating,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class ProductInsight:
    """Insight sobre producto"""
    product_id: str
    product_name: str
    total_uses: int
    average_rating: float
    last_used: str
    days_since_last_use: int
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "total_uses": self.total_uses,
            "average_rating": self.average_rating,
            "last_used": self.last_used,
            "days_since_last_use": self.days_since_last_use,
            "recommendation": self.recommendation
        }


class ProductTracker:
    """Sistema de seguimiento de productos"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.usage_records: Dict[str, List[ProductUsage]] = {}  # user_id -> [usages]
    
    def record_usage(self, user_id: str, product_id: str, product_name: str,
                    usage_date: str, frequency: str = "daily",
                    effectiveness_rating: Optional[int] = None,
                    notes: Optional[str] = None) -> ProductUsage:
        """Registra uso de producto"""
        usage = ProductUsage(
            id=str(uuid.uuid4()),
            user_id=user_id,
            product_id=product_id,
            product_name=product_name,
            usage_date=usage_date,
            frequency=frequency,
            effectiveness_rating=effectiveness_rating,
            notes=notes
        )
        
        if user_id not in self.usage_records:
            self.usage_records[user_id] = []
        
        self.usage_records[user_id].append(usage)
        return usage
    
    def get_product_insights(self, user_id: str, product_id: str) -> Optional[ProductInsight]:
        """Obtiene insights de un producto"""
        user_usages = self.usage_records.get(user_id, [])
        product_usages = [u for u in user_usages if u.product_id == product_id]
        
        if not product_usages:
            return None
        
        # Calcular estadísticas
        total_uses = len(product_usages)
        
        ratings = [u.effectiveness_rating for u in product_usages if u.effectiveness_rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        # Último uso
        sorted_usages = sorted(product_usages, key=lambda x: x.usage_date, reverse=True)
        last_usage = sorted_usages[0]
        last_used = last_usage.usage_date
        
        # Días desde último uso
        last_date = datetime.fromisoformat(last_used)
        days_since = (datetime.now() - last_date).days
        
        # Generar recomendación
        recommendation = self._generate_recommendation(avg_rating, days_since, total_uses)
        
        return ProductInsight(
            product_id=product_id,
            product_name=product_usages[0].product_name,
            total_uses=total_uses,
            average_rating=avg_rating,
            last_used=last_used,
            days_since_last_use=days_since,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, avg_rating: float, days_since: int,
                                total_uses: int) -> str:
        """Genera recomendación sobre producto"""
        if avg_rating >= 4.0:
            return "Producto altamente efectivo. Continúa usándolo regularmente."
        elif avg_rating >= 3.0:
            return "Producto moderadamente efectivo. Considera ajustar frecuencia o combinación."
        elif avg_rating > 0:
            return "Producto con baja efectividad. Considera reemplazarlo o consultar alternativas."
        else:
            return "No hay suficientes datos de efectividad. Continúa usando y registra resultados."
    
    def get_user_product_summary(self, user_id: str) -> Dict:
        """Obtiene resumen de productos del usuario"""
        user_usages = self.usage_records.get(user_id, [])
        
        # Productos únicos
        unique_products = {}
        for usage in user_usages:
            if usage.product_id not in unique_products:
                unique_products[usage.product_id] = {
                    "name": usage.product_name,
                    "uses": 0,
                    "last_used": None
                }
            unique_products[usage.product_id]["uses"] += 1
            
            if unique_products[usage.product_id]["last_used"] is None or \
               usage.usage_date > unique_products[usage.product_id]["last_used"]:
                unique_products[usage.product_id]["last_used"] = usage.usage_date
        
        return {
            "total_products": len(unique_products),
            "total_uses": len(user_usages),
            "products": unique_products
        }
    
    def get_usage_history(self, user_id: str, product_id: Optional[str] = None,
                         days: int = 30) -> List[ProductUsage]:
        """Obtiene historial de uso"""
        user_usages = self.usage_records.get(user_id, [])
        
        # Filtrar por producto si se especifica
        if product_id:
            user_usages = [u for u in user_usages if u.product_id == product_id]
        
        # Filtrar por fecha
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        filtered = [u for u in user_usages if u.usage_date >= cutoff_date]
        
        # Ordenar por fecha
        filtered.sort(key=lambda x: x.usage_date, reverse=True)
        
        return filtered






