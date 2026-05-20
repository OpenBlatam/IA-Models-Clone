"""
Sistema de seguimiento de efectividad de productos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class ProductUsage:
    """Uso de producto"""
    id: str
    user_id: str
    product_name: str
    product_category: str
    start_date: str
    end_date: Optional[str] = None
    frequency: str  # "daily", "weekly", "as_needed"
    application_area: Optional[str] = None
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
            "product_category": self.product_category,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "frequency": self.frequency,
            "application_area": self.application_area,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class EffectivenessRating:
    """Calificación de efectividad"""
    id: str
    product_usage_id: str
    user_id: str
    rating_date: str
    effectiveness_score: int  # 1-10
    improvement_areas: List[str] = None
    side_effects: List[str] = None
    would_recommend: bool = False
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.improvement_areas is None:
            self.improvement_areas = []
        if self.side_effects is None:
            self.side_effects = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "product_usage_id": self.product_usage_id,
            "user_id": self.user_id,
            "rating_date": self.rating_date,
            "effectiveness_score": self.effectiveness_score,
            "improvement_areas": self.improvement_areas,
            "side_effects": self.side_effects,
            "would_recommend": self.would_recommend,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class ProductEffectivenessReport:
    """Reporte de efectividad de producto"""
    product_name: str
    average_score: float
    total_ratings: int
    recommendation_rate: float
    improvement_areas: List[str]
    side_effects_frequency: Dict[str, int]
    usage_duration_days: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_name": self.product_name,
            "average_score": self.average_score,
            "total_ratings": self.total_ratings,
            "recommendation_rate": self.recommendation_rate,
            "improvement_areas": self.improvement_areas,
            "side_effects_frequency": self.side_effects_frequency,
            "usage_duration_days": self.usage_duration_days
        }


class ProductEffectivenessTracker:
    """Sistema de seguimiento de efectividad de productos"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.usages: Dict[str, List[ProductUsage]] = {}
        self.ratings: Dict[str, List[EffectivenessRating]] = {}
    
    def add_product_usage(self, user_id: str, product_name: str, product_category: str,
                         start_date: str, frequency: str,
                         application_area: Optional[str] = None,
                         notes: Optional[str] = None) -> ProductUsage:
        """Agrega uso de producto"""
        usage = ProductUsage(
            id=str(uuid.uuid4()),
            user_id=user_id,
            product_name=product_name,
            product_category=product_category,
            start_date=start_date,
            frequency=frequency,
            application_area=application_area,
            notes=notes
        )
        
        if user_id not in self.usages:
            self.usages[user_id] = []
        
        self.usages[user_id].append(usage)
        return usage
    
    def add_effectiveness_rating(self, user_id: str, product_usage_id: str,
                                rating_date: str, effectiveness_score: int,
                                improvement_areas: Optional[List[str]] = None,
                                side_effects: Optional[List[str]] = None,
                                would_recommend: bool = False,
                                notes: Optional[str] = None) -> EffectivenessRating:
        """Agrega calificación de efectividad"""
        rating = EffectivenessRating(
            id=str(uuid.uuid4()),
            product_usage_id=product_usage_id,
            user_id=user_id,
            rating_date=rating_date,
            effectiveness_score=effectiveness_score,
            improvement_areas=improvement_areas or [],
            side_effects=side_effects or [],
            would_recommend=would_recommend,
            notes=notes
        )
        
        if product_usage_id not in self.ratings:
            self.ratings[product_usage_id] = []
        
        self.ratings[product_usage_id].append(rating)
        return rating
    
    def generate_effectiveness_report(self, user_id: str, product_name: str) -> Optional[ProductEffectivenessReport]:
        """Genera reporte de efectividad"""
        user_usages = self.usages.get(user_id, [])
        product_usages = [u for u in user_usages if u.product_name == product_name]
        
        if not product_usages:
            return None
        
        # Obtener todas las calificaciones para este producto
        all_ratings = []
        for usage in product_usages:
            usage_ratings = self.ratings.get(usage.id, [])
            all_ratings.extend(usage_ratings)
        
        if not all_ratings:
            return ProductEffectivenessReport(
                product_name=product_name,
                average_score=0.0,
                total_ratings=0,
                recommendation_rate=0.0,
                improvement_areas=[],
                side_effects_frequency={}
            )
        
        # Calcular estadísticas
        scores = [r.effectiveness_score for r in all_ratings]
        avg_score = statistics.mean(scores)
        
        recommendations = [r for r in all_ratings if r.would_recommend]
        recommendation_rate = len(recommendations) / len(all_ratings) if all_ratings else 0.0
        
        # Áreas de mejora más comunes
        improvement_areas = []
        for rating in all_ratings:
            improvement_areas.extend(rating.improvement_areas)
        
        improvement_counts = {}
        for area in improvement_areas:
            improvement_counts[area] = improvement_counts.get(area, 0) + 1
        
        top_improvements = sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_improvement_areas = [area for area, _ in top_improvements]
        
        # Efectos secundarios
        side_effects = []
        for rating in all_ratings:
            side_effects.extend(rating.side_effects)
        
        side_effects_freq = {}
        for effect in side_effects:
            side_effects_freq[effect] = side_effects_freq.get(effect, 0) + 1
        
        # Duración de uso
        usage_duration = None
        if product_usages:
            latest_usage = max(product_usages, key=lambda u: u.start_date)
            if latest_usage.end_date:
                start = datetime.fromisoformat(latest_usage.start_date)
                end = datetime.fromisoformat(latest_usage.end_date)
                usage_duration = (end - start).days
        
        return ProductEffectivenessReport(
            product_name=product_name,
            average_score=avg_score,
            total_ratings=len(all_ratings),
            recommendation_rate=recommendation_rate,
            improvement_areas=top_improvement_areas,
            side_effects_frequency=side_effects_freq,
            usage_duration_days=usage_duration
        )






