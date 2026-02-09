"""
Sistema de análisis de tendencias de mercado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class MarketTrend:
    """Tendencia de mercado"""
    category: str
    trend_direction: str  # "rising", "falling", "stable"
    popularity_score: float
    growth_rate: float
    top_products: List[str]
    insights: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "category": self.category,
            "trend_direction": self.trend_direction,
            "popularity_score": self.popularity_score,
            "growth_rate": self.growth_rate,
            "top_products": self.top_products,
            "insights": self.insights,
            "created_at": self.created_at
        }


class MarketTrendsAnalyzer:
    """Sistema de análisis de tendencias de mercado"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.trend_data: Dict[str, List[float]] = {}  # category -> [scores over time]
    
    def analyze_category_trend(self, category: str, historical_data: List[Dict]) -> MarketTrend:
        """
        Analiza tendencia de una categoría
        
        Args:
            category: Categoría a analizar
            historical_data: Datos históricos de popularidad
            
        Returns:
            Tendencia de mercado
        """
        # Extraer scores de popularidad
        scores = [d.get("popularity_score", 0) for d in historical_data if d.get("category") == category]
        
        if len(scores) < 2:
            # Datos insuficientes
            return MarketTrend(
                category=category,
                trend_direction="stable",
                popularity_score=0.0,
                growth_rate=0.0,
                top_products=[],
                insights=["Datos insuficientes para análisis"]
            )
        
        # Calcular métricas
        current_score = scores[-1]
        previous_score = scores[-2] if len(scores) > 1 else scores[0]
        avg_score = statistics.mean(scores)
        
        # Dirección de tendencia
        growth_rate = ((current_score - previous_score) / previous_score * 100) if previous_score > 0 else 0.0
        
        if growth_rate > 5:
            trend_direction = "rising"
        elif growth_rate < -5:
            trend_direction = "falling"
        else:
            trend_direction = "stable"
        
        # Productos top (simulado - en producción vendría de datos reales)
        top_products = self._get_top_products(category, historical_data)
        
        # Insights
        insights = self._generate_insights(category, trend_direction, growth_rate, current_score)
        
        return MarketTrend(
            category=category,
            trend_direction=trend_direction,
            popularity_score=current_score,
            growth_rate=growth_rate,
            top_products=top_products,
            insights=insights
        )
    
    def _get_top_products(self, category: str, historical_data: List[Dict]) -> List[str]:
        """Obtiene productos top de una categoría"""
        # Simulado - en producción vendría de base de datos real
        category_products = {
            "cleanser": ["Cleanser A", "Cleanser B", "Cleanser C"],
            "moisturizer": ["Moisturizer X", "Moisturizer Y", "Moisturizer Z"],
            "serum": ["Serum 1", "Serum 2", "Serum 3"],
            "sunscreen": ["SPF 50+", "SPF 30", "Mineral SPF"]
        }
        
        return category_products.get(category, [])
    
    def _generate_insights(self, category: str, trend_direction: str,
                          growth_rate: float, popularity_score: float) -> List[str]:
        """Genera insights sobre tendencia"""
        insights = []
        
        if trend_direction == "rising":
            insights.append(f"La categoría {category} está en crecimiento ({growth_rate:.1f}%)")
            insights.append("Es un buen momento para explorar productos en esta categoría")
        elif trend_direction == "falling":
            insights.append(f"La categoría {category} está en declive ({growth_rate:.1f}%)")
            insights.append("Considera alternativas o espera a que la tendencia cambie")
        else:
            insights.append(f"La categoría {category} se mantiene estable")
        
        if popularity_score > 80:
            insights.append("Alta popularidad en el mercado")
        elif popularity_score < 40:
            insights.append("Baja popularidad - categoría emergente o nicho")
        
        return insights
    
    def compare_categories(self, categories: List[str],
                         historical_data: List[Dict]) -> Dict[str, MarketTrend]:
        """Compara múltiples categorías"""
        trends = {}
        
        for category in categories:
            trends[category] = self.analyze_category_trend(category, historical_data)
        
        return trends
    
    def get_recommended_categories(self, user_preferences: Dict,
                                  historical_data: List[Dict]) -> List[str]:
        """Obtiene categorías recomendadas basadas en preferencias"""
        # Analizar todas las categorías
        all_categories = set(d.get("category") for d in historical_data if d.get("category"))
        
        category_scores = {}
        for category in all_categories:
            trend = self.analyze_category_trend(category, historical_data)
            
            # Score basado en tendencia y popularidad
            score = trend.popularity_score
            if trend.trend_direction == "rising":
                score *= 1.2  # Bonus por tendencia creciente
            
            category_scores[category] = score
        
        # Ordenar por score
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filtrar por preferencias del usuario si existen
        preferred = user_preferences.get("preferred_categories", [])
        if preferred:
            sorted_categories = [
                (cat, score) for cat, score in sorted_categories
                if cat in preferred or not preferred
            ]
        
        return [cat for cat, _ in sorted_categories[:5]]  # Top 5






