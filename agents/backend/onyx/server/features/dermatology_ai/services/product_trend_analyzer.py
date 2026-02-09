"""
Sistema de análisis de tendencias de productos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class ProductTrend:
    """Tendencia de producto"""
    product_id: str
    product_name: str
    category: str
    popularity_score: float  # 0-100
    trend_direction: str  # "rising", "falling", "stable"
    growth_rate: float  # Porcentaje
    user_satisfaction: float
    price_trend: str  # "increasing", "decreasing", "stable"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "category": self.category,
            "popularity_score": self.popularity_score,
            "trend_direction": self.trend_direction,
            "growth_rate": self.growth_rate,
            "user_satisfaction": self.user_satisfaction,
            "price_trend": self.price_trend
        }


@dataclass
class CategoryTrend:
    """Tendencia de categoría"""
    category: str
    trending_products: List[ProductTrend]
    average_satisfaction: float
    market_share_change: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "category": self.category,
            "trending_products": [p.to_dict() for p in self.trending_products],
            "average_satisfaction": self.average_satisfaction,
            "market_share_change": self.market_share_change,
            "recommendations": self.recommendations
        }


class ProductTrendAnalyzer:
    """Sistema de análisis de tendencias de productos"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.product_data: Dict[str, List[Dict]] = {}  # product_id -> [data_points]
    
    def add_product_data(self, product_id: str, product_name: str, category: str,
                        popularity: float, satisfaction: float, price: float):
        """Agrega dato de producto"""
        if product_id not in self.product_data:
            self.product_data[product_id] = []
        
        self.product_data[product_id].append({
            "product_name": product_name,
            "category": category,
            "popularity": popularity,
            "satisfaction": satisfaction,
            "price": price,
            "timestamp": datetime.now().isoformat()
        })
    
    def analyze_product_trend(self, product_id: str) -> Optional[ProductTrend]:
        """Analiza tendencia de un producto"""
        data_points = self.product_data.get(product_id)
        
        if not data_points or len(data_points) < 2:
            return None
        
        # Ordenar por timestamp
        data_points.sort(key=lambda x: x["timestamp"])
        
        # Calcular métricas
        recent_popularity = data_points[-1]["popularity"]
        previous_popularity = data_points[-2]["popularity"] if len(data_points) > 1 else recent_popularity
        
        growth_rate = ((recent_popularity - previous_popularity) / previous_popularity * 100) if previous_popularity > 0 else 0.0
        
        if growth_rate > 5:
            trend_direction = "rising"
        elif growth_rate < -5:
            trend_direction = "falling"
        else:
            trend_direction = "stable"
        
        # Satisfacción promedio
        satisfaction = statistics.mean([d["satisfaction"] for d in data_points])
        
        # Tendencia de precio
        recent_price = data_points[-1]["price"]
        previous_price = data_points[-2]["price"] if len(data_points) > 1 else recent_price
        
        if recent_price > previous_price * 1.05:
            price_trend = "increasing"
        elif recent_price < previous_price * 0.95:
            price_trend = "decreasing"
        else:
            price_trend = "stable"
        
        return ProductTrend(
            product_id=product_id,
            product_name=data_points[-1]["product_name"],
            category=data_points[-1]["category"],
            popularity_score=recent_popularity,
            trend_direction=trend_direction,
            growth_rate=growth_rate,
            user_satisfaction=satisfaction,
            price_trend=price_trend
        )
    
    def analyze_category_trends(self, category: str) -> Optional[CategoryTrend]:
        """Analiza tendencias de una categoría"""
        category_products = [
            (pid, data) for pid, data in self.product_data.items()
            if data and data[-1].get("category") == category
        ]
        
        if not category_products:
            return None
        
        trending_products = []
        satisfactions = []
        
        for product_id, _ in category_products:
            trend = self.analyze_product_trend(product_id)
            if trend:
                trending_products.append(trend)
                satisfactions.append(trend.user_satisfaction)
        
        # Ordenar por popularidad
        trending_products.sort(key=lambda x: x.popularity_score, reverse=True)
        
        avg_satisfaction = statistics.mean(satisfactions) if satisfactions else 0.0
        
        # Recomendaciones
        recommendations = []
        if trending_products:
            top_product = trending_products[0]
            if top_product.trend_direction == "rising":
                recommendations.append(f"{top_product.product_name} está en tendencia creciente")
            if top_product.user_satisfaction > 4.5:
                recommendations.append(f"{top_product.product_name} tiene alta satisfacción de usuarios")
        
        return CategoryTrend(
            category=category,
            trending_products=trending_products[:5],  # Top 5
            average_satisfaction=avg_satisfaction,
            market_share_change=0.0,  # Simulado
            recommendations=recommendations
        )
    
    def get_trending_products(self, limit: int = 10) -> List[ProductTrend]:
        """Obtiene productos en tendencia"""
        all_trends = []
        
        for product_id in self.product_data.keys():
            trend = self.analyze_product_trend(product_id)
            if trend and trend.trend_direction == "rising":
                all_trends.append(trend)
        
        # Ordenar por growth rate
        all_trends.sort(key=lambda x: x.growth_rate, reverse=True)
        
        return all_trends[:limit]






