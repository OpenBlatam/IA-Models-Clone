"""
Intelligent Recommendation System
=================================
Sistema de recomendaciones inteligentes basado en ML
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class UserPreference:
    """Preferencias del usuario"""
    user_id: str
    preferred_categories: List[str]
    preferred_colors: List[str]
    preferred_styles: List[str]
    usage_history: List[Dict]
    ratings: Dict[str, float]  # item_id -> rating


@dataclass
class Recommendation:
    """Recomendación"""
    item_id: str
    item_type: str  # 'template', 'color', 'style', 'clothing'
    score: float
    reason: str
    metadata: Dict


class IntelligentRecommender:
    """
    Sistema de recomendaciones inteligentes
    """
    
    def __init__(self):
        self.user_preferences: Dict[str, UserPreference] = {}
        self.item_features: Dict[str, Dict] = {}
        self.collaborative_scores: Dict[Tuple[str, str], float] = {}
        self.popularity_scores: Dict[str, float] = {}
    
    def update_user_preference(
        self,
        user_id: str,
        category: Optional[str] = None,
        color: Optional[str] = None,
        style: Optional[str] = None,
        item_id: Optional[str] = None,
        rating: Optional[float] = None
    ):
        """
        Actualizar preferencias del usuario
        
        Args:
            user_id: ID del usuario
            category: Categoría preferida
            color: Color preferido
            style: Estilo preferido
            item_id: ID del item usado
            rating: Calificación del item
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreference(
                user_id=user_id,
                preferred_categories=[],
                preferred_colors=[],
                preferred_styles=[],
                usage_history=[],
                ratings={}
            )
        
        pref = self.user_preferences[user_id]
        
        if category and category not in pref.preferred_categories:
            pref.preferred_categories.append(category)
        
        if color and color not in pref.preferred_colors:
            pref.preferred_colors.append(color)
        
        if style and style not in pref.preferred_styles:
            pref.preferred_styles.append(style)
        
        if item_id:
            pref.usage_history.append({
                'item_id': item_id,
                'timestamp': time.time(),
                'category': category,
                'color': color,
                'style': style
            })
        
        if item_id and rating:
            pref.ratings[item_id] = rating
    
    def register_item(
        self,
        item_id: str,
        item_type: str,
        features: Dict
    ):
        """Registrar item con sus características"""
        self.item_features[item_id] = {
            'type': item_type,
            'features': features,
            'usage_count': 0,
            'total_rating': 0.0,
            'rating_count': 0
        }
    
    def record_item_usage(self, item_id: str, rating: Optional[float] = None):
        """Registrar uso de item"""
        if item_id in self.item_features:
            self.item_features[item_id]['usage_count'] += 1
            
            if rating:
                item = self.item_features[item_id]
                item['total_rating'] += rating
                item['rating_count'] += 1
                avg_rating = item['total_rating'] / item['rating_count']
                self.popularity_scores[item_id] = self._calculate_popularity_score(
                    item['usage_count'],
                    avg_rating
                )
    
    def _calculate_popularity_score(self, usage_count: int, avg_rating: float) -> float:
        """Calcular score de popularidad"""
        # Combinación de uso y rating
        usage_score = min(math.log(usage_count + 1) / 10, 1.0)
        rating_score = avg_rating / 5.0  # Normalizar a 0-1
        
        return (usage_score * 0.4 + rating_score * 0.6)
    
    def recommend_clothing_templates(
        self,
        user_id: str,
        limit: int = 5,
        category: Optional[str] = None
    ) -> List[Recommendation]:
        """
        Recomendar plantillas de ropa
        
        Args:
            user_id: ID del usuario
            limit: Número de recomendaciones
            category: Categoría específica
        """
        if user_id not in self.user_preferences:
            # Recomendaciones basadas en popularidad
            return self._get_popular_recommendations(limit, category)
        
        pref = self.user_preferences[user_id]
        
        # Calcular scores para cada template
        recommendations = []
        
        for item_id, item_data in self.item_features.items():
            if item_data['type'] != 'template':
                continue
            
            if category and item_data['features'].get('category') != category:
                continue
            
            score = self._calculate_recommendation_score(item_id, item_data, pref)
            
            if score > 0:
                reason = self._generate_recommendation_reason(item_id, item_data, pref)
                
                recommendations.append(Recommendation(
                    item_id=item_id,
                    item_type='template',
                    score=score,
                    reason=reason,
                    metadata=item_data['features']
                ))
        
        # Ordenar por score
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        return recommendations[:limit]
    
    def _calculate_recommendation_score(
        self,
        item_id: str,
        item_data: Dict,
        pref: UserPreference
    ) -> float:
        """Calcular score de recomendación"""
        features = item_data['features']
        score = 0.0
        
        # Score basado en categoría
        if features.get('category') in pref.preferred_categories:
            score += 0.3
        
        # Score basado en colores
        item_colors = features.get('colors', [])
        matching_colors = len(set(item_colors) & set(pref.preferred_colors))
        if item_colors:
            score += (matching_colors / len(item_colors)) * 0.2
        
        # Score basado en estilos
        item_styles = features.get('styles', [])
        matching_styles = len(set(item_styles) & set(pref.preferred_styles))
        if item_styles:
            score += (matching_styles / len(item_styles)) * 0.2
        
        # Score de popularidad
        popularity = self.popularity_scores.get(item_id, 0.5)
        score += popularity * 0.2
        
        # Score basado en rating del usuario
        user_rating = pref.ratings.get(item_id, 0)
        if user_rating > 0:
            score += (user_rating / 5.0) * 0.1
        
        return min(score, 1.0)
    
    def _generate_recommendation_reason(
        self,
        item_id: str,
        item_data: Dict,
        pref: UserPreference
    ) -> str:
        """Generar razón de recomendación"""
        features = item_data['features']
        reasons = []
        
        if features.get('category') in pref.preferred_categories:
            reasons.append(f"Coincide con tu categoría preferida: {features.get('category')}")
        
        item_colors = features.get('colors', [])
        matching_colors = set(item_colors) & set(pref.preferred_colors)
        if matching_colors:
            reasons.append(f"Incluye tus colores favoritos: {', '.join(matching_colors)}")
        
        item_styles = features.get('styles', [])
        matching_styles = set(item_styles) & set(pref.preferred_styles)
        if matching_styles:
            reasons.append(f"Estilo que te gusta: {', '.join(matching_styles)}")
        
        popularity = self.popularity_scores.get(item_id, 0)
        if popularity > 0.7:
            reasons.append("Muy popular entre otros usuarios")
        
        if not reasons:
            return "Recomendación basada en tendencias"
        
        return ". ".join(reasons)
    
    def _get_popular_recommendations(
        self,
        limit: int,
        category: Optional[str]
    ) -> List[Recommendation]:
        """Obtener recomendaciones populares"""
        items = []
        
        for item_id, item_data in self.item_features.items():
            if item_data['type'] != 'template':
                continue
            
            if category and item_data['features'].get('category') != category:
                continue
            
            popularity = self.popularity_scores.get(item_id, 0)
            
            items.append(Recommendation(
                item_id=item_id,
                item_type='template',
                score=popularity,
                reason="Popular entre todos los usuarios",
                metadata=item_data['features']
            ))
        
        items.sort(key=lambda r: r.score, reverse=True)
        return items[:limit]
    
    def recommend_colors(
        self,
        user_id: str,
        base_color: Optional[str] = None
    ) -> List[str]:
        """Recomendar colores"""
        if user_id in self.user_preferences:
            pref = self.user_preferences[user_id]
            if base_color:
                # Colores complementarios
                return self._get_complementary_colors(base_color, pref.preferred_colors)
            return pref.preferred_colors[:5]
        
        return ['black', 'white', 'navy', 'gray', 'beige']
    
    def _get_complementary_colors(
        self,
        base_color: str,
        preferred_colors: List[str]
    ) -> List[str]:
        """Obtener colores complementarios"""
        color_map = {
            'black': ['white', 'gray', 'beige'],
            'white': ['black', 'navy', 'gray'],
            'navy': ['white', 'beige', 'gray'],
            'red': ['white', 'black', 'navy'],
            'blue': ['white', 'beige', 'gray']
        }
        
        complementary = color_map.get(base_color.lower(), [])
        # Combinar con preferencias
        result = list(set(complementary + preferred_colors))
        return result[:5]
    
    def get_user_insights(self, user_id: str) -> Dict:
        """Obtener insights del usuario"""
        if user_id not in self.user_preferences:
            return {}
        
        pref = self.user_preferences[user_id]
        
        # Análisis de uso
        category_counts = defaultdict(int)
        color_counts = defaultdict(int)
        style_counts = defaultdict(int)
        
        for usage in pref.usage_history:
            if usage.get('category'):
                category_counts[usage['category']] += 1
            if usage.get('color'):
                color_counts[usage['color']] += 1
            if usage.get('style'):
                style_counts[usage['style']] += 1
        
        return {
            'total_usage': len(pref.usage_history),
            'top_categories': sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'top_colors': sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'top_styles': sorted(style_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'average_rating': sum(pref.ratings.values()) / len(pref.ratings) if pref.ratings else 0
        }


# Instancia global
intelligent_recommender = IntelligentRecommender()

