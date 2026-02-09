"""
Recommendation Engine - Sistema de recomendaciones inteligentes
=================================================================
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Motor de recomendaciones basado en uso y similitud.
    """
    
    def __init__(self):
        """Inicializar motor de recomendaciones"""
        self.usage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.paper_similarity: Dict[str, List[str]] = {}
    
    def recommend_papers(
        self,
        user_id: Optional[str] = None,
        current_paper_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recomienda papers basándose en uso y similitud.
        
        Args:
            user_id: ID del usuario (opcional)
            current_paper_id: ID del paper actual (opcional)
            limit: Número de recomendaciones
            
        Returns:
            Lista de papers recomendados
        """
        recommendations = []
        
        # Recomendaciones basadas en uso del usuario
        if user_id:
            user_recommendations = self._recommend_by_usage(user_id, limit)
            recommendations.extend(user_recommendations)
        
        # Recomendaciones basadas en similitud
        if current_paper_id:
            similar_recommendations = self._recommend_by_similarity(current_paper_id, limit)
            recommendations.extend(similar_recommendations)
        
        # Recomendaciones populares
        if not recommendations:
            popular_recommendations = self._recommend_popular(limit)
            recommendations.extend(popular_recommendations)
        
        # Deduplicar y limitar
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            paper_id = rec.get("paper_id")
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= limit:
                    break
        
        return unique_recommendations
    
    def _recommend_by_usage(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Recomienda basándose en uso del usuario"""
        user_stats = self.usage_stats.get(user_id, {})
        
        if not user_stats:
            return []
        
        # Papers más usados por el usuario
        sorted_papers = sorted(
            user_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                "paper_id": paper_id,
                "reason": "Frequently used by you",
                "score": count
            }
            for paper_id, count in sorted_papers
        ]
    
    def _recommend_by_similarity(self, paper_id: str, limit: int) -> List[Dict[str, Any]]:
        """Recomienda papers similares"""
        similar = self.paper_similarity.get(paper_id, [])
        
        return [
            {
                "paper_id": similar_id,
                "reason": "Similar to current paper",
                "score": 1.0
            }
            for similar_id in similar[:limit]
        ]
    
    def _recommend_popular(self, limit: int) -> List[Dict[str, Any]]:
        """Recomienda papers populares"""
        # Agregar todos los usos
        total_usage = defaultdict(int)
        for user_stats in self.usage_stats.values():
            for paper_id, count in user_stats.items():
                total_usage[paper_id] += count
        
        # Ordenar por popularidad
        popular = sorted(
            total_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                "paper_id": paper_id,
                "reason": "Popular among users",
                "score": count
            }
            for paper_id, count in popular
        ]
    
    def record_usage(self, user_id: str, paper_id: str):
        """
        Registra uso de un paper por un usuario.
        
        Args:
            user_id: ID del usuario
            paper_id: ID del paper
        """
        self.usage_stats[user_id][paper_id] += 1
        logger.debug(f"Uso registrado: {user_id} -> {paper_id}")
    
    def update_similarity(self, paper_id: str, similar_papers: List[str]):
        """
        Actualiza similitud entre papers.
        
        Args:
            paper_id: ID del paper
            similar_papers: Lista de papers similares
        """
        self.paper_similarity[paper_id] = similar_papers




