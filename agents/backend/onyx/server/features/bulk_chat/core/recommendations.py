"""
Recommendations - Sistema de Recomendaciones ML
==============================================

Sistema de recomendaciones basado en machine learning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Recomendación generada."""
    item_id: str
    item_type: str  # "topic", "response", "action", etc.
    score: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecommendationEngine:
    """Motor de recomendaciones basado en ML."""
    
    def __init__(self):
        self.user_interactions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.item_similarity: Dict[str, Dict[str, float]] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
    
    async def record_interaction(
        self,
        user_id: str,
        item_id: str,
        item_type: str,
        rating: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Registrar interacción de usuario.
        
        Args:
            user_id: ID del usuario
            item_id: ID del item
            item_type: Tipo de item
            rating: Rating (0.0 - 1.0)
            metadata: Metadatos adicionales
        """
        interaction = {
            "item_id": item_id,
            "item_type": item_type,
            "rating": rating,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
        }
        
        self.user_interactions[user_id].append(interaction)
        
        # Actualizar perfil de usuario
        await self._update_user_profile(user_id)
    
    async def _update_user_profile(self, user_id: str):
        """Actualizar perfil de usuario."""
        interactions = self.user_interactions[user_id]
        
        if not interactions:
            return
        
        # Calcular preferencias
        item_preferences = defaultdict(float)
        type_preferences = defaultdict(float)
        
        for interaction in interactions:
            item_preferences[interaction["item_id"]] += interaction["rating"]
            type_preferences[interaction["item_type"]] += interaction["rating"]
        
        # Normalizar
        total_rating = sum(item_preferences.values())
        if total_rating > 0:
            item_preferences = {
                k: v / total_rating
                for k, v in item_preferences.items()
            }
        
        total_type = sum(type_preferences.values())
        if total_type > 0:
            type_preferences = {
                k: v / total_type
                for k, v in type_preferences.items()
            }
        
        self.user_profiles[user_id] = {
            "item_preferences": dict(item_preferences),
            "type_preferences": dict(type_preferences),
            "total_interactions": len(interactions),
            "last_updated": datetime.now(),
        }
    
    async def calculate_similarity(self, item1: str, item2: str) -> float:
        """Calcular similitud entre items (collaborative filtering)."""
        # Usuarios que interactuaron con ambos items
        users_item1 = set()
        users_item2 = set()
        
        for user_id, interactions in self.user_interactions.items():
            item_ids = {i["item_id"] for i in interactions}
            if item1 in item_ids:
                users_item1.add(user_id)
            if item2 in item_ids:
                users_item2.add(user_id)
        
        # Intersección de usuarios
        common_users = users_item1 & users_item2
        
        if not common_users:
            return 0.0
        
        # Calcular similitud usando Jaccard
        union = users_item1 | users_item2
        similarity = len(common_users) / len(union) if union else 0.0
        
        return similarity
    
    async def recommend_items(
        self,
        user_id: str,
        item_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Recommendation]:
        """
        Generar recomendaciones para usuario.
        
        Args:
            user_id: ID del usuario
            item_type: Tipo de item (opcional)
            limit: Número máximo de recomendaciones
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Si el usuario tiene perfil, usar collaborative filtering
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            user_items = set(profile["item_preferences"].keys())
            
            # Encontrar items similares
            all_items = set()
            for interactions in self.user_interactions.values():
                all_items.update(i["item_id"] for i in interactions)
            
            candidate_items = all_items - user_items
            
            for candidate_item in candidate_items:
                # Calcular score basado en similitud
                max_similarity = 0.0
                similar_item = None
                
                for user_item in user_items:
                    similarity = await self.calculate_similarity(
                        user_item,
                        candidate_item,
                    )
                    if similarity > max_similarity:
                        max_similarity = similarity
                        similar_item = user_item
                
                if max_similarity > 0.1:  # Threshold mínimo
                    recommendations.append(Recommendation(
                        item_id=candidate_item,
                        item_type=item_type or "unknown",
                        score=max_similarity,
                        reason=f"Similar to {similar_item}",
                    ))
        
        # Si no hay suficiente, usar popular items
        if len(recommendations) < limit:
            popular = await self._get_popular_items(item_type, limit - len(recommendations))
            recommendations.extend(popular)
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return recommendations[:limit]
    
    async def _get_popular_items(
        self,
        item_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Recommendation]:
        """Obtener items populares."""
        item_counts = defaultdict(int)
        
        for interactions in self.user_interactions.values():
            for interaction in interactions:
                if item_type is None or interaction["item_type"] == item_type:
                    item_counts[interaction["item_id"]] += 1
        
        # Ordenar por popularidad
        sorted_items = sorted(
            item_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        total = sum(item_counts.values())
        
        recommendations = []
        for item_id, count in sorted_items[:limit]:
            score = count / total if total > 0 else 0.0
            recommendations.append(Recommendation(
                item_id=item_id,
                item_type=item_type or "unknown",
                score=score,
                reason="Popular item",
            ))
        
        return recommendations
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtener perfil de usuario."""
        return self.user_profiles.get(user_id)
































