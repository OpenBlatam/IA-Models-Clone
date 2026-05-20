"""
Recommendations - Sistema de recomendaciones inteligentes
"""

import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Motor de recomendaciones"""

    def __init__(self):
        """Inicializar motor de recomendaciones"""
        self.user_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.content_similarity: Dict[str, List[str]] = {}
        self.usage_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record_usage(
        self,
        user_id: str,
        operation: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar uso para aprendizaje.

        Args:
            user_id: ID del usuario
            operation: Tipo de operación
            content_type: Tipo de contenido
            metadata: Metadatos adicionales
        """
        self.usage_patterns[user_id].append({
            "operation": operation,
            "content_type": content_type,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Actualizar preferencias
        if "preferred_operations" not in self.user_preferences[user_id]:
            self.user_preferences[user_id]["preferred_operations"] = defaultdict(int)
        
        self.user_preferences[user_id]["preferred_operations"][operation] += 1

    def recommend_position(
        self,
        content: str,
        addition: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recomendar posición para agregar contenido.

        Args:
            content: Contenido original
            addition: Contenido a agregar
            user_id: ID del usuario

        Returns:
            Recomendación de posición
        """
        # Análisis básico del contenido
        content_length = len(content)
        addition_length = len(addition)
        
        # Si el usuario tiene preferencias, usarlas
        if user_id and user_id in self.user_preferences:
            prefs = self.user_preferences[user_id]
            preferred_pos = prefs.get("preferred_position", "end")
        else:
            # Lógica de recomendación basada en contenido
            if content_length < 500:
                preferred_pos = "end"
            elif addition_length > content_length * 0.5:
                preferred_pos = "start"
            else:
                preferred_pos = "end"
        
        return {
            "recommended_position": preferred_pos,
            "confidence": 0.7,
            "reason": f"Basado en análisis de contenido y preferencias del usuario"
        }

    def recommend_removals(
        self,
        content: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recomendar elementos a eliminar.

        Args:
            content: Contenido
            user_id: ID del usuario

        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Buscar patrones comunes de contenido innecesario
        import re
        
        # Detectar espacios múltiples
        if re.search(r'\s{3,}', content):
            recommendations.append({
                "type": "whitespace",
                "pattern": r'\s{3,}',
                "reason": "Espacios múltiples detectados",
                "priority": 1
            })
        
        # Detectar líneas vacías múltiples
        if re.search(r'\n\s*\n\s*\n\s*\n', content):
            recommendations.append({
                "type": "empty_lines",
                "pattern": r'\n\s*\n\s*\n\s*\n',
                "reason": "Múltiples líneas vacías",
                "priority": 2
            })
        
        return recommendations

    def get_user_recommendations(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones para un usuario.

        Args:
            user_id: ID del usuario
            limit: Número de recomendaciones

        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        if user_id not in self.user_preferences:
            return recommendations
        
        prefs = self.user_preferences[user_id]
        
        # Recomendaciones basadas en uso
        usage = self.usage_patterns.get(user_id, [])
        if usage:
            recent_operations = [u["operation"] for u in usage[-10:]]
            most_common = max(set(recent_operations), key=recent_operations.count)
            
            recommendations.append({
                "type": "operation_suggestion",
                "message": f"Basado en tu uso, podrías usar más la operación: {most_common}",
                "priority": 1
            })
        
        return recommendations[:limit]

    def learn_from_feedback(
        self,
        user_id: str,
        recommendation_id: str,
        accepted: bool
    ):
        """
        Aprender de feedback del usuario.

        Args:
            user_id: ID del usuario
            recommendation_id: ID de la recomendación
            accepted: Si fue aceptada
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        if "recommendation_feedback" not in self.user_preferences[user_id]:
            self.user_preferences[user_id]["recommendation_feedback"] = {
                "accepted": 0,
                "rejected": 0
            }
        
        if accepted:
            self.user_preferences[user_id]["recommendation_feedback"]["accepted"] += 1
        else:
            self.user_preferences[user_id]["recommendation_feedback"]["rejected"] += 1
        
        logger.info(f"Feedback registrado: {user_id} - {recommendation_id} - {accepted}")






