"""
Personalization Engine - Sistema de personalización de contenido
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Perfil de usuario"""
    user_id: str
    preferences: Dict[str, Any]
    reading_history: List[str]
    interaction_history: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class PersonalizationEngine:
    """Motor de personalización"""

    def __init__(self):
        """Inicializar motor"""
        self.user_profiles: Dict[str, UserProfile] = {}
        self.content_tags: Dict[str, List[str]] = {}

    def create_user_profile(
        self,
        user_id: str,
        initial_preferences: Optional[Dict[str, Any]] = None
    ):
        """
        Crear perfil de usuario.

        Args:
            user_id: ID del usuario
            initial_preferences: Preferencias iniciales
        """
        profile = UserProfile(
            user_id=user_id,
            preferences=initial_preferences or {},
            reading_history=[],
            interaction_history=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.user_profiles[user_id] = profile
        logger.debug(f"Perfil de usuario creado: {user_id}")

    def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ):
        """
        Actualizar preferencias de usuario.

        Args:
            user_id: ID del usuario
            preferences: Nuevas preferencias
        """
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        profile.preferences.update(preferences)
        profile.updated_at = datetime.utcnow()
        
        logger.debug(f"Preferencias actualizadas para usuario: {user_id}")

    def record_interaction(
        self,
        user_id: str,
        content_id: str,
        interaction_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar interacción de usuario.

        Args:
            user_id: ID del usuario
            content_id: ID del contenido
            interaction_type: Tipo de interacción (read, like, share, etc.)
            metadata: Metadatos adicionales
        """
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        # Agregar a historial de lectura si es lectura
        if interaction_type == "read" and content_id not in profile.reading_history:
            profile.reading_history.append(content_id)
            # Limitar tamaño
            if len(profile.reading_history) > 100:
                profile.reading_history = profile.reading_history[-100:]
        
        # Agregar a historial de interacciones
        interaction = {
            "content_id": content_id,
            "type": interaction_type,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        profile.interaction_history.append(interaction)
        
        # Limitar tamaño
        if len(profile.interaction_history) > 200:
            profile.interaction_history = profile.interaction_history[-200:]
        
        profile.updated_at = datetime.utcnow()
        logger.debug(f"Interacción registrada: {user_id} - {interaction_type} - {content_id}")

    def personalize_content(
        self,
        user_id: str,
        content: str,
        content_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Personalizar contenido para usuario.

        Args:
            user_id: ID del usuario
            content: Contenido original
            content_id: ID del contenido (opcional)

        Returns:
            Contenido personalizado y recomendaciones
        """
        if user_id not in self.user_profiles:
            return {
                "personalized": False,
                "content": content,
                "message": "No hay perfil de usuario disponible"
            }
        
        profile = self.user_profiles[user_id]
        
        # Obtener preferencias
        preferences = profile.preferences
        
        # Personalizar según preferencias
        personalized_content = content
        modifications = []
        
        # Ajustar tono según preferencias
        preferred_tone = preferences.get("tone", "neutral")
        if preferred_tone != "neutral":
            modifications.append(f"Tono ajustado a: {preferred_tone}")
        
        # Ajustar longitud según preferencias
        preferred_length = preferences.get("content_length", "medium")
        if preferred_length == "short" and len(content.split()) > 300:
            # Sugerir versión corta
            modifications.append("Versión corta recomendada")
        elif preferred_length == "long" and len(content.split()) < 500:
            modifications.append("Versión extendida recomendada")
        
        # Generar recomendaciones basadas en historial
        recommendations = self._generate_recommendations(profile, content_id)
        
        return {
            "personalized": True,
            "content": personalized_content,
            "modifications": modifications,
            "recommendations": recommendations,
            "user_preferences": preferences
        }

    def get_content_recommendations(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones de contenido para usuario.

        Args:
            user_id: ID del usuario
            limit: Límite de recomendaciones

        Returns:
            Lista de recomendaciones
        """
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        
        # Analizar historial de lectura
        read_content = set(profile.reading_history)
        
        # Analizar interacciones
        liked_content = {
            item["content_id"]
            for item in profile.interaction_history
            if item.get("type") == "like"
        }
        
        # Generar recomendaciones basadas en tags
        recommendations = []
        
        # Contenido similar basado en tags
        for content_id, tags in self.content_tags.items():
            if content_id not in read_content:
                # Calcular score de relevancia
                score = self._calculate_relevance_score(profile, tags, liked_content)
                if score > 0:
                    recommendations.append({
                        "content_id": content_id,
                        "score": score,
                        "tags": tags,
                        "reason": "Basado en tu historial de lectura"
                    })
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations[:limit]

    def tag_content(
        self,
        content_id: str,
        tags: List[str]
    ):
        """
        Etiquetar contenido.

        Args:
            content_id: ID del contenido
            tags: Lista de tags
        """
        self.content_tags[content_id] = tags
        logger.debug(f"Contenido etiquetado: {content_id} - {tags}")

    def _generate_recommendations(
        self,
        profile: UserProfile,
        current_content_id: Optional[str]
    ) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        # Basado en historial
        if profile.reading_history:
            recent_content = profile.reading_history[-5:]
            recommendations.append(f"Basado en tu historial reciente ({len(recent_content)} contenidos)")
        
        # Basado en preferencias
        if profile.preferences:
            recommendations.append("Personalizado según tus preferencias")
        
        return recommendations

    def _calculate_relevance_score(
        self,
        profile: UserProfile,
        content_tags: List[str],
        liked_content: set
    ) -> float:
        """Calcular score de relevancia"""
        score = 0.0
        
        # Analizar tags de contenido leído
        read_tags = []
        for content_id in profile.reading_history:
            if content_id in self.content_tags:
                read_tags.extend(self.content_tags[content_id])
        
        # Calcular overlap de tags
        if read_tags and content_tags:
            common_tags = set(read_tags).intersection(set(content_tags))
            if common_tags:
                score += len(common_tags) / max(len(read_tags), len(content_tags))
        
        # Bonus por contenido similar a contenido que le gustó
        if liked_content:
            liked_tags = []
            for content_id in liked_content:
                if content_id in self.content_tags:
                    liked_tags.extend(self.content_tags[content_id])
            
            if liked_tags and content_tags:
                common_liked_tags = set(liked_tags).intersection(set(content_tags))
                if common_liked_tags:
                    score += len(common_liked_tags) / max(len(liked_tags), len(content_tags)) * 0.5
        
        return score






