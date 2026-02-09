"""
Sistema de recomendaciones inteligentes
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from ..services.storage_service import StorageService
from ..analytics.analytics_service import AnalyticsService
from ..db.models import IdentityProfileModel, GeneratedContentModel
from ..db.base import get_db_session
from sqlalchemy import func

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Recomendación"""
    recommendation_type: str
    title: str
    description: str
    action: str
    priority: int  # 1-5, mayor es más importante
    data: Dict[str, Any]


class RecommendationService:
    """Servicio de recomendaciones inteligentes"""
    
    def __init__(self):
        self.storage = StorageService()
        self.analytics = AnalyticsService()
    
    def get_recommendations(self, identity_id: str) -> List[Recommendation]:
        """
        Obtiene recomendaciones para una identidad
        
        Args:
            identity_id: ID de la identidad
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        identity = self.storage.get_identity(identity_id)
        if not identity:
            return recommendations
        
        # Recomendación: Más contenido
        if identity.total_videos < 10 or identity.total_posts < 10:
            recommendations.append(Recommendation(
                recommendation_type="more_content",
                title="Agregar más contenido",
                description=f"La identidad tiene {identity.total_videos} videos y {identity.total_posts} posts. Se recomienda extraer más contenido para un análisis más preciso.",
                action="extract_more_content",
                priority=3,
                data={"current_videos": identity.total_videos, "current_posts": identity.total_posts}
            ))
        
        # Recomendación: Generar contenido
        with get_db_session() as db:
            content_count = db.query(func.count(GeneratedContentModel.id)).filter_by(
                identity_profile_id=identity_id
            ).scalar() or 0
        
        if content_count == 0:
            recommendations.append(Recommendation(
                recommendation_type="generate_content",
                title="Generar contenido",
                description="No se ha generado contenido aún. Comienza a generar contenido basado en esta identidad.",
                action="generate_content",
                priority=5,
                data={"identity_id": identity_id}
            ))
        
        # Recomendación: Actualizar identidad
        days_since_update = (datetime.utcnow() - identity.updated_at).days
        if days_since_update > 30:
            recommendations.append(Recommendation(
                recommendation_type="update_identity",
                title="Actualizar identidad",
                description=f"La identidad no se ha actualizado en {days_since_update} días. Se recomienda re-extraer perfiles para mantener la información actualizada.",
                action="update_identity",
                priority=2,
                data={"days_since_update": days_since_update}
            ))
        
        # Recomendación: Temas populares
        if identity.content_analysis.topics:
            recommendations.append(Recommendation(
                recommendation_type="popular_topics",
                title="Temas populares",
                description=f"Los temas más frecuentes son: {', '.join(identity.content_analysis.topics[:3])}. Considera generar contenido sobre estos temas.",
                action="generate_content_by_topic",
                priority=4,
                data={"topics": identity.content_analysis.topics[:3]}
            ))
        
        # Recomendación: Versión
        from ..services.versioning_service import VersioningService
        versioning = VersioningService()
        versions = versioning.list_versions(identity_id, limit=1)
        if not versions:
            recommendations.append(Recommendation(
                recommendation_type="create_version",
                title="Crear versión",
                description="No hay versiones guardadas. Se recomienda crear una versión para poder restaurar cambios en el futuro.",
                action="create_version",
                priority=2,
                data={"identity_id": identity_id}
            ))
        
        # Ordenar por prioridad
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations
    
    def get_system_recommendations(self) -> List[Recommendation]:
        """Obtiene recomendaciones a nivel de sistema"""
        recommendations = []
        
        stats = self.analytics.get_system_stats()
        
        # Recomendación: Pocas identidades
        if stats["total_identities"] < 5:
            recommendations.append(Recommendation(
                recommendation_type="more_identities",
                title="Crear más identidades",
                description=f"Solo hay {stats['total_identities']} identidades. Crea más para aprovechar mejor el sistema.",
                action="create_identity",
                priority=3,
                data={"current_count": stats["total_identities"]}
            ))
        
        # Recomendación: Poco contenido generado
        if stats["total_generated_content"] < 10:
            recommendations.append(Recommendation(
                recommendation_type="generate_more_content",
                title="Generar más contenido",
                description=f"Solo se ha generado {stats['total_generated_content']} contenido. Genera más para ver el potencial del sistema.",
                action="generate_content",
                priority=4,
                data={"current_count": stats["total_generated_content"]}
            ))
        
        return recommendations


# Singleton global
_recommendation_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    """Obtiene instancia singleton del servicio de recomendaciones"""
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service




