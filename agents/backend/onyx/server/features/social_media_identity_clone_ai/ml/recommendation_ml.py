"""
Machine Learning para recomendaciones mejoradas
"""

import logging
from typing import List, Dict, Any
from collections import Counter

from ..services.storage_service import StorageService
from ..db.models import GeneratedContentModel, IdentityProfileModel
from ..db.base import get_db_session

logger = logging.getLogger(__name__)


class RecommendationML:
    """ML para mejorar recomendaciones"""
    
    def __init__(self):
        self.storage = StorageService()
    
    def get_smart_recommendations(self, identity_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene recomendaciones inteligentes usando ML
        
        Args:
            identity_id: ID de la identidad
            
        Returns:
            Lista de recomendaciones inteligentes
        """
        recommendations = []
        
        identity = self.storage.get_identity(identity_id)
        if not identity:
            return recommendations
        
        # Analizar contenido generado
        with get_db_session() as db:
            content_list = db.query(GeneratedContentModel).filter_by(
                identity_profile_id=identity_id
            ).all()
            
            # Recomendación basada en frecuencia de generación
            if len(content_list) < 5:
                recommendations.append({
                    "type": "generate_more",
                    "priority": 5,
                    "title": "Genera más contenido",
                    "description": f"Solo has generado {len(content_list)} contenidos. Genera más para tener mejor análisis.",
                    "action": "generate_content",
                    "data": {"current_count": len(content_list)}
                })
            
            # Recomendación basada en plataformas usadas
            platforms_used = [c.platform for c in content_list]
            platform_counts = Counter(platforms_used)
            
            if len(platform_counts) < 2:
                recommendations.append({
                    "type": "diversify_platforms",
                    "priority": 4,
                    "title": "Diversifica plataformas",
                    "description": f"Has usado principalmente {list(platform_counts.keys())[0]}. Prueba otras plataformas.",
                    "action": "generate_for_other_platform",
                    "data": {"current_platforms": list(platform_counts.keys())}
                })
            
            # Recomendación basada en temas
            if identity.content_analysis.topics:
                # Encontrar temas menos usados
                all_topics = identity.content_analysis.topics
                used_topics = set()
                for content in content_list:
                    # Extraer temas del contenido (simplificado)
                    content_lower = content.content.lower()
                    for topic in all_topics:
                        if topic.lower() in content_lower:
                            used_topics.add(topic)
                
                unused_topics = set(all_topics) - used_topics
                if unused_topics:
                    recommendations.append({
                        "type": "explore_topics",
                        "priority": 3,
                        "title": "Explora nuevos temas",
                        "description": f"Tienes {len(unused_topics)} temas que aún no has explorado: {', '.join(list(unused_topics)[:3])}",
                        "action": "generate_content_by_topic",
                        "data": {"unused_topics": list(unused_topics)[:5]}
                    })
        
        return recommendations




