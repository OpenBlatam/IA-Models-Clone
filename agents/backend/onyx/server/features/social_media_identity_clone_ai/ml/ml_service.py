"""
Servicio de Machine Learning para mejoras en el sistema
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..services.storage_service import StorageService
from ..db.models import GeneratedContentModel, IdentityProfileModel
from ..db.base import get_db_session
from sqlalchemy import func

logger = logging.getLogger(__name__)


class MLService:
    """Servicio de Machine Learning"""
    
    def __init__(self):
        self.storage = StorageService()
    
    def predict_content_performance(
        self,
        content: str,
        platform: str,
        identity_id: str
    ) -> Dict[str, Any]:
        """
        Predice el rendimiento de contenido antes de publicar
        
        Args:
            content: Contenido a predecir
            platform: Plataforma
            identity_id: ID de la identidad
            
        Returns:
            Predicción de rendimiento
        """
        # Obtener historial de contenido similar
        with get_db_session() as db:
            similar_content = db.query(GeneratedContentModel).filter_by(
                identity_profile_id=identity_id,
                platform=platform
            ).all()
        
        if not similar_content:
            return {
                "predicted_engagement": 0.5,
                "confidence": 0.3,
                "factors": ["No hay historial suficiente"]
            }
        
        # Análisis básico (en producción usaría ML real)
        factors = []
        score = 0.5
        
        # Factor: Longitud
        content_length = len(content)
        if 100 <= content_length <= 500:
            score += 0.1
            factors.append("Longitud óptima")
        elif content_length < 50:
            score -= 0.2
            factors.append("Contenido muy corto")
        elif content_length > 2000:
            score -= 0.1
            factors.append("Contenido muy largo")
        
        # Factor: Hashtags
        hashtag_count = content.count('#')
        if 5 <= hashtag_count <= 15:
            score += 0.15
            factors.append("Número óptimo de hashtags")
        elif hashtag_count == 0:
            score -= 0.1
            factors.append("Sin hashtags")
        elif hashtag_count > 30:
            score -= 0.05
            factors.append("Demasiados hashtags")
        
        # Factor: Emojis
        emoji_count = sum(1 for c in content if ord(c) > 127)
        if 1 <= emoji_count <= 5:
            score += 0.1
            factors.append("Uso moderado de emojis")
        elif emoji_count == 0:
            score -= 0.05
            factors.append("Sin emojis")
        
        # Factor: Engagement hooks
        hooks = ['?', '!', 'comenta', 'dime', 'qué opinas']
        has_hook = any(hook.lower() in content.lower() for hook in hooks)
        if has_hook:
            score += 0.15
            factors.append("Tiene hook de engagement")
        
        # Normalizar score
        score = max(0.0, min(1.0, score))
        
        # Calcular confianza basada en historial
        confidence = min(0.9, 0.3 + (len(similar_content) * 0.1))
        
        return {
            "predicted_engagement": score,
            "confidence": confidence,
            "factors": factors,
            "recommendation": self._get_recommendation(score)
        }
    
    def _get_recommendation(self, score: float) -> str:
        """Obtiene recomendación basada en score"""
        if score >= 0.8:
            return "Excelente contenido, listo para publicar"
        elif score >= 0.6:
            return "Buen contenido, considera pequeños ajustes"
        elif score >= 0.4:
            return "Contenido aceptable, pero puede mejorarse"
        else:
            return "Recomendamos revisar y mejorar el contenido antes de publicar"
    
    def analyze_content_trends(self, identity_id: str) -> Dict[str, Any]:
        """Analiza tendencias de contenido para una identidad"""
        with get_db_session() as db:
            # Obtener todo el contenido generado
            content_list = db.query(GeneratedContentModel).filter_by(
                identity_profile_id=identity_id
            ).all()
            
            if not content_list:
                return {"trends": [], "insights": []}
            
            # Analizar por plataforma
            platform_stats = {}
            for content in content_list:
                platform = content.platform
                if platform not in platform_stats:
                    platform_stats[platform] = {
                        "count": 0,
                        "avg_confidence": 0.0,
                        "total_confidence": 0.0
                    }
                
                platform_stats[platform]["count"] += 1
                if content.confidence_score:
                    platform_stats[platform]["total_confidence"] += content.confidence_score
            
            # Calcular promedios
            for platform, stats in platform_stats.items():
                if stats["count"] > 0:
                    stats["avg_confidence"] = stats["total_confidence"] / stats["count"]
            
            # Identificar mejor plataforma
            best_platform = max(
                platform_stats.items(),
                key=lambda x: x[1]["avg_confidence"]
            ) if platform_stats else None
            
            insights = []
            if best_platform:
                insights.append(
                    f"Mejor rendimiento en {best_platform[0]} "
                    f"(confianza promedio: {best_platform[1]['avg_confidence']:.2f})"
                )
            
            return {
                "platform_stats": platform_stats,
                "best_platform": best_platform[0] if best_platform else None,
                "insights": insights,
                "total_content": len(content_list)
            }


# Singleton global
_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Obtiene instancia singleton del servicio de ML"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service




