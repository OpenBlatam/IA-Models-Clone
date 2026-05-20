"""
Servicio de Recomendaciones (Legacy)
====================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar services.recommendation.recommendation_service.RecommendationService
"""

from .recommendation.recommendation_service import RecommendationService

__all__ = ["RecommendationService"]

