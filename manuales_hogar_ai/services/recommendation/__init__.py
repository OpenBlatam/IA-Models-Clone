"""
Recommendation Service Module
============================

Módulo especializado para recomendaciones de manuales.
"""

from .recommendation_service import RecommendationService
from .similarity_engine import SimilarityEngine
from .popularity_engine import PopularityEngine

__all__ = [
    "RecommendationService",
    "SimilarityEngine",
    "PopularityEngine",
]

