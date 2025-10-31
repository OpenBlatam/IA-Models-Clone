"""
Recommendation Engine Package
============================

Advanced recommendation system with collaborative filtering,
content-based filtering, and hybrid approaches for document classification.
"""

from .intelligent_recommender import (
    IntelligentRecommender,
    UserProfile,
    Item,
    Interaction,
    Recommendation,
    RecommendationSet,
    RecommendationType,
    RecommendationConfidence,
    FilteringMethod,
    CollaborativeFilter,
    ContentBasedFilter,
    HybridFilter
)

__all__ = [
    "IntelligentRecommender",
    "UserProfile",
    "Item",
    "Interaction",
    "Recommendation",
    "RecommendationSet",
    "RecommendationType",
    "RecommendationConfidence",
    "FilteringMethod",
    "CollaborativeFilter",
    "ContentBasedFilter",
    "HybridFilter"
]

























