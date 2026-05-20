"""
AI Services Module

High-level AI services organized into sub-modules:
- embeddings: Semantic embedding services
- analysis: Sentiment analysis and content moderation
- generation: Text and image generation
- recommendations: AI-powered recommendations
"""

from .embeddings import (
    EmbeddingService,
    EmbeddingServiceRefactored
)

from .analysis import (
    SentimentService,
    SentimentServiceRefactored,
    ModerationService,
    ModerationServiceRefactored
)

from .generation import (
    TextGenerationService,
    DiffusionService
)

from .recommendations import (
    RecommendationService
)

__all__ = [
    # Embeddings
    "EmbeddingService",
    "EmbeddingServiceRefactored",
    # Analysis
    "SentimentService",
    "SentimentServiceRefactored",
    "ModerationService",
    "ModerationServiceRefactored",
    # Generation
    "TextGenerationService",
    "DiffusionService",
    # Recommendations
    "RecommendationService",
]

