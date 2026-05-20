"""
AI Service Dependencies for FastAPI

Dependency functions for AI services following the same pattern
as other dependencies in the codebase.
"""

from typing import Annotated
from fastapi import Depends
from sqlalchemy.orm import Session

from ..dependencies import get_db
from ..services.ai import (
    EmbeddingService,
    SentimentService,
    ModerationService,
    TextGenerationService,
    RecommendationService
)


def get_embedding_service(
    db: Annotated[Session, Depends(get_db)]
) -> EmbeddingService:
    """
    Dependency for embedding service.
    
    Args:
        db: Database session (injected)
        
    Returns:
        EmbeddingService instance
    """
    return EmbeddingService(db)


def get_sentiment_service(
    db: Annotated[Session, Depends(get_db)]
) -> SentimentService:
    """
    Dependency for sentiment service.
    
    Args:
        db: Database session (injected)
        
    Returns:
        SentimentService instance
    """
    return SentimentService(db)


def get_moderation_service(
    db: Annotated[Session, Depends(get_db)]
) -> ModerationService:
    """
    Dependency for moderation service.
    
    Args:
        db: Database session (injected)
        
    Returns:
        ModerationService instance
    """
    return ModerationService(db)


def get_text_generation_service() -> TextGenerationService:
    """
    Dependency for text generation service.
    
    Returns:
        TextGenerationService instance
    """
    return TextGenerationService()


def get_recommendation_service(
    db: Annotated[Session, Depends(get_db)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)]
) -> RecommendationService:
    """
    Dependency for recommendation service.
    
    Args:
        db: Database session (injected)
        embedding_service: Embedding service (injected)
        
    Returns:
        RecommendationService instance
    """
    return RecommendationService(db, embedding_service)



