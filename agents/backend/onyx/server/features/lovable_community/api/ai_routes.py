"""
AI-powered endpoints for Lovable Community

Endpoints for:
- Semantic search
- Sentiment analysis
- Content moderation
- Text generation
- Recommendations
"""

import logging
from typing import List, Optional, Annotated
from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_service_factory
from ..factories import ServiceFactory
from ..repositories import ChatRepository
from ..services.ai import (
    EmbeddingService,
    SentimentService,
    ModerationService,
    TextGenerationService,
    RecommendationService
)
from ..exceptions import ChatNotFoundError
from .dependencies_ai import (
    get_embedding_service,
    get_sentiment_service,
    get_moderation_service,
    get_text_generation_service,
    get_recommendation_service
)
from .route_helpers import handle_route_errors, get_chat_or_raise_404

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI"])


def get_chat_repository(
    factory: Annotated[ServiceFactory, Depends(get_service_factory)]
) -> ChatRepository:
    """
    Dependency for chat repository.
    
    Args:
        factory: ServiceFactory (injected)
        
    Returns:
        ChatRepository instance
    """
    return factory.repository_factory.get_chat_repository()


@router.post("/embeddings/{chat_id}")
@handle_route_errors("creating embedding")
async def create_embedding(
    chat_id: str,
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    chat_repository: Annotated[ChatRepository, Depends(get_chat_repository)]
):
    """
    Create or update embedding for a chat
    
    This enables semantic search for the chat.
    """
    chat = get_chat_or_raise_404(chat_repository, chat_id)
    embedding = embedding_service.create_or_update_embedding(chat_id, chat=chat)
    
    return {
        "chat_id": chat_id,
        "embedding_id": embedding.id,
        "model": embedding.embedding_model,
        "dimension": len(embedding.embedding),
        "created_at": embedding.created_at
    }


@router.get("/search/semantic")
@handle_route_errors("performing semantic search")
async def semantic_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    min_similarity: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)]
):
    """
    Semantic search using embeddings
    
    Find chats similar to the query text using semantic similarity.
    """
    results = embedding_service.find_similar_chats(
        query_text=query,
        limit=limit,
        min_similarity=min_similarity
    )
    
    return {
        "query": query,
        "results": [
            {
                "chat_id": item["chat_id"],
                "similarity": item["similarity"],
                "title": item["chat"].title,
                "description": item["chat"].description,
                "score": item["chat"].score
            }
            for item in results
        ],
        "count": len(results)
    }


@router.post("/sentiment/{chat_id}")
@handle_route_errors("analyzing sentiment")
async def analyze_sentiment(
    chat_id: str,
    sentiment_service: Annotated[SentimentService, Depends(get_sentiment_service)],
    chat_repository: Annotated[ChatRepository, Depends(get_chat_repository)]
):
    """
    Analyze sentiment of a chat
    
    Returns sentiment label (positive/negative/neutral) and confidence score.
    """
    chat = get_chat_or_raise_404(chat_repository, chat_id)
    metadata = sentiment_service.analyze_chat_sentiment(chat_id, chat=chat)
    
    return {
        "chat_id": chat_id,
        "sentiment": {
            "label": metadata.sentiment_label,
            "score": metadata.sentiment_score
        }
    }


@router.post("/sentiment/analyze")
@handle_route_errors("analyzing text sentiment")
async def analyze_text_sentiment(
    text: str = Query(..., description="Text to analyze"),
    sentiment_service: Annotated[SentimentService, Depends(get_sentiment_service)]
):
    """
    Analyze sentiment of arbitrary text
    
    Useful for analyzing text before publishing.
    """
    result = sentiment_service.analyze_sentiment(text)
    
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "sentiment": {
            "label": result["label"],
            "score": result["score"]
        }
    }


@router.post("/moderate/{chat_id}")
@handle_route_errors("moderating chat")
async def moderate_chat(
    chat_id: str,
    moderation_service: Annotated[ModerationService, Depends(get_moderation_service)],
    chat_repository: Annotated[ChatRepository, Depends(get_chat_repository)]
):
    """
    Moderate a chat for toxic or inappropriate content
    
    Returns moderation results including toxicity score and flags.
    """
    chat = get_chat_or_raise_404(chat_repository, chat_id)
    metadata = moderation_service.moderate_chat(chat_id, chat=chat)
    
    return {
        "chat_id": chat_id,
        "moderation": {
            "is_toxic": metadata.is_toxic,
            "toxicity_score": metadata.toxicity_score,
            "flags": metadata.moderation_flags
        }
    }


@router.post("/moderate/check")
@handle_route_errors("checking content")
async def check_content(
    text: str = Query(..., description="Text to check"),
    moderation_service: Annotated[ModerationService, Depends(get_moderation_service)]
):
    """
    Check text for toxicity before publishing
    
    Returns whether content should be blocked.
    """
    result = moderation_service.moderate_content(text)
    should_block = moderation_service.should_block_content(text)
    
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "is_toxic": result["is_toxic"],
        "toxicity_score": result["toxicity_score"],
        "should_block": should_block,
        "flags": result.get("flags", [])
    }


@router.post("/generate")
@handle_route_errors("generating text")
async def generate_text(
    prompt: str = Query(..., description="Text prompt"),
    max_length: Optional[int] = Query(None, ge=10, le=500, description="Maximum length"),
    temperature: float = Query(0.7, ge=0.0, le=2.0, description="Sampling temperature"),
    text_generation_service: Annotated[TextGenerationService, Depends(get_text_generation_service)]
):
    """
    Generate text using LLM
    
    Useful for text completion, enhancement, or generation.
    """
    result = text_generation_service.generate_text(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature
    )
    
    return {
        "prompt": prompt,
        "generated_text": result.get("generated_text", ""),
        "model": result.get("model", "unknown")
    }


@router.post("/enhance/description")
@handle_route_errors("enhancing description")
async def enhance_description(
    description: str = Query(..., description="Description to enhance"),
    text_generation_service: Annotated[TextGenerationService, Depends(get_text_generation_service)]
):
    """
    Enhance or embellish a chat description
    
    Improves the description while keeping the same meaning.
    """
    enhanced = text_generation_service.enhance_description(description)
    
    return {
        "original": description,
        "enhanced": enhanced
    }


@router.post("/generate/tags")
@handle_route_errors("generating tags")
async def generate_tags(
    title: str = Query(..., description="Chat title"),
    description: Optional[str] = Query(None, description="Optional description"),
    text_generation_service: Annotated[TextGenerationService, Depends(get_text_generation_service)]
):
    """
    Generate relevant tags for a chat
    
    Uses AI to suggest tags based on title and description.
    """
    tags = text_generation_service.generate_tags(title, description)
    
    return {
        "title": title,
        "description": description,
        "suggested_tags": tags
    }


@router.get("/recommend/{chat_id}")
@handle_route_errors("getting recommendations")
async def recommend_similar(
    chat_id: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum recommendations"),
    recommendation_service: Annotated[RecommendationService, Depends(get_recommendation_service)]
):
    """
    Get recommendations for chats similar to the given chat
    
    Uses semantic similarity based on embeddings.
    """
    recommendations = recommendation_service.recommend_similar_chats(
        chat_id=chat_id,
        limit=limit
    )
    
    return {
        "chat_id": chat_id,
        "recommendations": [
            {
                "chat_id": rec["chat_id"],
                "similarity": rec["similarity"],
                "title": rec["chat"].title,
                "description": rec["chat"].description,
                "score": rec["chat"].score
            }
            for rec in recommendations
        ],
        "count": len(recommendations)
    }


@router.get("/recommend/user/{user_id}")
@handle_route_errors("getting user recommendations")
async def recommend_for_user(
    user_id: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum recommendations"),
    use_sentiment: bool = Query(True, description="Use sentiment preferences"),
    recommendation_service: Annotated[RecommendationService, Depends(get_recommendation_service)]
):
    """
    Get personalized recommendations for a user
    
    Based on user's voting history and preferences.
    """
    recommendations = recommendation_service.recommend_for_user(
        user_id=user_id,
        limit=limit,
        use_sentiment=use_sentiment
    )
    
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "chat_id": rec["chat_id"],
                "similarity": rec.get("similarity", 0.0),
                "title": rec["chat"].title,
                "description": rec["chat"].description,
                "score": rec["chat"].score,
                "sentiment_match": rec.get("sentiment_match", False)
            }
            for rec in recommendations
        ],
        "count": len(recommendations)
    }


@router.get("/recommend/tags")
@handle_route_errors("getting tag recommendations")
async def recommend_by_tags(
    tags: List[str] = Query(..., description="Tags to search for"),
    limit: int = Query(10, ge=1, le=50, description="Maximum recommendations"),
    recommendation_service: Annotated[RecommendationService, Depends(get_recommendation_service)]
):
    """
    Get recommendations based on tags
    
    Returns chats matching the provided tags, ordered by score.
    """
    chats = recommendation_service.recommend_by_tags(tags, limit=limit)
    
    return {
        "tags": tags,
        "recommendations": [
            {
                "chat_id": chat.id,
                "title": chat.title,
                "description": chat.description,
                "score": chat.score,
                "tags": chat.tags
            }
            for chat in chats
        ],
        "count": len(chats)
    }









