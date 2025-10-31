"""
AI-powered features API endpoints
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from ....services.ai_service import AIService
from ....api.dependencies import CurrentUserDep
from ....core.exceptions import ExternalServiceError

router = APIRouter()


class ContentGenerationRequest(BaseModel):
    """Request model for content generation."""
    topic: str = Field(..., min_length=1, max_length=200, description="Blog post topic")
    style: str = Field(default="informative", description="Writing style")
    length: str = Field(default="medium", description="Content length (short, medium, long)")
    tone: str = Field(default="professional", description="Content tone")


class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis."""
    content: str = Field(..., min_length=1, description="Content to analyze")


class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search."""
    content: str = Field(..., min_length=1, description="Content to find similarities for")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")


class SEOOptimizationRequest(BaseModel):
    """Request model for SEO optimization."""
    title: str = Field(..., min_length=1, max_length=200, description="Blog post title")
    content: str = Field(..., min_length=1, description="Blog post content")
    target_keywords: List[str] = Field(default_factory=list, description="Target SEO keywords")


class TagSuggestionRequest(BaseModel):
    """Request model for tag suggestions."""
    content: str = Field(..., min_length=1, description="Content to suggest tags for")
    existing_tags: List[str] = Field(default_factory=list, description="Existing tags")


class PlagiarismDetectionRequest(BaseModel):
    """Request model for plagiarism detection."""
    content: str = Field(..., min_length=1, description="Content to check for plagiarism")


async def get_ai_service() -> AIService:
    """Get AI service instance."""
    return AIService()


@router.post("/generate-content", response_model=Dict[str, Any])
async def generate_blog_content(
    request: ContentGenerationRequest,
    background_tasks: BackgroundTasks,
    ai_service: AIService = Depends(get_ai_service),
    current_user: CurrentUserDep = Depends()
):
    """Generate blog content using AI."""
    try:
        generated_content = await ai_service.generate_blog_post(
            topic=request.topic,
            style=request.style,
            length=request.length,
            tone=request.tone
        )
        
        return {
            "success": True,
            "data": generated_content,
            "message": "Content generated successfully"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate content"
        )


@router.post("/analyze-sentiment", response_model=Dict[str, Any])
async def analyze_content_sentiment(
    request: ContentAnalysisRequest,
    ai_service: AIService = Depends(get_ai_service),
    current_user: CurrentUserDep = Depends()
):
    """Analyze content sentiment."""
    try:
        sentiment_analysis = await ai_service.analyze_content_sentiment(request.content)
        
        return {
            "success": True,
            "data": sentiment_analysis,
            "message": "Sentiment analysis completed"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze sentiment"
        )


@router.post("/classify-content", response_model=Dict[str, Any])
async def classify_content(
    request: ContentAnalysisRequest,
    ai_service: AIService = Depends(get_ai_service),
    current_user: CurrentUserDep = Depends()
):
    """Classify content into categories."""
    try:
        classification = await ai_service.classify_content(request.content)
        
        return {
            "success": True,
            "data": classification,
            "message": "Content classification completed"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to classify content"
        )


@router.post("/summarize-content", response_model=Dict[str, Any])
async def summarize_content(
    request: ContentAnalysisRequest,
    max_length: int = Field(default=150, ge=50, le=500, description="Maximum summary length"),
    ai_service: AIService = Depends(get_ai_service),
    current_user: CurrentUserDep = Depends()
):
    """Generate a summary of the content."""
    try:
        summary = await ai_service.summarize_content(request.content, max_length)
        
        return {
            "success": True,
            "data": {
                "summary": summary,
                "original_length": len(request.content),
                "summary_length": len(summary)
            },
            "message": "Content summarization completed"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to summarize content"
        )


@router.post("/suggest-tags", response_model=Dict[str, Any])
async def suggest_tags(
    request: TagSuggestionRequest,
    ai_service: AIService = Depends(get_ai_service),
    current_user: CurrentUserDep = Depends()
):
    """Suggest relevant tags for content."""
    try:
        suggested_tags = await ai_service.suggest_tags(
            request.content,
            request.existing_tags
        )
        
        return {
            "success": True,
            "data": {
                "suggested_tags": suggested_tags,
                "total_suggestions": len(suggested_tags)
            },
            "message": "Tag suggestions generated"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to suggest tags"
        )


@router.post("/optimize-seo", response_model=Dict[str, Any])
async def optimize_content_for_seo(
    request: SEOOptimizationRequest,
    ai_service: AIService = Depends(get_ai_service),
    current_user: CurrentUserDep = Depends()
):
    """Optimize content for SEO."""
    try:
        seo_optimization = await ai_service.optimize_content_for_seo(
            request.title,
            request.content,
            request.target_keywords
        )
        
        return {
            "success": True,
            "data": seo_optimization,
            "message": "SEO optimization completed"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize content for SEO"
        )


@router.post("/detect-plagiarism", response_model=Dict[str, Any])
async def detect_plagiarism(
    request: PlagiarismDetectionRequest,
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Plagiarism threshold"),
    ai_service: AIService = Depends(get_ai_service),
    current_user: CurrentUserDep = Depends()
):
    """Detect potential plagiarism in content."""
    try:
        # In a real implementation, you would pass existing posts from the database
        # For now, we'll use an empty list
        existing_posts = []
        
        plagiarism_result = await ai_service.detect_plagiarism(
            request.content,
            existing_posts,
            threshold
        )
        
        return {
            "success": True,
            "data": plagiarism_result,
            "message": "Plagiarism detection completed"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect plagiarism"
        )


@router.post("/generate-embeddings", response_model=Dict[str, Any])
async def generate_embeddings(
    request: ContentAnalysisRequest,
    ai_service: AIService = Depends(get_ai_service),
    current_user: CurrentUserDep = Depends()
):
    """Generate embeddings for content."""
    try:
        embeddings = await ai_service.generate_embeddings(request.content)
        
        return {
            "success": True,
            "data": {
                "embeddings": embeddings,
                "dimension": len(embeddings)
            },
            "message": "Embeddings generated successfully"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings"
        )


@router.get("/ai-status", response_model=Dict[str, Any])
async def get_ai_service_status(
    ai_service: AIService = Depends(get_ai_service)
):
    """Get AI service status and available features."""
    try:
        status_info = {
            "openai_available": ai_service.openai_client is not None,
            "sentiment_analyzer_available": ai_service.sentiment_analyzer is not None,
            "text_classifier_available": ai_service.text_classifier is not None,
            "summarizer_available": ai_service.summarizer is not None,
            "embeddings_available": ai_service.embeddings_model is not None,
            "available_features": []
        }
        
        # Determine available features
        if status_info["openai_available"]:
            status_info["available_features"].extend([
                "content_generation",
                "seo_optimization",
                "tag_suggestions"
            ])
        
        if status_info["sentiment_analyzer_available"]:
            status_info["available_features"].append("sentiment_analysis")
        
        if status_info["text_classifier_available"]:
            status_info["available_features"].append("content_classification")
        
        if status_info["summarizer_available"]:
            status_info["available_features"].append("content_summarization")
        
        if status_info["embeddings_available"]:
            status_info["available_features"].extend([
                "similarity_search",
                "plagiarism_detection"
            ])
        
        return {
            "success": True,
            "data": status_info,
            "message": "AI service status retrieved"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AI service status"
        )






























