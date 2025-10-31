"""
AI Content Analysis Routes - Advanced AI-powered content analysis API
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import json

from ..core.ai_content_analyzer import (
    analyze_content_with_ai,
    generate_ai_insights,
    get_ai_analyzer_health,
    initialize_ai_analyzer,
    AIContentAnalysis,
    ContentInsights
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["AI Analysis"])


# Pydantic models for request/response validation
class ContentAnalysisRequest(BaseModel):
    """Request model for AI content analysis"""
    content: str = Field(..., min_length=1, max_length=10000, description="Content text to analyze")
    content_id: str = Field(default="", description="Optional content identifier")
    include_insights: bool = Field(default=False, description="Whether to include detailed insights")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()


class BatchAnalysisRequest(BaseModel):
    """Request model for batch AI analysis"""
    content_items: List[Dict[str, str]] = Field(..., min_items=1, max_items=100, description="List of content items")
    include_insights: bool = Field(default=False, description="Whether to include detailed insights")
    batch_size: int = Field(default=5, ge=1, le=20, description="Number of items to process per batch")
    
    @validator('content_items')
    def validate_content_items(cls, v):
        for item in v:
            if 'content' not in item:
                raise ValueError('Each content item must have a "content" field')
            if not item['content'].strip():
                raise ValueError('Content cannot be empty')
        return v


class ContentComparisonRequest(BaseModel):
    """Request model for content comparison"""
    content_1: str = Field(..., min_length=1, description="First content to compare")
    content_2: str = Field(..., min_length=1, description="Second content to compare")
    content_id_1: str = Field(default="", description="Optional identifier for first content")
    content_id_2: str = Field(default="", description="Optional identifier for second content")
    comparison_type: str = Field(default="comprehensive", description="Type of comparison to perform")
    
    @validator('comparison_type')
    def validate_comparison_type(cls, v):
        allowed_types = ["comprehensive", "sentiment", "topics", "quality", "similarity"]
        if v not in allowed_types:
            raise ValueError(f'Comparison type must be one of: {allowed_types}')
        return v


# Response models
class AIContentAnalysisResponse(BaseModel):
    """Response model for AI content analysis"""
    content_id: str
    sentiment_score: float
    sentiment_label: str
    emotion_scores: Dict[str, float]
    topic_classification: Dict[str, float]
    language_detection: str
    readability_score: float
    complexity_score: float
    key_phrases: List[str]
    named_entities: List[Dict[str, Any]]
    content_quality_score: float
    ai_confidence: float
    analysis_timestamp: str


class ContentInsightsResponse(BaseModel):
    """Response model for content insights"""
    content_id: str
    summary: str
    main_topics: List[str]
    key_insights: List[str]
    recommendations: List[str]
    content_type: str
    target_audience: str
    engagement_prediction: float
    seo_score: float
    brand_voice_alignment: float
    analysis_timestamp: str


class ContentComparisonResponse(BaseModel):
    """Response model for content comparison"""
    content_1_analysis: AIContentAnalysisResponse
    content_2_analysis: AIContentAnalysisResponse
    comparison_metrics: Dict[str, Any]
    similarity_score: float
    differences: List[str]
    recommendations: List[str]
    comparison_timestamp: str


# Dependency functions
async def get_current_user() -> Dict[str, str]:
    """Dependency to get current user (placeholder for auth)"""
    return {"user_id": "anonymous", "role": "user"}


async def validate_api_key(api_key: Optional[str] = Query(None)) -> bool:
    """Dependency to validate API key"""
    # Placeholder for API key validation
    return True


# Route handlers
@router.post("/analyze", response_model=AIContentAnalysisResponse)
async def analyze_content_ai(
    request: ContentAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> AIContentAnalysisResponse:
    """
    Perform comprehensive AI analysis of content
    
    - **content**: Content text to analyze (max 10,000 characters)
    - **content_id**: Optional content identifier
    - **include_insights**: Whether to include detailed insights
    """
    
    try:
        # Perform AI analysis
        ai_analysis = await analyze_content_with_ai(
            content=request.content,
            content_id=request.content_id
        )
        
        # Convert to response model
        response = AIContentAnalysisResponse(
            content_id=ai_analysis.content_id,
            sentiment_score=ai_analysis.sentiment_score,
            sentiment_label=ai_analysis.sentiment_label,
            emotion_scores=ai_analysis.emotion_scores,
            topic_classification=ai_analysis.topic_classification,
            language_detection=ai_analysis.language_detection,
            readability_score=ai_analysis.readability_score,
            complexity_score=ai_analysis.complexity_score,
            key_phrases=ai_analysis.key_phrases,
            named_entities=ai_analysis.named_entities,
            content_quality_score=ai_analysis.content_quality_score,
            ai_confidence=ai_analysis.ai_confidence,
            analysis_timestamp=ai_analysis.analysis_timestamp.isoformat()
        )
        
        logger.info(f"AI analysis completed for content: {request.content_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in AI analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in AI content analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during AI analysis")


@router.post("/insights", response_model=ContentInsightsResponse)
async def generate_content_insights(
    request: ContentAnalysisRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> ContentInsightsResponse:
    """
    Generate comprehensive content insights using AI
    
    - **content**: Content text to analyze
    - **content_id**: Optional content identifier
    - **include_insights**: Whether to include detailed insights (always true for this endpoint)
    """
    
    try:
        # Generate AI insights
        insights = await generate_ai_insights(
            content=request.content,
            content_id=request.content_id
        )
        
        # Convert to response model
        response = ContentInsightsResponse(
            content_id=insights.content_id,
            summary=insights.summary,
            main_topics=insights.main_topics,
            key_insights=insights.key_insights,
            recommendations=insights.recommendations,
            content_type=insights.content_type,
            target_audience=insights.target_audience,
            engagement_prediction=insights.engagement_prediction,
            seo_score=insights.seo_score,
            brand_voice_alignment=insights.brand_voice_alignment,
            analysis_timestamp=insights.analysis_timestamp.isoformat()
        )
        
        logger.info(f"AI insights generated for content: {request.content_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in insights generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating AI insights: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during insights generation")


@router.post("/compare", response_model=ContentComparisonResponse)
async def compare_content_ai(
    request: ContentComparisonRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> ContentComparisonResponse:
    """
    Compare two pieces of content using AI analysis
    
    - **content_1**: First content to compare
    - **content_2**: Second content to compare
    - **content_id_1**: Optional identifier for first content
    - **content_id_2**: Optional identifier for second content
    - **comparison_type**: Type of comparison to perform
    """
    
    try:
        # Analyze both contents
        analysis_1 = await analyze_content_with_ai(request.content_1, request.content_id_1)
        analysis_2 = await analyze_content_with_ai(request.content_2, request.content_id_2)
        
        # Perform comparison based on type
        comparison_metrics = await _perform_content_comparison(
            analysis_1, analysis_2, request.comparison_type
        )
        
        # Calculate similarity score
        similarity_score = await _calculate_ai_similarity(analysis_1, analysis_2)
        
        # Generate differences and recommendations
        differences = await _identify_differences(analysis_1, analysis_2)
        recommendations = await _generate_comparison_recommendations(analysis_1, analysis_2)
        
        # Convert analyses to response models
        analysis_1_response = AIContentAnalysisResponse(
            content_id=analysis_1.content_id,
            sentiment_score=analysis_1.sentiment_score,
            sentiment_label=analysis_1.sentiment_label,
            emotion_scores=analysis_1.emotion_scores,
            topic_classification=analysis_1.topic_classification,
            language_detection=analysis_1.language_detection,
            readability_score=analysis_1.readability_score,
            complexity_score=analysis_1.complexity_score,
            key_phrases=analysis_1.key_phrases,
            named_entities=analysis_1.named_entities,
            content_quality_score=analysis_1.content_quality_score,
            ai_confidence=analysis_1.ai_confidence,
            analysis_timestamp=analysis_1.analysis_timestamp.isoformat()
        )
        
        analysis_2_response = AIContentAnalysisResponse(
            content_id=analysis_2.content_id,
            sentiment_score=analysis_2.sentiment_score,
            sentiment_label=analysis_2.sentiment_label,
            emotion_scores=analysis_2.emotion_scores,
            topic_classification=analysis_2.topic_classification,
            language_detection=analysis_2.language_detection,
            readability_score=analysis_2.readability_score,
            complexity_score=analysis_2.complexity_score,
            key_phrases=analysis_2.key_phrases,
            named_entities=analysis_2.named_entities,
            content_quality_score=analysis_2.content_quality_score,
            ai_confidence=analysis_2.ai_confidence,
            analysis_timestamp=analysis_2.analysis_timestamp.isoformat()
        )
        
        return ContentComparisonResponse(
            content_1_analysis=analysis_1_response,
            content_2_analysis=analysis_2_response,
            comparison_metrics=comparison_metrics,
            similarity_score=similarity_score,
            differences=differences,
            recommendations=recommendations,
            comparison_timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        logger.warning(f"Validation error in content comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in AI content comparison: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during content comparison")


@router.post("/batch-analyze")
async def batch_analyze_content(
    request: BatchAnalysisRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Perform batch AI analysis on multiple content items
    
    - **content_items**: List of content items to analyze
    - **include_insights**: Whether to include detailed insights
    - **batch_size**: Number of items to process per batch
    """
    
    try:
        results = []
        total_items = len(request.content_items)
        
        # Process in batches
        for i in range(0, total_items, request.batch_size):
            batch = request.content_items[i:i + request.batch_size]
            batch_results = []
            
            for item in batch:
                try:
                    content_id = item.get('id', f"batch_item_{i}")
                    content = item['content']
                    
                    # Perform analysis
                    analysis = await analyze_content_with_ai(content, content_id)
                    
                    # Include insights if requested
                    insights = None
                    if request.include_insights:
                        insights = await generate_ai_insights(content, content_id)
                    
                    batch_results.append({
                        "content_id": content_id,
                        "analysis": {
                            "sentiment_score": analysis.sentiment_score,
                            "sentiment_label": analysis.sentiment_label,
                            "readability_score": analysis.readability_score,
                            "content_quality_score": analysis.content_quality_score,
                            "ai_confidence": analysis.ai_confidence,
                            "key_phrases": analysis.key_phrases,
                            "main_topics": list(analysis.topic_classification.keys())[:3]
                        },
                        "insights": {
                            "summary": insights.summary if insights else None,
                            "recommendations": insights.recommendations if insights else None,
                            "content_type": insights.content_type if insights else None,
                            "target_audience": insights.target_audience if insights else None
                        } if insights else None
                    })
                    
                except Exception as e:
                    logger.error(f"Error analyzing batch item {i}: {e}")
                    batch_results.append({
                        "content_id": item.get('id', f"batch_item_{i}"),
                        "error": str(e)
                    })
            
            results.extend(batch_results)
        
        return {
            "total_items": total_items,
            "processed_items": len(results),
            "successful_analyses": len([r for r in results if 'error' not in r]),
            "failed_analyses": len([r for r in results if 'error' in r]),
            "results": results,
            "batch_size": request.batch_size,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Validation error in batch analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch AI analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch analysis")


@router.get("/health")
async def ai_health_check() -> Dict[str, Any]:
    """Health check endpoint for AI analysis service"""
    
    try:
        health_status = await get_ai_analyzer_health()
        
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
            "service": "ai-content-analyzer",
            "timestamp": datetime.now().isoformat(),
            "ai_analyzer": health_status
        }
        
    except Exception as e:
        logger.error(f"AI health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "ai-content-analyzer",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/models")
async def get_available_models() -> Dict[str, Any]:
    """Get information about available AI models"""
    
    try:
        health_status = await get_ai_analyzer_health()
        
        return {
            "available_models": health_status.get("available_models", {}),
            "device": health_status.get("device", "unknown"),
            "models_loaded": health_status.get("models_loaded", False),
            "model_info": {
                "sentiment_analyzer": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "emotion_analyzer": "j-hartmann/emotion-english-distilroberta-base",
                "topic_classifier": "facebook/bart-large-mnli",
                "language_detector": "papluca/xlm-roberta-base-language-detection",
                "ner_pipeline": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "summarizer": "facebook/bart-large-cnn",
                "sentence_transformer": "all-MiniLM-L6-v2"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model information: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


# Utility functions for route handlers
async def _perform_content_comparison(
    analysis_1: AIContentAnalysis, 
    analysis_2: AIContentAnalysis, 
    comparison_type: str
) -> Dict[str, Any]:
    """Perform content comparison based on type"""
    
    metrics = {}
    
    if comparison_type in ["comprehensive", "sentiment"]:
        metrics["sentiment_difference"] = abs(analysis_1.sentiment_score - analysis_2.sentiment_score)
        metrics["sentiment_1"] = analysis_1.sentiment_score
        metrics["sentiment_2"] = analysis_2.sentiment_score
    
    if comparison_type in ["comprehensive", "topics"]:
        # Compare topic classifications
        topics_1 = set(analysis_1.topic_classification.keys())
        topics_2 = set(analysis_2.topic_classification.keys())
        metrics["topic_overlap"] = len(topics_1.intersection(topics_2)) / len(topics_1.union(topics_2)) if topics_1.union(topics_2) else 0
        metrics["unique_topics_1"] = list(topics_1 - topics_2)
        metrics["unique_topics_2"] = list(topics_2 - topics_1)
    
    if comparison_type in ["comprehensive", "quality"]:
        metrics["quality_difference"] = abs(analysis_1.content_quality_score - analysis_2.content_quality_score)
        metrics["quality_1"] = analysis_1.content_quality_score
        metrics["quality_2"] = analysis_2.content_quality_score
        metrics["readability_difference"] = abs(analysis_1.readability_score - analysis_2.readability_score)
    
    if comparison_type in ["comprehensive", "similarity"]:
        # Compare key phrases
        phrases_1 = set(analysis_1.key_phrases)
        phrases_2 = set(analysis_2.key_phrases)
        metrics["phrase_overlap"] = len(phrases_1.intersection(phrases_2)) / len(phrases_1.union(phrases_2)) if phrases_1.union(phrases_2) else 0
    
    return metrics


async def _calculate_ai_similarity(analysis_1: AIContentAnalysis, analysis_2: AIContentAnalysis) -> float:
    """Calculate AI-based similarity score"""
    
    # Combine multiple similarity factors
    sentiment_sim = 1 - abs(analysis_1.sentiment_score - analysis_2.sentiment_score) / 2
    quality_sim = 1 - abs(analysis_1.content_quality_score - analysis_2.content_quality_score)
    readability_sim = 1 - abs(analysis_1.readability_score - analysis_2.readability_score) / 100
    
    # Topic similarity
    topics_1 = set(analysis_1.topic_classification.keys())
    topics_2 = set(analysis_2.topic_classification.keys())
    topic_sim = len(topics_1.intersection(topics_2)) / len(topics_1.union(topics_2)) if topics_1.union(topics_2) else 0
    
    # Key phrase similarity
    phrases_1 = set(analysis_1.key_phrases)
    phrases_2 = set(analysis_2.key_phrases)
    phrase_sim = len(phrases_1.intersection(phrases_2)) / len(phrases_1.union(phrases_2)) if phrases_1.union(phrases_2) else 0
    
    # Weighted average
    similarity = (
        sentiment_sim * 0.2 +
        quality_sim * 0.3 +
        readability_sim * 0.2 +
        topic_sim * 0.2 +
        phrase_sim * 0.1
    )
    
    return min(1.0, similarity)


async def _identify_differences(analysis_1: AIContentAnalysis, analysis_2: AIContentAnalysis) -> List[str]:
    """Identify key differences between content analyses"""
    
    differences = []
    
    # Sentiment differences
    sentiment_diff = abs(analysis_1.sentiment_score - analysis_2.sentiment_score)
    if sentiment_diff > 0.5:
        differences.append(f"Significant sentiment difference: {analysis_1.sentiment_label} vs {analysis_2.sentiment_label}")
    
    # Quality differences
    quality_diff = abs(analysis_1.content_quality_score - analysis_2.content_quality_score)
    if quality_diff > 0.3:
        differences.append(f"Content quality differs significantly: {analysis_1.content_quality_score:.2f} vs {analysis_2.content_quality_score:.2f}")
    
    # Readability differences
    readability_diff = abs(analysis_1.readability_score - analysis_2.readability_score)
    if readability_diff > 30:
        differences.append(f"Readability levels differ: {analysis_1.readability_score:.1f} vs {analysis_2.readability_score:.1f}")
    
    # Topic differences
    topics_1 = set(analysis_1.topic_classification.keys())
    topics_2 = set(analysis_2.topic_classification.keys())
    unique_topics_1 = topics_1 - topics_2
    unique_topics_2 = topics_2 - topics_1
    
    if unique_topics_1:
        differences.append(f"Content 1 has unique topics: {', '.join(list(unique_topics_1)[:3])}")
    if unique_topics_2:
        differences.append(f"Content 2 has unique topics: {', '.join(list(unique_topics_2)[:3])}")
    
    return differences


async def _generate_comparison_recommendations(analysis_1: AIContentAnalysis, analysis_2: AIContentAnalysis) -> List[str]:
    """Generate recommendations based on content comparison"""
    
    recommendations = []
    
    # Quality recommendations
    if analysis_1.content_quality_score > analysis_2.content_quality_score + 0.2:
        recommendations.append("Consider applying content structure techniques from content 1 to content 2")
    elif analysis_2.content_quality_score > analysis_1.content_quality_score + 0.2:
        recommendations.append("Consider applying content structure techniques from content 2 to content 1")
    
    # Readability recommendations
    readability_diff = abs(analysis_1.readability_score - analysis_2.readability_score)
    if readability_diff > 30:
        if analysis_1.readability_score > analysis_2.readability_score:
            recommendations.append("Content 1 is more readable - consider simplifying content 2")
        else:
            recommendations.append("Content 2 is more readable - consider simplifying content 1")
    
    # Sentiment recommendations
    sentiment_diff = abs(analysis_1.sentiment_score - analysis_2.sentiment_score)
    if sentiment_diff > 0.7:
        recommendations.append("Consider balancing emotional tone between both contents for consistency")
    
    # Topic recommendations
    topics_1 = set(analysis_1.topic_classification.keys())
    topics_2 = set(analysis_2.topic_classification.keys())
    if len(topics_1.intersection(topics_2)) < 2:
        recommendations.append("Contents have different topic focuses - consider aligning if targeting same audience")
    
    return recommendations




