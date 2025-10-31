"""
Content Intelligence Routes - Advanced content intelligence and insights API
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field, validator
import json

from ..core.content_intelligence_engine import (
    analyze_content_intelligence,
    analyze_content_trends,
    generate_content_insights,
    generate_content_strategy,
    get_content_intelligence_health,
    initialize_content_intelligence_engine,
    ContentIntelligence,
    ContentTrend,
    ContentInsight,
    ContentStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/intelligence", tags=["Content Intelligence"])


# Pydantic models for request/response validation
class IntelligenceAnalysisRequest(BaseModel):
    """Request model for content intelligence analysis"""
    content: str = Field(..., min_length=1, max_length=20000, description="Content text to analyze")
    content_id: str = Field(default="", description="Optional content identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for analysis")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()


class TrendsAnalysisRequest(BaseModel):
    """Request model for content trends analysis"""
    content_list: List[str] = Field(..., min_items=2, max_items=100, description="List of content items to analyze")
    time_period: str = Field(default="30d", description="Time period for trend analysis")
    
    @validator('content_list')
    def validate_content_list(cls, v):
        for content in v:
            if not content.strip():
                raise ValueError('Content items cannot be empty')
        return v


class InsightsGenerationRequest(BaseModel):
    """Request model for content insights generation"""
    content: str = Field(..., min_length=1, max_length=20000, description="Content text to analyze")
    content_id: str = Field(default="", description="Optional content identifier")
    insight_types: List[str] = Field(default=["engagement", "viral", "seo"], description="Types of insights to generate")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()


class StrategyGenerationRequest(BaseModel):
    """Request model for content strategy generation"""
    content: str = Field(..., min_length=1, max_length=20000, description="Content text to analyze")
    goals: List[str] = Field(default=["engagement", "seo", "conversion"], description="Strategy goals")
    priority_focus: str = Field(default="balanced", description="Priority focus for strategy")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()


# Response models
class ContentIntelligenceResponse(BaseModel):
    """Response model for content intelligence analysis"""
    content_id: str
    intelligence_score: float
    content_type: str
    complexity_level: str
    target_audience: str
    engagement_potential: float
    viral_potential: float
    seo_potential: float
    conversion_potential: float
    brand_alignment: float
    content_gaps: List[str]
    improvement_opportunities: List[str]
    competitive_advantages: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    analysis_timestamp: str


class ContentTrendResponse(BaseModel):
    """Response model for content trend"""
    trend_type: str
    trend_score: float
    trend_direction: str
    trend_confidence: float
    related_keywords: List[str]
    trend_impact: str
    trend_timeline: str
    trend_source: str


class ContentInsightResponse(BaseModel):
    """Response model for content insight"""
    insight_type: str
    insight_score: float
    insight_description: str
    insight_evidence: List[str]
    insight_impact: str
    insight_confidence: float
    insight_recommendations: List[str]


class ContentStrategyResponse(BaseModel):
    """Response model for content strategy"""
    strategy_type: str
    strategy_priority: int
    strategy_description: str
    strategy_goals: List[str]
    strategy_tactics: List[str]
    strategy_metrics: List[str]
    strategy_timeline: str
    strategy_resources: List[str]
    expected_impact: float


# Dependency functions
async def get_current_user() -> Dict[str, str]:
    """Dependency to get current user (placeholder for auth)"""
    return {"user_id": "anonymous", "role": "user"}


async def validate_api_key(api_key: Optional[str] = Query(None)) -> bool:
    """Dependency to validate API key"""
    # Placeholder for API key validation
    return True


# Route handlers
@router.post("/analyze", response_model=ContentIntelligenceResponse)
async def analyze_content_intelligence_endpoint(
    request: IntelligenceAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> ContentIntelligenceResponse:
    """
    Perform comprehensive content intelligence analysis
    
    - **content**: Content text to analyze (max 20,000 characters)
    - **content_id**: Optional content identifier
    - **context**: Additional context for analysis
    """
    
    try:
        # Perform content intelligence analysis
        intelligence = await analyze_content_intelligence(
            content=request.content,
            content_id=request.content_id,
            context=request.context
        )
        
        # Convert to response model
        response = ContentIntelligenceResponse(
            content_id=intelligence.content_id,
            intelligence_score=intelligence.intelligence_score,
            content_type=intelligence.content_type,
            complexity_level=intelligence.complexity_level,
            target_audience=intelligence.target_audience,
            engagement_potential=intelligence.engagement_potential,
            viral_potential=intelligence.viral_potential,
            seo_potential=intelligence.seo_potential,
            conversion_potential=intelligence.conversion_potential,
            brand_alignment=intelligence.brand_alignment,
            content_gaps=intelligence.content_gaps,
            improvement_opportunities=intelligence.improvement_opportunities,
            competitive_advantages=intelligence.competitive_advantages,
            risk_factors=intelligence.risk_factors,
            recommendations=intelligence.recommendations,
            analysis_timestamp=intelligence.analysis_timestamp.isoformat()
        )
        
        logger.info(f"Content intelligence analysis completed for: {request.content_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in content intelligence analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in content intelligence analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during content intelligence analysis")


@router.post("/trends", response_model=List[ContentTrendResponse])
async def analyze_content_trends_endpoint(
    request: TrendsAnalysisRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> List[ContentTrendResponse]:
    """
    Analyze content trends across multiple pieces of content
    
    - **content_list**: List of content items to analyze
    - **time_period**: Time period for trend analysis
    """
    
    try:
        # Perform content trends analysis
        trends = await analyze_content_trends(
            content_list=request.content_list,
            time_period=request.time_period
        )
        
        # Convert to response models
        trends_response = [
            ContentTrendResponse(
                trend_type=trend.trend_type,
                trend_score=trend.trend_score,
                trend_direction=trend.trend_direction,
                trend_confidence=trend.trend_confidence,
                related_keywords=trend.related_keywords,
                trend_impact=trend.trend_impact,
                trend_timeline=trend.trend_timeline,
                trend_source=trend.trend_source
            )
            for trend in trends
        ]
        
        logger.info(f"Content trends analysis completed for {len(request.content_list)} items")
        return trends_response
        
    except ValueError as e:
        logger.warning(f"Validation error in content trends analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in content trends analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during content trends analysis")


@router.post("/insights", response_model=List[ContentInsightResponse])
async def generate_content_insights_endpoint(
    request: InsightsGenerationRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> List[ContentInsightResponse]:
    """
    Generate detailed content insights
    
    - **content**: Content text to analyze
    - **content_id**: Optional content identifier
    - **insight_types**: Types of insights to generate
    """
    
    try:
        # Generate content insights
        insights = await generate_content_insights(
            content=request.content,
            content_id=request.content_id
        )
        
        # Filter insights by requested types
        filtered_insights = [
            insight for insight in insights
            if insight.insight_type in request.insight_types
        ]
        
        # Convert to response models
        insights_response = [
            ContentInsightResponse(
                insight_type=insight.insight_type,
                insight_score=insight.insight_score,
                insight_description=insight.insight_description,
                insight_evidence=insight.insight_evidence,
                insight_impact=insight.insight_impact,
                insight_confidence=insight.insight_confidence,
                insight_recommendations=insight.insight_recommendations
            )
            for insight in filtered_insights
        ]
        
        logger.info(f"Content insights generated for: {request.content_id}")
        return insights_response
        
    except ValueError as e:
        logger.warning(f"Validation error in content insights generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating content insights: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during content insights generation")


@router.post("/strategy", response_model=List[ContentStrategyResponse])
async def generate_content_strategy_endpoint(
    request: StrategyGenerationRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> List[ContentStrategyResponse]:
    """
    Generate content strategy recommendations
    
    - **content**: Content text to analyze
    - **goals**: Strategy goals to focus on
    - **priority_focus**: Priority focus for strategy
    """
    
    try:
        # Generate content strategy
        strategies = await generate_content_strategy(
            content=request.content,
            goals=request.goals
        )
        
        # Sort strategies by priority
        strategies.sort(key=lambda x: x.strategy_priority)
        
        # Convert to response models
        strategies_response = [
            ContentStrategyResponse(
                strategy_type=strategy.strategy_type,
                strategy_priority=strategy.strategy_priority,
                strategy_description=strategy.strategy_description,
                strategy_goals=strategy.strategy_goals,
                strategy_tactics=strategy.strategy_tactics,
                strategy_metrics=strategy.strategy_metrics,
                strategy_timeline=strategy.strategy_timeline,
                strategy_resources=strategy.strategy_resources,
                expected_impact=strategy.expected_impact
            )
            for strategy in strategies
        ]
        
        logger.info(f"Content strategy generated for content with {len(request.goals)} goals")
        return strategies_response
        
    except ValueError as e:
        logger.warning(f"Validation error in content strategy generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating content strategy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during content strategy generation")


@router.get("/intelligence-score")
async def get_intelligence_score(
    content: str = Query(..., description="Content to analyze for intelligence score"),
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get content intelligence score and breakdown
    
    - **content**: Content text to analyze
    """
    
    try:
        # Analyze content intelligence
        intelligence = await analyze_content_intelligence(content, "score_request")
        
        return {
            "intelligence_score": intelligence.intelligence_score,
            "score_breakdown": {
                "engagement_potential": intelligence.engagement_potential,
                "viral_potential": intelligence.viral_potential,
                "seo_potential": intelligence.seo_potential,
                "conversion_potential": intelligence.conversion_potential,
                "brand_alignment": intelligence.brand_alignment
            },
            "content_analysis": {
                "content_type": intelligence.content_type,
                "complexity_level": intelligence.complexity_level,
                "target_audience": intelligence.target_audience
            },
            "recommendations_count": len(intelligence.recommendations),
            "analysis_timestamp": intelligence.analysis_timestamp.isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Validation error in intelligence score calculation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error calculating intelligence score: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during intelligence score calculation")


@router.get("/content-types")
async def get_content_types() -> Dict[str, Any]:
    """Get available content types and their descriptions"""
    
    content_types = {
        "tutorial": {
            "description": "Step-by-step instructional content",
            "characteristics": ["how-to", "guide", "tutorial", "instructions"],
            "best_for": ["Educational content", "Product demos", "Training materials"],
            "engagement_tips": ["Use clear steps", "Include examples", "Add visuals"]
        },
        "news": {
            "description": "Current events and news content",
            "characteristics": ["breaking", "update", "announcement", "news"],
            "best_for": ["Industry updates", "Company news", "Market insights"],
            "engagement_tips": ["Be timely", "Include sources", "Add context"]
        },
        "review": {
            "description": "Opinion and analysis content",
            "characteristics": ["review", "opinion", "analysis", "thoughts"],
            "best_for": ["Product reviews", "Service evaluations", "Expert opinions"],
            "engagement_tips": ["Be honest", "Include pros/cons", "Add personal experience"]
        },
        "story": {
            "description": "Narrative and storytelling content",
            "characteristics": ["story", "narrative", "experience", "journey"],
            "best_for": ["Brand stories", "Case studies", "Personal experiences"],
            "engagement_tips": ["Use storytelling", "Include emotions", "Create connection"]
        },
        "commercial": {
            "description": "Sales and marketing content",
            "characteristics": ["product", "service", "buy", "purchase"],
            "best_for": ["Product descriptions", "Sales pages", "Marketing content"],
            "engagement_tips": ["Highlight benefits", "Include CTAs", "Add social proof"]
        },
        "support": {
            "description": "Help and support content",
            "characteristics": ["help", "support", "faq", "question"],
            "best_for": ["Customer support", "FAQ sections", "Help documentation"],
            "engagement_tips": ["Be clear", "Include examples", "Anticipate questions"]
        },
        "general": {
            "description": "General purpose content",
            "characteristics": ["informational", "general", "overview"],
            "best_for": ["Blog posts", "Articles", "General information"],
            "engagement_tips": ["Be informative", "Use clear structure", "Add value"]
        }
    }
    
    return {
        "content_types": content_types,
        "total_types": len(content_types),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/audience-types")
async def get_audience_types() -> Dict[str, Any]:
    """Get available audience types and their characteristics"""
    
    audience_types = {
        "professionals": {
            "description": "Working professionals and experts",
            "characteristics": ["experienced", "technical", "business-focused"],
            "content_preferences": ["In-depth analysis", "Industry insights", "Professional tone"],
            "engagement_style": ["Data-driven", "Evidence-based", "Professional"]
        },
        "students": {
            "description": "Students and learners",
            "characteristics": ["learning", "curious", "academic"],
            "content_preferences": ["Educational content", "Clear explanations", "Examples"],
            "engagement_style": ["Interactive", "Visual", "Step-by-step"]
        },
        "general_public": {
            "description": "General audience with varied backgrounds",
            "characteristics": ["diverse", "accessible", "broad"],
            "content_preferences": ["Easy to understand", "Relevant", "Engaging"],
            "engagement_style": ["Conversational", "Relatable", "Inclusive"]
        },
        "experts": {
            "description": "Subject matter experts and specialists",
            "characteristics": ["knowledgeable", "specialized", "technical"],
            "content_preferences": ["Advanced topics", "Technical details", "Research"],
            "engagement_style": ["Detailed", "Technical", "Evidence-based"]
        },
        "beginners": {
            "description": "Newcomers to the topic or field",
            "characteristics": ["new", "learning", "basic"],
            "content_preferences": ["Basic concepts", "Simple explanations", "Foundation"],
            "engagement_style": ["Simple", "Clear", "Supportive"]
        },
        "business_owners": {
            "description": "Business owners and entrepreneurs",
            "characteristics": ["business-focused", "results-oriented", "practical"],
            "content_preferences": ["Business insights", "ROI focus", "Practical advice"],
            "engagement_style": ["Results-focused", "Practical", "Business-oriented"]
        },
        "consumers": {
            "description": "End consumers and customers",
            "characteristics": ["customer-focused", "value-oriented", "practical"],
            "content_preferences": ["Product benefits", "User experience", "Value"],
            "engagement_style": ["Benefit-focused", "User-friendly", "Value-driven"]
        },
        "developers": {
            "description": "Software developers and technical professionals",
            "characteristics": ["technical", "code-focused", "problem-solving"],
            "content_preferences": ["Technical details", "Code examples", "Documentation"],
            "engagement_style": ["Technical", "Code-focused", "Problem-solving"]
        }
    }
    
    return {
        "audience_types": audience_types,
        "total_types": len(audience_types),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health")
async def intelligence_health_check() -> Dict[str, Any]:
    """Health check endpoint for content intelligence service"""
    
    try:
        health_status = await get_content_intelligence_health()
        
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
            "service": "content-intelligence",
            "timestamp": datetime.now().isoformat(),
            "intelligence_engine": health_status
        }
        
    except Exception as e:
        logger.error(f"Intelligence health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "content-intelligence",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/capabilities")
async def get_intelligence_capabilities() -> Dict[str, Any]:
    """Get content intelligence capabilities and features"""
    
    capabilities = {
        "analysis_types": {
            "content_intelligence": "Comprehensive content intelligence analysis",
            "trends_analysis": "Content trends and pattern analysis",
            "insights_generation": "Detailed content insights and recommendations",
            "strategy_generation": "Content strategy recommendations"
        },
        "intelligence_metrics": {
            "intelligence_score": "Overall content intelligence score",
            "engagement_potential": "Potential for reader engagement",
            "viral_potential": "Potential for viral sharing",
            "seo_potential": "Search engine optimization potential",
            "conversion_potential": "Potential for conversions",
            "brand_alignment": "Alignment with brand guidelines"
        },
        "content_analysis": {
            "content_type_detection": "Automatic content type identification",
            "complexity_analysis": "Content complexity level assessment",
            "audience_analysis": "Target audience identification",
            "gap_analysis": "Content gaps identification",
            "improvement_opportunities": "Improvement opportunities detection",
            "competitive_advantages": "Competitive advantages identification",
            "risk_assessment": "Content risk factors analysis"
        },
        "insights_types": {
            "engagement_insights": "Engagement-focused insights and recommendations",
            "viral_insights": "Viral potential insights and strategies",
            "seo_insights": "SEO optimization insights and recommendations",
            "conversion_insights": "Conversion optimization insights",
            "brand_insights": "Brand alignment insights and recommendations"
        },
        "strategy_types": {
            "engagement_strategy": "Strategies to improve content engagement",
            "seo_strategy": "Strategies to optimize content for SEO",
            "conversion_strategy": "Strategies to improve content conversions",
            "viral_strategy": "Strategies to increase viral potential",
            "brand_strategy": "Strategies to improve brand alignment"
        }
    }
    
    return {
        "capabilities": capabilities,
        "total_capabilities": sum(len(cap) for cap in capabilities.values()),
        "timestamp": datetime.now().isoformat()
    }


# Startup and shutdown handlers
@router.on_event("startup")
async def startup_intelligence_service():
    """Initialize intelligence service on startup"""
    try:
        await initialize_content_intelligence_engine()
        logger.info("Content intelligence service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize content intelligence service: {e}")


@router.on_event("shutdown")
async def shutdown_intelligence_service():
    """Shutdown intelligence service on shutdown"""
    try:
        logger.info("Content intelligence service shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown content intelligence service: {e}")




