"""
Analytics Routes - Advanced content analytics and business intelligence API
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field, validator
import json

from ..core.content_analytics_engine import (
    analyze_content_analytics,
    generate_business_intelligence,
    create_content_dashboard,
    generate_content_report,
    get_analytics_engine_health,
    initialize_content_analytics_engine,
    ContentAnalytics,
    BusinessIntelligence,
    ContentDashboard,
    ContentReport
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["Content Analytics"])


# Pydantic models for request/response validation
class AnalyticsAnalysisRequest(BaseModel):
    """Request model for content analytics analysis"""
    content: str = Field(..., min_length=1, max_length=20000, description="Content text to analyze")
    content_id: str = Field(default="", description="Optional content identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for analysis")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()


class BusinessIntelligenceRequest(BaseModel):
    """Request model for business intelligence analysis"""
    content_list: List[str] = Field(..., min_items=1, max_items=100, description="List of content items to analyze")
    time_period: str = Field(default="30d", description="Time period for analysis")
    
    @validator('content_list')
    def validate_content_list(cls, v):
        for content in v:
            if not content.strip():
                raise ValueError('Content items cannot be empty')
        return v


class DashboardRequest(BaseModel):
    """Request model for dashboard creation"""
    content_list: List[str] = Field(..., min_items=1, max_items=100, description="List of content items for dashboard")
    dashboard_config: Dict[str, Any] = Field(default_factory=dict, description="Dashboard configuration")
    
    @validator('content_list')
    def validate_content_list(cls, v):
        for content in v:
            if not content.strip():
                raise ValueError('Content items cannot be empty')
        return v


class ReportRequest(BaseModel):
    """Request model for report generation"""
    content_list: List[str] = Field(..., min_items=1, max_items=100, description="List of content items for report")
    report_type: str = Field(default="comprehensive", description="Type of report to generate")
    report_period: str = Field(default="30d", description="Time period for report")
    
    @validator('content_list')
    def validate_content_list(cls, v):
        for content in v:
            if not content.strip():
                raise ValueError('Content items cannot be empty')
        return v
    
    @validator('report_type')
    def validate_report_type(cls, v):
        allowed_types = ["comprehensive", "performance", "quality", "seo", "audience"]
        if v not in allowed_types:
            raise ValueError(f'Report type must be one of: {allowed_types}')
        return v


# Response models
class ContentAnalyticsResponse(BaseModel):
    """Response model for content analytics analysis"""
    content_id: str
    analytics_timestamp: str
    content_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    engagement_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    seo_metrics: Dict[str, Any]
    audience_metrics: Dict[str, Any]
    competitive_metrics: Dict[str, Any]
    trend_metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]


class BusinessIntelligenceResponse(BaseModel):
    """Response model for business intelligence analysis"""
    analysis_id: str
    analysis_timestamp: str
    content_performance: Dict[str, Any]
    audience_insights: Dict[str, Any]
    market_trends: Dict[str, Any]
    competitive_analysis: Dict[str, Any]
    roi_metrics: Dict[str, Any]
    growth_metrics: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    strategic_recommendations: List[str]


class ContentDashboardResponse(BaseModel):
    """Response model for content dashboard"""
    dashboard_id: str
    dashboard_timestamp: str
    overview_metrics: Dict[str, Any]
    content_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    audience_metrics: Dict[str, Any]
    trend_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    charts_data: Dict[str, Any]
    kpis: Dict[str, Any]


class ContentReportResponse(BaseModel):
    """Response model for content report"""
    report_id: str
    report_timestamp: str
    report_type: str
    report_period: str
    executive_summary: str
    key_findings: List[str]
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    appendices: Dict[str, Any]
    metadata: Dict[str, Any]


# Dependency functions
async def get_current_user() -> Dict[str, str]:
    """Dependency to get current user (placeholder for auth)"""
    return {"user_id": "anonymous", "role": "user"}


async def validate_api_key(api_key: Optional[str] = Query(None)) -> bool:
    """Dependency to validate API key"""
    # Placeholder for API key validation
    return True


# Route handlers
@router.post("/analyze", response_model=ContentAnalyticsResponse)
async def analyze_content_analytics_endpoint(
    request: AnalyticsAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> ContentAnalyticsResponse:
    """
    Perform comprehensive content analytics analysis
    
    - **content**: Content text to analyze (max 20,000 characters)
    - **content_id**: Optional content identifier
    - **context**: Additional context for analysis
    """
    
    try:
        # Perform content analytics analysis
        analytics = await analyze_content_analytics(
            content=request.content,
            content_id=request.content_id,
            context=request.context
        )
        
        # Convert to response model
        response = ContentAnalyticsResponse(
            content_id=analytics.content_id,
            analytics_timestamp=analytics.analytics_timestamp.isoformat(),
            content_metrics=analytics.content_metrics,
            performance_metrics=analytics.performance_metrics,
            engagement_metrics=analytics.engagement_metrics,
            quality_metrics=analytics.quality_metrics,
            seo_metrics=analytics.seo_metrics,
            audience_metrics=analytics.audience_metrics,
            competitive_metrics=analytics.competitive_metrics,
            trend_metrics=analytics.trend_metrics,
            insights=analytics.insights,
            recommendations=analytics.recommendations
        )
        
        logger.info(f"Content analytics analysis completed for: {request.content_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in content analytics analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in content analytics analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during content analytics analysis")


@router.post("/business-intelligence", response_model=BusinessIntelligenceResponse)
async def generate_business_intelligence_endpoint(
    request: BusinessIntelligenceRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> BusinessIntelligenceResponse:
    """
    Generate business intelligence analysis for multiple content pieces
    
    - **content_list**: List of content items to analyze
    - **time_period**: Time period for analysis
    """
    
    try:
        # Generate business intelligence
        bi_analysis = await generate_business_intelligence(
            content_list=request.content_list,
            time_period=request.time_period
        )
        
        # Convert to response model
        response = BusinessIntelligenceResponse(
            analysis_id=bi_analysis.analysis_id,
            analysis_timestamp=bi_analysis.analysis_timestamp.isoformat(),
            content_performance=bi_analysis.content_performance,
            audience_insights=bi_analysis.audience_insights,
            market_trends=bi_analysis.market_trends,
            competitive_analysis=bi_analysis.competitive_analysis,
            roi_metrics=bi_analysis.roi_metrics,
            growth_metrics=bi_analysis.growth_metrics,
            risk_assessment=bi_analysis.risk_assessment,
            strategic_recommendations=bi_analysis.strategic_recommendations
        )
        
        logger.info(f"Business intelligence analysis completed for {len(request.content_list)} content pieces")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in business intelligence analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in business intelligence analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during business intelligence analysis")


@router.post("/dashboard", response_model=ContentDashboardResponse)
async def create_dashboard_endpoint(
    request: DashboardRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> ContentDashboardResponse:
    """
    Create content analytics dashboard
    
    - **content_list**: List of content items for dashboard
    - **dashboard_config**: Dashboard configuration options
    """
    
    try:
        # Create content dashboard
        dashboard = await create_content_dashboard(
            content_list=request.content_list,
            dashboard_config=request.dashboard_config
        )
        
        # Convert to response model
        response = ContentDashboardResponse(
            dashboard_id=dashboard.dashboard_id,
            dashboard_timestamp=dashboard.dashboard_timestamp.isoformat(),
            overview_metrics=dashboard.overview_metrics,
            content_metrics=dashboard.content_metrics,
            performance_metrics=dashboard.performance_metrics,
            audience_metrics=dashboard.audience_metrics,
            trend_metrics=dashboard.trend_metrics,
            alerts=dashboard.alerts,
            charts_data=dashboard.charts_data,
            kpis=dashboard.kpis
        )
        
        logger.info(f"Content dashboard created for {len(request.content_list)} content pieces")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in dashboard creation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during dashboard creation")


@router.post("/report", response_model=ContentReportResponse)
async def generate_report_endpoint(
    request: ReportRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> ContentReportResponse:
    """
    Generate comprehensive content analytics report
    
    - **content_list**: List of content items for report
    - **report_type**: Type of report to generate
    - **report_period**: Time period for report
    """
    
    try:
        # Generate content report
        report = await generate_content_report(
            content_list=request.content_list,
            report_type=request.report_type,
            report_period=request.report_period
        )
        
        # Convert to response model
        response = ContentReportResponse(
            report_id=report.report_id,
            report_timestamp=report.report_timestamp.isoformat(),
            report_type=report.report_type,
            report_period=report.report_period,
            executive_summary=report.executive_summary,
            key_findings=report.key_findings,
            detailed_analysis=report.detailed_analysis,
            recommendations=report.recommendations,
            appendices=report.appendices,
            metadata=report.metadata
        )
        
        logger.info(f"Content report generated for {len(request.content_list)} content pieces")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in report generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during report generation")


@router.get("/metrics")
async def get_analytics_metrics(
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get analytics engine metrics"""
    
    try:
        # Get analytics metrics
        health_status = await get_analytics_engine_health()
        
        return {
            "analytics_engine": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during metrics retrieval")


@router.get("/kpi-definitions")
async def get_kpi_definitions() -> Dict[str, Any]:
    """Get KPI definitions and targets"""
    
    kpi_definitions = {
        "content_quality": {
            "readability_score": {
                "description": "Content readability score based on Flesch Reading Ease",
                "target": 70,
                "weight": 0.3,
                "unit": "score",
                "range": "0-100"
            },
            "sentiment_score": {
                "description": "Overall sentiment of the content",
                "target": 0.7,
                "weight": 0.2,
                "unit": "score",
                "range": "-1 to 1"
            },
            "uniqueness_score": {
                "description": "Content uniqueness and originality",
                "target": 0.8,
                "weight": 0.3,
                "unit": "score",
                "range": "0-1"
            },
            "completeness_score": {
                "description": "Content completeness and structure",
                "target": 0.9,
                "weight": 0.2,
                "unit": "score",
                "range": "0-1"
            }
        },
        "performance": {
            "engagement_rate": {
                "description": "User engagement rate with content",
                "target": 0.15,
                "weight": 0.4,
                "unit": "percentage",
                "range": "0-1"
            },
            "conversion_rate": {
                "description": "Content conversion rate",
                "target": 0.05,
                "weight": 0.3,
                "unit": "percentage",
                "range": "0-1"
            },
            "bounce_rate": {
                "description": "Content bounce rate",
                "target": 0.3,
                "weight": 0.2,
                "unit": "percentage",
                "range": "0-1"
            },
            "time_on_page": {
                "description": "Average time spent on content",
                "target": 180,
                "weight": 0.1,
                "unit": "seconds",
                "range": "0-∞"
            }
        },
        "seo": {
            "organic_traffic": {
                "description": "Organic traffic to content",
                "target": 1000,
                "weight": 0.4,
                "unit": "visitors",
                "range": "0-∞"
            },
            "keyword_rankings": {
                "description": "Average keyword ranking position",
                "target": 10,
                "weight": 0.3,
                "unit": "position",
                "range": "1-∞"
            },
            "backlinks": {
                "description": "Number of backlinks to content",
                "target": 50,
                "weight": 0.2,
                "unit": "links",
                "range": "0-∞"
            },
            "page_speed": {
                "description": "Page loading speed",
                "target": 2.0,
                "weight": 0.1,
                "unit": "seconds",
                "range": "0-∞"
            }
        }
    }
    
    return {
        "kpi_definitions": kpi_definitions,
        "total_kpis": sum(len(category) for category in kpi_definitions.values()),
        "categories": list(kpi_definitions.keys()),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/alert-rules")
async def get_alert_rules() -> Dict[str, Any]:
    """Get alert rules and thresholds"""
    
    alert_rules = {
        "performance_alerts": {
            "low_engagement": {
                "description": "Alert when engagement rate is below threshold",
                "threshold": 0.05,
                "severity": "warning",
                "action": "review_content_strategy"
            },
            "high_bounce_rate": {
                "description": "Alert when bounce rate exceeds threshold",
                "threshold": 0.7,
                "severity": "critical",
                "action": "optimize_content_engagement"
            },
            "low_conversion": {
                "description": "Alert when conversion rate is below threshold",
                "threshold": 0.01,
                "severity": "warning",
                "action": "improve_call_to_action"
            }
        },
        "quality_alerts": {
            "low_readability": {
                "description": "Alert when readability score is below threshold",
                "threshold": 50,
                "severity": "warning",
                "action": "improve_content_readability"
            },
            "negative_sentiment": {
                "description": "Alert when sentiment is negative",
                "threshold": -0.5,
                "severity": "critical",
                "action": "review_content_tone"
            },
            "duplicate_content": {
                "description": "Alert when content similarity is high",
                "threshold": 0.8,
                "severity": "warning",
                "action": "create_unique_content"
            }
        },
        "seo_alerts": {
            "traffic_drop": {
                "description": "Alert when traffic drops significantly",
                "threshold": -0.2,
                "severity": "critical",
                "action": "investigate_seo_issues"
            },
            "ranking_drop": {
                "description": "Alert when keyword rankings drop",
                "threshold": -5,
                "severity": "warning",
                "action": "optimize_keywords"
            },
            "slow_page_speed": {
                "description": "Alert when page speed is slow",
                "threshold": 5.0,
                "severity": "warning",
                "action": "optimize_page_speed"
            }
        }
    }
    
    return {
        "alert_rules": alert_rules,
        "total_rules": sum(len(category) for category in alert_rules.values()),
        "severity_levels": ["info", "warning", "critical"],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/report-templates")
async def get_report_templates() -> Dict[str, Any]:
    """Get available report templates"""
    
    report_templates = {
        "comprehensive": {
            "description": "Complete content analytics report with all metrics",
            "sections": [
                "executive_summary",
                "content_performance",
                "audience_insights",
                "market_trends",
                "competitive_analysis",
                "roi_analysis",
                "recommendations"
            ],
            "estimated_time": "5-10 minutes",
            "content_limit": 100
        },
        "performance": {
            "description": "Focus on content performance metrics",
            "sections": [
                "performance_overview",
                "engagement_metrics",
                "conversion_analysis",
                "performance_recommendations"
            ],
            "estimated_time": "2-5 minutes",
            "content_limit": 50
        },
        "quality": {
            "description": "Focus on content quality assessment",
            "sections": [
                "quality_overview",
                "readability_analysis",
                "content_structure",
                "quality_recommendations"
            ],
            "estimated_time": "2-5 minutes",
            "content_limit": 50
        },
        "seo": {
            "description": "Focus on SEO optimization analysis",
            "sections": [
                "seo_overview",
                "keyword_analysis",
                "technical_seo",
                "seo_recommendations"
            ],
            "estimated_time": "3-7 minutes",
            "content_limit": 50
        },
        "audience": {
            "description": "Focus on audience insights and demographics",
            "sections": [
                "audience_overview",
                "demographics_analysis",
                "behavior_insights",
                "audience_recommendations"
            ],
            "estimated_time": "3-7 minutes",
            "content_limit": 50
        }
    }
    
    return {
        "report_templates": report_templates,
        "total_templates": len(report_templates),
        "default_template": "comprehensive",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/capabilities")
async def get_analytics_capabilities() -> Dict[str, Any]:
    """Get analytics engine capabilities and features"""
    
    capabilities = {
        "analysis_types": {
            "content_analytics": "Comprehensive content analytics analysis",
            "business_intelligence": "Business intelligence and strategic insights",
            "dashboard_creation": "Interactive content analytics dashboards",
            "report_generation": "Detailed content analytics reports"
        },
        "metrics_categories": {
            "content_metrics": "Word count, readability, vocabulary richness, structure",
            "performance_metrics": "Views, engagement, conversion, bounce rate",
            "engagement_metrics": "Likes, shares, comments, bookmarks",
            "quality_metrics": "Completeness, uniqueness, depth, structure",
            "seo_metrics": "Keywords, meta tags, links, page speed",
            "audience_metrics": "Demographics, audience type, targeting",
            "competitive_metrics": "Market position, differentiation, advantage",
            "trend_metrics": "Trend alignment, viral potential, relevance"
        },
        "business_intelligence": {
            "content_performance": "Aggregated content performance analysis",
            "audience_insights": "Audience behavior and demographics",
            "market_trends": "Market trend analysis and alignment",
            "competitive_analysis": "Competitive positioning and advantage",
            "roi_metrics": "Return on investment calculations",
            "growth_metrics": "Growth rate and improvement tracking",
            "risk_assessment": "Content risk identification and mitigation"
        },
        "dashboard_features": {
            "overview_metrics": "High-level performance overview",
            "content_metrics": "Detailed content quality metrics",
            "performance_metrics": "Engagement and conversion metrics",
            "audience_metrics": "Audience insights and demographics",
            "trend_metrics": "Trend analysis and viral potential",
            "alerts": "Automated alerts and notifications",
            "charts_data": "Visual charts and graphs data",
            "kpis": "Key performance indicators tracking"
        },
        "report_features": {
            "executive_summary": "High-level executive summary",
            "key_findings": "Important insights and discoveries",
            "detailed_analysis": "Comprehensive analysis breakdown",
            "recommendations": "Actionable recommendations",
            "appendices": "Supporting data and methodology",
            "metadata": "Report generation information"
        }
    }
    
    return {
        "capabilities": capabilities,
        "total_capabilities": sum(len(cap) for cap in capabilities.values()),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health")
async def analytics_health_check() -> Dict[str, Any]:
    """Health check endpoint for analytics service"""
    
    try:
        health_status = await get_analytics_engine_health()
        
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
            "service": "content-analytics",
            "timestamp": datetime.now().isoformat(),
            "analytics_engine": health_status
        }
        
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "content-analytics",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# Startup and shutdown handlers
@router.on_event("startup")
async def startup_analytics_service():
    """Initialize analytics service on startup"""
    try:
        await initialize_content_analytics_engine()
        logger.info("Content analytics service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize content analytics service: {e}")


@router.on_event("shutdown")
async def shutdown_analytics_service():
    """Shutdown analytics service on shutdown"""
    try:
        logger.info("Content analytics service shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown content analytics service: {e}")