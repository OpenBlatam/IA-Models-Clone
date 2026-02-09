"""
Comprehensive Content API - Advanced Content Management and Intelligence
===================================================================

This module provides a unified API for all content-related operations including:
- Content intelligence and analysis
- Content generation and optimization
- Workflow automation
- Performance analytics
- Multi-platform integration
- Real-time content processing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import our engines
from content_intelligence_engine import ContentIntelligenceEngine, ContentType, ContentMetrics
from content_generation_engine import ContentGenerationEngine, ContentRequest, GeneratedContent
from content_workflow_engine import ContentWorkflowEngine, WorkflowDefinition, WorkflowExecution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global engines
intelligence_engine = None
generation_engine = None
workflow_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global intelligence_engine, generation_engine, workflow_engine
    
    # Initialize engines
    config = {
        "openai_api_key": "your-openai-api-key",
        "anthropic_api_key": "your-anthropic-api-key",
        "use_local_models": True,
        "max_workers": 10
    }
    
    intelligence_engine = ContentIntelligenceEngine(config)
    generation_engine = ContentGenerationEngine(config)
    workflow_engine = ContentWorkflowEngine(config)
    
    logger.info("All engines initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down engines")

# FastAPI app
app = FastAPI(
    title="Comprehensive Content API",
    description="Advanced AI-powered content management and intelligence platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ContentAnalysisRequest(BaseModel):
    content: str
    content_type: str = "article"
    include_insights: bool = True
    include_recommendations: bool = True

class ContentGenerationRequest(BaseModel):
    topic: str
    format: str = "article"
    tone: str = "professional"
    length: str = "medium"
    target_audience: str = "general"
    keywords: List[str] = []
    requirements: List[str] = []
    include_cta: bool = True
    seo_optimized: bool = True

class WorkflowExecutionRequest(BaseModel):
    template_id: str
    name: str
    description: str
    variables: Dict[str, Any] = {}

class ContentOptimizationRequest(BaseModel):
    content_id: str
    optimization_goals: List[str] = ["seo", "engagement", "readability"]

class BatchAnalysisRequest(BaseModel):
    content_batch: List[Dict[str, Any]]

class ContentInsightResponse(BaseModel):
    content_id: str
    metrics: Dict[str, Any]
    insights: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime

class ContentGenerationResponse(BaseModel):
    content_id: str
    title: str
    content: str
    meta_description: str
    tags: List[str]
    metrics: Dict[str, Any]
    generated_at: datetime

class WorkflowExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: str
    status: str
    started_at: datetime
    tasks: List[Dict[str, Any]]

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - in production, implement proper JWT validation"""
    # This is a simplified version - implement proper authentication
    return {"user_id": "demo_user", "role": "admin"}

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Comprehensive Content API",
        "version": "1.0.0",
        "features": [
            "Content Intelligence & Analysis",
            "AI Content Generation",
            "Workflow Automation",
            "Performance Analytics",
            "Multi-platform Integration"
        ],
        "endpoints": {
            "analysis": "/api/v1/analyze",
            "generation": "/api/v1/generate",
            "workflows": "/api/v1/workflows",
            "optimization": "/api/v1/optimize",
            "batch": "/api/v1/batch",
            "analytics": "/api/v1/analytics"
        }
    }

@app.post("/api/v1/analyze", response_model=ContentInsightResponse)
async def analyze_content(
    request: ContentAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze content for intelligence and insights"""
    try:
        # Convert string to enum
        content_type = ContentType(request.content_type)
        
        # Analyze content
        metrics = await intelligence_engine.analyze_content(request.content, content_type)
        
        # Generate insights
        insights = []
        if request.include_insights:
            insights = await intelligence_engine.generate_insights(
                f"content_{uuid.uuid4()}", request.content, metrics
            )
        
        # Generate recommendations
        recommendations = []
        if request.include_recommendations:
            for insight in insights:
                recommendations.extend(insight.recommendations)
        
        return ContentInsightResponse(
            content_id=f"analysis_{uuid.uuid4()}",
            metrics={
                "word_count": metrics.word_count,
                "readability_score": metrics.readability_score,
                "sentiment_score": metrics.sentiment_score,
                "engagement_score": metrics.engagement_score,
                "seo_score": metrics.seo_score,
                "quality_score": metrics.quality_score
            },
            insights=[
                {
                    "type": insight.insight_type,
                    "value": insight.insight_value,
                    "confidence": insight.confidence,
                    "explanation": insight.explanation
                }
                for insight in insights
            ],
            recommendations=recommendations,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate", response_model=ContentGenerationResponse)
async def generate_content(
    request: ContentGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate AI-powered content"""
    try:
        # Convert strings to enums
        from content_generation_engine import ContentFormat, ContentTone, ContentLength
        
        content_request = ContentRequest(
            topic=request.topic,
            format=ContentFormat(request.format),
            tone=ContentTone(request.tone),
            length=ContentLength(request.length),
            target_audience=request.target_audience,
            keywords=request.keywords,
            requirements=request.requirements,
            include_cta=request.include_cta,
            seo_optimized=request.seo_optimized
        )
        
        # Generate content
        generated_content = await generation_engine.generate_content(content_request)
        
        return ContentGenerationResponse(
            content_id=generated_content.content_id,
            title=generated_content.title,
            content=generated_content.content,
            meta_description=generated_content.meta_description,
            tags=generated_content.tags,
            metrics={
                "word_count": generated_content.word_count,
                "readability_score": generated_content.readability_score,
                "seo_score": generated_content.seo_score,
                "engagement_score": generated_content.engagement_score,
                "generation_time": generated_content.generation_time
            },
            generated_at=generated_content.generated_at
        )
        
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute a content workflow"""
    try:
        # Create workflow from template
        workflow = await workflow_engine.create_workflow(
            template_id=request.template_id,
            name=request.name,
            description=request.description,
            variables=request.variables
        )
        
        # Execute workflow
        execution = await workflow_engine.execute_workflow(
            workflow_id=workflow.workflow_id,
            variables=request.variables
        )
        
        return WorkflowExecutionResponse(
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            status=execution.status.value,
            started_at=execution.started_at,
            tasks=[
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "status": task.status.value,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }
                for task in execution.tasks
            ]
        )
        
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/workflows/{execution_id}/status")
async def get_workflow_status(
    execution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get workflow execution status"""
    try:
        status = await workflow_engine.get_workflow_status(execution_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/optimize")
async def optimize_content(
    request: ContentOptimizationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Optimize existing content"""
    try:
        # This would integrate with the optimization engine
        # For now, return a mock response
        return {
            "content_id": request.content_id,
            "optimization_goals": request.optimization_goals,
            "optimized_metrics": {
                "seo_score": 0.85,
                "engagement_score": 0.78,
                "readability_score": 82.0
            },
            "optimizations_applied": [
                "Improved keyword density",
                "Enhanced readability",
                "Added engagement elements"
            ],
            "optimized_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error optimizing content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch/analyze")
async def batch_analyze_content(
    request: BatchAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze multiple content pieces in batch"""
    try:
        results = []
        
        for content_item in request.content_batch:
            content = content_item.get('content', '')
            content_type = ContentType(content_item.get('type', 'article'))
            
            metrics = await intelligence_engine.analyze_content(content, content_type)
            insights = await intelligence_engine.generate_insights(
                content_item.get('id', f"batch_{uuid.uuid4()}"), content, metrics
            )
            
            results.append({
                "content_id": content_item.get('id', f"batch_{uuid.uuid4()}"),
                "metrics": {
                    "word_count": metrics.word_count,
                    "readability_score": metrics.readability_score,
                    "sentiment_score": metrics.sentiment_score,
                    "engagement_score": metrics.engagement_score,
                    "seo_score": metrics.seo_score,
                    "quality_score": metrics.quality_score
                },
                "insights": [
                    {
                        "type": insight.insight_type,
                        "value": insight.insight_value,
                        "confidence": insight.confidence,
                        "explanation": insight.explanation
                    }
                    for insight in insights
                ]
            })
        
        return {
            "batch_id": f"batch_{uuid.uuid4()}",
            "total_analyzed": len(results),
            "results": results,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/trends")
async def get_content_trends(
    time_period: str = "30d",
    current_user: dict = Depends(get_current_user)
):
    """Get content trends and analytics"""
    try:
        # This would integrate with actual analytics data
        # For now, return mock data
        return {
            "time_period": time_period,
            "total_content": 1250,
            "average_engagement": 0.72,
            "average_quality": 0.68,
            "top_topics": [
                {"topic": "AI", "count": 45},
                {"topic": "Machine Learning", "count": 38},
                {"topic": "Data Science", "count": 32}
            ],
            "sentiment_distribution": {
                "positive": 0.65,
                "neutral": 0.25,
                "negative": 0.10
            },
            "performance_trends": {
                "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "scores": [0.68, 0.72, 0.75]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting content trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/workflows/{workflow_id}")
async def get_workflow_analytics(
    workflow_id: str,
    time_period: str = "30d",
    current_user: dict = Depends(get_current_user)
):
    """Get workflow analytics"""
    try:
        analytics = await workflow_engine.get_workflow_analytics(workflow_id, time_period)
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting workflow analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/templates")
async def get_workflow_templates(current_user: dict = Depends(get_current_user)):
    """Get available workflow templates"""
    try:
        templates = []
        for template_id, template in workflow_engine.templates.items():
            templates.append({
                "template_id": template_id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "task_count": len(template.tasks),
                "trigger_count": len(template.triggers)
            })
        
        return {"templates": templates}
        
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/{execution_id}/pause")
async def pause_workflow(
    execution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Pause a running workflow"""
    try:
        success = await workflow_engine.pause_workflow(execution_id)
        return {"success": success, "message": "Workflow paused" if success else "Failed to pause workflow"}
        
    except Exception as e:
        logger.error(f"Error pausing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/{execution_id}/resume")
async def resume_workflow(
    execution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Resume a paused workflow"""
    try:
        success = await workflow_engine.resume_workflow(execution_id)
        return {"success": success, "message": "Workflow resumed" if success else "Failed to resume workflow"}
        
    except Exception as e:
        logger.error(f"Error resuming workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/{execution_id}/cancel")
async def cancel_workflow(
    execution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Cancel a running workflow"""
    try:
        success = await workflow_engine.cancel_workflow(execution_id)
        return {"success": success, "message": "Workflow cancelled" if success else "Failed to cancel workflow"}
        
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "engines": {
            "intelligence_engine": intelligence_engine is not None,
            "generation_engine": generation_engine is not None,
            "workflow_engine": workflow_engine is not None
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "comprehensive_content_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )