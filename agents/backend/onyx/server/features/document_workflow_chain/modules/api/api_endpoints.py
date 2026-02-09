"""
API Endpoints for Document Workflow Chain
=========================================

This module provides REST API endpoints for managing document workflow chains,
including creation, continuation, and management of AI-powered document generation flows.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .workflow_chain_engine import WorkflowChainEngine, WorkflowChain, DocumentNode
from .dashboard import dashboard_router, initialize_dashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/document-workflow-chain", tags=["Document Workflow Chain"])

# Global engine instance (in production, this should be dependency injected)
workflow_engine = WorkflowChainEngine()
enhanced_workflow_engine = None  # Will be initialized as EnhancedWorkflowChainEngine

# Initialize dashboard
initialize_dashboard(workflow_engine)

# Pydantic models for request/response
class CreateWorkflowRequest(BaseModel):
    name: str = Field(..., description="Name of the workflow chain")
    description: str = Field(..., description="Description of the workflow purpose")
    initial_prompt: str = Field(..., description="Initial prompt to start the chain")
    settings: Optional[Dict[str, Any]] = Field(None, description="Optional settings for the workflow")

class ContinueWorkflowRequest(BaseModel):
    chain_id: str = Field(..., description="ID of the workflow chain to continue")
    continuation_prompt: Optional[str] = Field(None, description="Optional custom prompt for continuation")

class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    root_node_id: Optional[str]
    status: str
    node_count: int
    settings: Dict[str, Any]

class DocumentNodeResponse(BaseModel):
    id: str
    title: str
    content: str
    prompt: str
    generated_at: str
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict[str, Any]

class ChainHistoryResponse(BaseModel):
    chain_id: str
    documents: List[DocumentNodeResponse]
    total_documents: int

class GenerateTitleRequest(BaseModel):
    content: str = Field(..., description="Content to generate title from")

class TitleResponse(BaseModel):
    title: str
    generated_at: str

class TemplateWorkflowRequest(BaseModel):
    template_id: str = Field(..., description="ID of the content template to use")
    topic: str = Field(..., description="Main topic for the content")
    name: str = Field(..., description="Name of the workflow chain")
    description: str = Field(..., description="Description of the workflow")
    language_code: str = Field(default="en", description="Language code for content generation")
    word_count: Optional[int] = Field(None, description="Target word count")
    tone: Optional[str] = Field(None, description="Content tone")
    audience: Optional[str] = Field(None, description="Target audience")

class ContentAnalysisRequest(BaseModel):
    content: str = Field(..., description="Content to analyze")
    title: str = Field(..., description="Document title")
    language_code: str = Field(default="en", description="Language code")

class PerformanceAnalysisResponse(BaseModel):
    chain_id: str
    performance_summary: Dict[str, Any]
    predictive_insights: List[Dict[str, Any]]
    optimization_recommendations: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    analysis_timestamp: str

# New advanced feature models
class PromptOptimizationRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to optimize")
    optimization_goals: Optional[List[str]] = Field(None, description="List of optimization goals")
    target_length: Optional[int] = Field(None, description="Target length in tokens")

class IntelligentContentRequest(BaseModel):
    topic: str = Field(..., description="Content topic")
    content_type: str = Field("blog", description="Type of content")
    target_audience: str = Field("general", description="Target audience")
    tone: str = Field("professional", description="Content tone")
    length_preference: str = Field("medium", description="Length preference")
    quality_requirements: Optional[List[str]] = Field(None, description="Quality requirements")

class TrendAnalysisRequest(BaseModel):
    time_period: int = Field(30, description="Time period in days")
    categories: Optional[List[str]] = Field(None, description="Categories to analyze")

class MarketIntelligenceRequest(BaseModel):
    market_segment: str = Field(..., description="Market segment to analyze")
    time_period: int = Field(90, description="Time period in days")

class ExternalIntegrationRequest(BaseModel):
    service_name: str = Field(..., description="Name of external service")
    action: str = Field(..., description="Action to perform")
    data: Dict[str, Any] = Field(..., description="Data for the action")

class ContextOptimizationRequest(BaseModel):
    target_tokens: int = Field(8000, description="Target token count")

# API Endpoints

@router.post("/create", response_model=WorkflowResponse)
async def create_workflow_chain(request: CreateWorkflowRequest):
    """
    Create a new document workflow chain
    
    This endpoint creates a new workflow chain with an initial document
    generated from the provided prompt.
    """
    try:
        chain = await workflow_engine.create_workflow_chain(
            name=request.name,
            description=request.description,
            initial_prompt=request.initial_prompt,
            settings=request.settings
        )
        
        response = WorkflowResponse(
            id=chain.id,
            name=chain.name,
            description=chain.description,
            created_at=chain.created_at.isoformat(),
            updated_at=chain.updated_at.isoformat(),
            root_node_id=chain.root_node_id,
            status=chain.status,
            node_count=len(chain.nodes),
            settings=chain.settings
        )
        
        logger.info(f"Created workflow chain: {chain.id}")
        return response
        
    except Exception as e:
        logger.error(f"Error creating workflow chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create workflow chain: {str(e)}")

@router.post("/continue", response_model=DocumentNodeResponse)
async def continue_workflow_chain(request: ContinueWorkflowRequest):
    """
    Continue a workflow chain by generating the next document
    
    This endpoint generates a new document in the chain, using the previous
    document's content as context for the next generation.
    """
    try:
        node = await workflow_engine.continue_workflow_chain(
            chain_id=request.chain_id,
            continuation_prompt=request.continuation_prompt
        )
        
        response = DocumentNodeResponse(
            id=node.id,
            title=node.title,
            content=node.content,
            prompt=node.prompt,
            generated_at=node.generated_at.isoformat(),
            parent_id=node.parent_id,
            children_ids=node.children_ids,
            metadata=node.metadata
        )
        
        logger.info(f"Continued workflow chain {request.chain_id} with document: {node.id}")
        return response
        
    except ValueError as e:
        logger.error(f"Workflow chain not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error continuing workflow chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to continue workflow chain: {str(e)}")

@router.get("/chain/{chain_id}", response_model=WorkflowResponse)
async def get_workflow_chain(chain_id: str):
    """
    Get details of a specific workflow chain
    """
    try:
        chain = workflow_engine.get_workflow_chain(chain_id)
        if not chain:
            raise HTTPException(status_code=404, detail="Workflow chain not found")
        
        response = WorkflowResponse(
            id=chain.id,
            name=chain.name,
            description=chain.description,
            created_at=chain.created_at.isoformat(),
            updated_at=chain.updated_at.isoformat(),
            root_node_id=chain.root_node_id,
            status=chain.status,
            node_count=len(chain.nodes),
            settings=chain.settings
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow chain: {str(e)}")

@router.get("/chain/{chain_id}/history", response_model=ChainHistoryResponse)
async def get_chain_history(chain_id: str):
    """
    Get the complete history of a workflow chain
    """
    try:
        history = workflow_engine.get_chain_history(chain_id)
        if not history:
            raise HTTPException(status_code=404, detail="Workflow chain not found or has no history")
        
        documents = [
            DocumentNodeResponse(
                id=node.id,
                title=node.title,
                content=node.content,
                prompt=node.prompt,
                generated_at=node.generated_at.isoformat(),
                parent_id=node.parent_id,
                children_ids=node.children_ids,
                metadata=node.metadata
            )
            for node in history
        ]
        
        response = ChainHistoryResponse(
            chain_id=chain_id,
            documents=documents,
            total_documents=len(documents)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chain history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chain history: {str(e)}")

@router.get("/chains", response_model=List[WorkflowResponse])
async def get_all_active_chains():
    """
    Get all active workflow chains
    """
    try:
        chains = workflow_engine.get_all_active_chains()
        
        response = [
            WorkflowResponse(
                id=chain.id,
                name=chain.name,
                description=chain.description,
                created_at=chain.created_at.isoformat(),
                updated_at=chain.updated_at.isoformat(),
                root_node_id=chain.root_node_id,
                status=chain.status,
                node_count=len(chain.nodes),
                settings=chain.settings
            )
            for chain in chains
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting active chains: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active chains: {str(e)}")

@router.post("/generate-title", response_model=TitleResponse)
async def generate_blog_title(request: GenerateTitleRequest):
    """
    Generate a compelling blog title from content
    """
    try:
        title = await workflow_engine.generate_blog_title(request.content)
        
        response = TitleResponse(
            title=title,
            generated_at=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating title: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate title: {str(e)}")

@router.post("/chain/{chain_id}/pause")
async def pause_workflow_chain(chain_id: str):
    """
    Pause a workflow chain
    """
    try:
        success = workflow_engine.pause_workflow_chain(chain_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow chain not found")
        
        return JSONResponse(
            content={"message": f"Workflow chain {chain_id} paused successfully"},
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing workflow chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to pause workflow chain: {str(e)}")

@router.post("/chain/{chain_id}/resume")
async def resume_workflow_chain(chain_id: str):
    """
    Resume a paused workflow chain
    """
    try:
        success = workflow_engine.resume_workflow_chain(chain_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow chain not found or not paused")
        
        return JSONResponse(
            content={"message": f"Workflow chain {chain_id} resumed successfully"},
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming workflow chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resume workflow chain: {str(e)}")

@router.post("/chain/{chain_id}/complete")
async def complete_workflow_chain(chain_id: str):
    """
    Mark a workflow chain as completed
    """
    try:
        success = workflow_engine.complete_workflow_chain(chain_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow chain not found")
        
        return JSONResponse(
            content={"message": f"Workflow chain {chain_id} completed successfully"},
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing workflow chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to complete workflow chain: {str(e)}")

@router.get("/chain/{chain_id}/export")
async def export_workflow_chain(chain_id: str):
    """
    Export a workflow chain to JSON format
    """
    try:
        export_data = workflow_engine.export_workflow_chain(chain_id)
        if not export_data:
            raise HTTPException(status_code=404, detail="Workflow chain not found")
        
        return JSONResponse(
            content=export_data,
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting workflow chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export workflow chain: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint for the document workflow chain service
    """
    try:
        active_chains_count = len(workflow_engine.active_chains)
        history_chains_count = len(workflow_engine.chain_history)
        
        return JSONResponse(
            content={
                "status": "healthy",
                "active_chains": active_chains_count,
                "completed_chains": history_chains_count,
                "timestamp": datetime.now().isoformat()
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

# Advanced Features Endpoints

@router.post("/create-with-template", response_model=WorkflowResponse)
async def create_workflow_with_template(request: TemplateWorkflowRequest):
    """
    Create a new workflow chain using a content template
    """
    try:
        chain = await workflow_engine.create_workflow_with_template(
            template_id=request.template_id,
            topic=request.topic,
            name=request.name,
            description=request.description,
            language_code=request.language_code,
            word_count=request.word_count,
            tone=request.tone,
            audience=request.audience
        )
        
        response = WorkflowResponse(
            id=chain.id,
            name=chain.name,
            description=chain.description,
            created_at=chain.created_at.isoformat(),
            updated_at=chain.updated_at.isoformat(),
            root_node_id=chain.root_node_id,
            status=chain.status,
            node_count=len(chain.nodes),
            settings=chain.settings
        )
        
        logger.info(f"Created workflow chain with template: {chain.id}")
        return response
        
    except ValueError as e:
        logger.error(f"Template not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating workflow with template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create workflow with template: {str(e)}")

@router.post("/analyze-content")
async def analyze_content(request: ContentAnalysisRequest):
    """
    Analyze content using advanced content analysis
    """
    try:
        analysis_result = await workflow_engine.enhance_document_with_analysis(
            content=request.content,
            title=request.title,
            language_code=request.language_code
        )
        
        return JSONResponse(content=analysis_result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error analyzing content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze content: {str(e)}")

@router.get("/chain/{chain_id}/performance", response_model=PerformanceAnalysisResponse)
async def get_workflow_performance(chain_id: str):
    """
    Get comprehensive performance analysis for a workflow chain
    """
    try:
        performance_analysis = await workflow_engine.analyze_workflow_performance(chain_id)
        
        if "error" in performance_analysis:
            raise HTTPException(status_code=404, detail=performance_analysis["error"])
        
        response = PerformanceAnalysisResponse(**performance_analysis)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow performance: {str(e)}")

@router.get("/chain/{chain_id}/insights")
async def get_workflow_insights(chain_id: str):
    """
    Get comprehensive insights for a workflow chain
    """
    try:
        insights = await workflow_engine.get_workflow_insights(chain_id)
        
        if "error" in insights:
            raise HTTPException(status_code=404, detail=insights["error"])
        
        return JSONResponse(content=insights, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow insights: {str(e)}")

@router.get("/templates")
async def get_available_templates(category: Optional[str] = None):
    """
    Get available content templates
    """
    try:
        templates = await workflow_engine.get_available_templates(category)
        return JSONResponse(content={"templates": templates}, status_code=200)
        
    except Exception as e:
        logger.error(f"Error getting templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.get("/languages")
async def get_supported_languages():
    """
    Get supported languages for content generation
    """
    try:
        languages = await workflow_engine.get_supported_languages()
        return JSONResponse(content={"languages": languages}, status_code=200)
        
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported languages: {str(e)}")

@router.get("/analytics/summary")
async def get_analytics_summary():
    """
    Get overall analytics summary
    """
    try:
        # Get performance summary for all chains
        summary = await workflow_engine.analytics.get_performance_summary(None, "30d")
        
        # Get insights for all chains
        insights = await workflow_engine.analytics.generate_predictive_insights(None)
        
        # Get recommendations
        recommendations = await workflow_engine.analytics.get_optimization_recommendations()
        
        return JSONResponse(content={
            "performance_summary": summary,
            "predictive_insights": [asdict(insight) for insight in insights],
            "optimization_recommendations": recommendations,
            "summary_timestamp": datetime.now().isoformat()
        }, status_code=200)
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics summary: {str(e)}")

# Advanced Feature Endpoints

@router.post("/optimize-prompt")
async def optimize_prompt(request: PromptOptimizationRequest):
    """
    Optimize a prompt using AI optimization techniques
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.prompt_optimizer.optimize_prompt(
            prompt=request.prompt,
            target_length=request.target_length,
            optimization_goals=request.optimization_goals
        )
        
        return JSONResponse(content={
            "success": True,
            "original_prompt": result.original_prompt,
            "optimized_prompt": result.optimized_prompt,
            "improvement_score": result.improvement_score,
            "tokens_saved": result.tokens_saved,
            "quality_improvement": result.expected_quality_improvement,
            "optimization_type": result.optimization_type,
            "metadata": result.metadata
        }, status_code=200)
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize prompt: {str(e)}")

@router.post("/generate-intelligent-content")
async def generate_intelligent_content(request: IntelligentContentRequest):
    """
    Generate content using intelligent generation system
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.generate_intelligent_content(
            topic=request.topic,
            content_type=request.content_type,
            target_audience=request.target_audience,
            tone=request.tone,
            length_preference=request.length_preference,
            quality_requirements=request.quality_requirements
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error generating intelligent content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate intelligent content: {str(e)}")

@router.post("/analyze-trends")
async def analyze_trends(request: TrendAnalysisRequest):
    """
    Analyze trends for content strategy
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.trend_analyzer.analyze_trends(
            time_period=request.time_period,
            categories=request.categories
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze trends: {str(e)}")

@router.post("/market-intelligence")
async def get_market_intelligence(request: MarketIntelligenceRequest):
    """
    Get market intelligence for content strategy
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.get_market_intelligence(
            market_segment=request.market_segment,
            time_period=request.time_period
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error getting market intelligence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get market intelligence: {str(e)}")

@router.post("/external-integration")
async def integrate_external_service(request: ExternalIntegrationRequest):
    """
    Integrate with external services
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.integrate_external_service(
            service_name=request.service_name,
            action=request.action,
            data=request.data
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error integrating with external service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to integrate with external service: {str(e)}")

@router.post("/chain/{chain_id}/optimize-context")
async def optimize_context_window(chain_id: str, request: ContextOptimizationRequest):
    """
    Optimize context window for a workflow
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.optimize_context_window(
            workflow_id=chain_id,
            target_tokens=request.target_tokens
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error optimizing context window: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize context window: {str(e)}")

@router.get("/chain/{chain_id}/comprehensive-insights")
async def get_comprehensive_workflow_insights(chain_id: str):
    """
    Get comprehensive insights for a workflow using all advanced features
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.get_comprehensive_workflow_insights(
            workflow_id=chain_id
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error getting comprehensive workflow insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get comprehensive workflow insights: {str(e)}")

@router.post("/chain/{chain_id}/analyze-trends")
async def analyze_workflow_trends(chain_id: str, request: TrendAnalysisRequest):
    """
    Analyze trends for a specific workflow
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.analyze_trends_for_workflow(
            workflow_id=chain_id,
            time_period=request.time_period,
            categories=request.categories
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error analyzing workflow trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze workflow trends: {str(e)}")

@router.post("/chain/{chain_id}/optimize-prompt")
async def optimize_workflow_prompt(chain_id: str, request: PromptOptimizationRequest):
    """
    Optimize prompt for a specific workflow
    """
    try:
        # Initialize enhanced engine if not already done
        if not enhanced_workflow_engine:
            from .workflow_chain_engine import EnhancedWorkflowChainEngine
            global enhanced_workflow_engine
            enhanced_workflow_engine = EnhancedWorkflowChainEngine()
            await enhanced_workflow_engine.initialize()
        
        result = await enhanced_workflow_engine.optimize_prompt_for_workflow(
            prompt=request.prompt,
            workflow_id=chain_id,
            optimization_goals=request.optimization_goals
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error optimizing workflow prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize workflow prompt: {str(e)}")

# Include dashboard router
router.include_router(dashboard_router)

# Advanced Integration Endpoints
@router.post("/enhanced-workflow")
async def create_enhanced_workflow(request: CreateWorkflowRequest):
    """Create a workflow with all advanced features enabled"""
    try:
        from .advanced_integration import advanced_integration
        
        result = await advanced_integration.create_enhanced_workflow(
            workflow_name=request.name,
            initial_prompt=request.initial_prompt,
            author=request.settings.get("author") if request.settings else None,
            enable_quality_control=request.settings.get("enable_quality_control", True) if request.settings else True,
            enable_versioning=request.settings.get("enable_versioning", True) if request.settings else True,
            enable_scheduling=request.settings.get("enable_scheduling", False) if request.settings else False,
            schedule_config=request.settings.get("schedule_config") if request.settings else None
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Enhanced workflow created successfully",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Error creating enhanced workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflow/{workflow_id}/execute-with-quality")
async def execute_workflow_with_quality_control(
    workflow_id: str,
    input_data: Optional[Dict[str, Any]] = None,
    quality_threshold: float = 0.7,
    auto_fix_issues: bool = True
):
    """Execute workflow with automatic quality control"""
    try:
        from .advanced_integration import advanced_integration
        
        result = await advanced_integration.execute_workflow_with_quality_control(
            workflow_id=workflow_id,
            input_data=input_data,
            quality_threshold=quality_threshold,
            auto_fix_issues=auto_fix_issues
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Workflow executed with quality control",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Error executing workflow with quality control: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflow/{workflow_id}/analytics-dashboard")
async def get_workflow_analytics_dashboard(
    workflow_id: str,
    days: int = 30
):
    """Get comprehensive analytics dashboard for a workflow"""
    try:
        from .advanced_integration import advanced_integration
        
        dashboard = await advanced_integration.get_workflow_analytics_dashboard(
            workflow_id=workflow_id,
            days=days
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Analytics dashboard generated successfully",
            "data": dashboard
        })
        
    except Exception as e:
        logger.error(f"Error generating analytics dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflow/{workflow_id}/optimize")
async def optimize_workflow_performance(
    workflow_id: str,
    optimization_goals: Optional[List[str]] = None
):
    """Optimize workflow performance using all available tools"""
    try:
        from .advanced_integration import advanced_integration
        
        if optimization_goals is None:
            optimization_goals = ["quality", "efficiency", "cost"]
            
        optimizations = await advanced_integration.optimize_workflow_performance(
            workflow_id=workflow_id,
            optimization_goals=optimization_goals
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Workflow optimization completed",
            "data": optimizations
        })
        
    except Exception as e:
        logger.error(f"Error optimizing workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Workflow Templates Endpoints
@router.post("/templates")
async def create_workflow_template(
    template_name: str,
    template_config: Dict[str, Any],
    author: Optional[str] = None
):
    """Create a reusable workflow template"""
    try:
        from .advanced_integration import advanced_integration
        
        template_id = await advanced_integration.create_workflow_template(
            template_name=template_name,
            template_config=template_config,
            author=author
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Workflow template created successfully",
            "data": {"template_id": template_id}
        })
        
    except Exception as e:
        logger.error(f"Error creating workflow template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_workflow_templates():
    """Get all available workflow templates"""
    try:
        from .advanced_integration import advanced_integration
        
        templates = await advanced_integration.get_workflow_templates()
        
        return JSONResponse(content={
            "success": True,
            "message": "Workflow templates retrieved successfully",
            "data": templates
        })
        
    except Exception as e:
        logger.error(f"Error retrieving workflow templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/templates/{template_id}/instantiate")
async def instantiate_workflow_from_template(
    template_id: str,
    workflow_name: str,
    customizations: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None
):
    """Create a new workflow from a template"""
    try:
        from .advanced_integration import advanced_integration
        
        result = await advanced_integration.instantiate_workflow_from_template(
            template_id=template_id,
            workflow_name=workflow_name,
            customizations=customizations,
            author=author
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Workflow instantiated from template successfully",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Error instantiating workflow from template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@router.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        content={"detail": "Resource not found"},
        status_code=404
    )

@router.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        content={"detail": "Internal server error"},
        status_code=500
    )
