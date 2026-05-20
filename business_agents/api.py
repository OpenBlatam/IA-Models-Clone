"""
Business Agents API
===================

FastAPI endpoints for the Business Agents system.
Provides REST API for workflow management, document generation, and agent coordination.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from .business_agents import BusinessAgentManager, BusinessArea, BusinessAgent, AgentCapability
from .workflow_engine import Workflow, WorkflowStatus, StepType
from .document_generator import DocumentType, DocumentFormat, DocumentRequest, GeneratedDocument
from .nlp_api import router as nlp_router
from .advanced_nlp_system import advanced_nlp_system
from .enhanced_nlp_api import router as enhanced_nlp_router
from .enhanced_nlp_system import enhanced_nlp_system
from .optimal_nlp_api import router as optimal_nlp_router
from .optimal_nlp_system import optimal_nlp_system
from .ultra_fast_api import router as ultra_fast_router
from .ultra_fast_nlp import ultra_fast_nlp
from .ultra_quality_api import router as ultra_quality_router
from .ultra_quality_nlp import ultra_quality_nlp
from .ml_nlp_api import router as ml_nlp_router
from .ml_nlp_system import ml_nlp_system
from .performance_nlp_api import router as performance_nlp_router
from .performance_nlp_system import performance_nlp_system
from .advanced_nlp_api import router as advanced_nlp_router
from .advanced_nlp_system import advanced_nlp_system
from .comprehensive_nlp_api import router as comprehensive_nlp_router
from .comprehensive_nlp_system import comprehensive_nlp_system
from .superior_nlp_api import router as superior_nlp_router
from .superior_nlp_system import superior_nlp_system
from .cutting_edge_nlp_api import router as cutting_edge_nlp_router
from .cutting_edge_nlp_system import cutting_edge_nlp_system
from .revolutionary_nlp_api import router as revolutionary_nlp_router
from .revolutionary_nlp_system import revolutionary_nlp_system
from .next_gen_nlp_api import router as next_gen_nlp_router
from .next_gen_nlp_system import next_gen_nlp_system
from .supreme_nlp_api import router as supreme_nlp_router
from .supreme_nlp_system import supreme_nlp_system

logger = logging.getLogger(__name__)

# Initialize the business agent manager
agent_manager = BusinessAgentManager()

# Create API router
router = APIRouter(prefix="/business-agents", tags=["Business Agents"])

# Include NLP routers
router.include_router(nlp_router)
router.include_router(enhanced_nlp_router)
router.include_router(optimal_nlp_router)
router.include_router(ultra_fast_router)
router.include_router(ultra_quality_router)
router.include_router(ml_nlp_router)
router.include_router(performance_nlp_router)
router.include_router(advanced_nlp_router)
router.include_router(comprehensive_nlp_router)
router.include_router(superior_nlp_router)
router.include_router(cutting_edge_nlp_router)
router.include_router(revolutionary_nlp_router)
router.include_router(next_gen_nlp_router)
router.include_router(supreme_nlp_router)

# Pydantic models for API requests/responses
class AgentCapabilityRequest(BaseModel):
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    parameters: Dict[str, Any] = {}
    estimated_duration: int = 300

class BusinessAgentRequest(BaseModel):
    name: str
    business_area: BusinessArea
    description: str
    capabilities: List[AgentCapabilityRequest]
    is_active: bool = True
    metadata: Dict[str, Any] = {}

class WorkflowStepRequest(BaseModel):
    name: str
    step_type: StepType
    description: str
    agent_type: str
    parameters: Dict[str, Any] = {}
    conditions: Optional[Dict[str, Any]] = None
    next_steps: List[str] = []
    parallel_steps: List[str] = []
    max_retries: int = 3
    timeout: int = 300

class WorkflowRequest(BaseModel):
    name: str
    description: str
    business_area: BusinessArea
    steps: List[WorkflowStepRequest]
    variables: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class DocumentRequestModel(BaseModel):
    document_type: DocumentType
    title: str
    description: str
    business_area: str
    variables: Dict[str, Any] = {}
    template_id: Optional[str] = None
    format: DocumentFormat = DocumentFormat.MARKDOWN
    priority: str = "normal"
    deadline: Optional[datetime] = None

class CapabilityExecutionRequest(BaseModel):
    agent_id: str
    capability_name: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = {}

# API Endpoints

@router.get("/", response_model=Dict[str, Any])
async def get_system_overview():
    """Get overview of the business agents system."""
    
    agents = agent_manager.list_agents()
    workflows = agent_manager.list_workflows()
    business_areas = agent_manager.get_business_areas()
    
    return {
        "system_name": "Business Agents System",
        "version": "1.0.0",
        "total_agents": len(agents),
        "active_agents": len([a for a in agents if a.is_active]),
        "total_workflows": len(workflows),
        "business_areas": [area.value for area in business_areas],
        "capabilities": {
            "workflow_management": True,
            "document_generation": True,
            "agent_coordination": True,
            "real_time_execution": True
        }
    }

# Agent Management Endpoints

@router.get("/agents", response_model=List[Dict[str, Any]])
async def list_agents(
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    is_active: Optional[bool] = Query(None, description="Filter by active status")
):
    """List all business agents with optional filters."""
    
    agents = agent_manager.list_agents(business_area=business_area, is_active=is_active)
    
    return [
        {
            "id": agent.id,
            "name": agent.name,
            "business_area": agent.business_area.value,
            "description": agent.description,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_types": cap.input_types,
                    "output_types": cap.output_types,
                    "estimated_duration": cap.estimated_duration
                }
                for cap in agent.capabilities
            ],
            "is_active": agent.is_active,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat()
        }
        for agent in agents
    ]

@router.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: str):
    """Get specific agent details."""
    
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return {
        "id": agent.id,
        "name": agent.name,
        "business_area": agent.business_area.value,
        "description": agent.description,
        "capabilities": [
            {
                "name": cap.name,
                "description": cap.description,
                "input_types": cap.input_types,
                "output_types": cap.output_types,
                "parameters": cap.parameters,
                "estimated_duration": cap.estimated_duration
            }
            for cap in agent.capabilities
        ],
        "is_active": agent.is_active,
        "created_at": agent.created_at.isoformat(),
        "updated_at": agent.updated_at.isoformat(),
        "metadata": agent.metadata
    }

@router.get("/agents/{agent_id}/capabilities", response_model=List[Dict[str, Any]])
async def get_agent_capabilities(agent_id: str):
    """Get capabilities for a specific agent."""
    
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    capabilities = agent_manager.get_agent_capabilities(agent_id)
    
    return [
        {
            "name": cap.name,
            "description": cap.description,
            "input_types": cap.input_types,
            "output_types": cap.output_types,
            "parameters": cap.parameters,
            "estimated_duration": cap.estimated_duration
        }
        for cap in capabilities
    ]

@router.post("/agents/{agent_id}/execute", response_model=Dict[str, Any])
async def execute_agent_capability(
    agent_id: str,
    request: CapabilityExecutionRequest,
    background_tasks: BackgroundTasks
):
    """Execute a specific agent capability."""
    
    try:
        result = await agent_manager.execute_agent_capability(
            agent_id=request.agent_id,
            capability_name=request.capability_name,
            inputs=request.inputs,
            parameters=request.parameters
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Capability execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Workflow Management Endpoints

@router.get("/workflows", response_model=List[Dict[str, Any]])
async def list_workflows(
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    status: Optional[WorkflowStatus] = Query(None, description="Filter by status")
):
    """List all workflows with optional filters."""
    
    workflows = agent_manager.list_workflows(business_area=business_area, created_by=created_by)
    
    # Apply status filter if provided
    if status:
        workflows = [w for w in workflows if w.status == status]
    
    return [
        {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "business_area": workflow.business_area,
            "status": workflow.status.value,
            "created_by": workflow.created_by,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "steps_count": len(workflow.steps),
            "variables": workflow.variables
        }
        for workflow in workflows
    ]

@router.get("/workflows/{workflow_id}", response_model=Dict[str, Any])
async def get_workflow(workflow_id: str):
    """Get specific workflow details."""
    
    workflow = agent_manager.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "business_area": workflow.business_area,
        "status": workflow.status.value,
        "created_by": workflow.created_by,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat(),
        "variables": workflow.variables,
        "metadata": workflow.metadata,
        "steps": [
            {
                "id": step.id,
                "name": step.name,
                "step_type": step.step_type.value,
                "description": step.description,
                "agent_type": step.agent_type,
                "parameters": step.parameters,
                "conditions": step.conditions,
                "next_steps": step.next_steps,
                "parallel_steps": step.parallel_steps,
                "max_retries": step.max_retries,
                "timeout": step.timeout,
                "status": step.status,
                "created_at": step.created_at.isoformat() if step.created_at else None,
                "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                "error_message": step.error_message
            }
            for step in workflow.steps
        ]
    }

@router.post("/workflows", response_model=Dict[str, Any])
async def create_workflow(
    request: WorkflowRequest,
    created_by: str = Query(..., description="User creating the workflow")
):
    """Create a new workflow."""
    
    try:
        # Convert request to workflow engine format
        steps = [
            {
                "name": step.name,
                "step_type": step.step_type.value,
                "description": step.description,
                "agent_type": step.agent_type,
                "parameters": step.parameters,
                "conditions": step.conditions,
                "next_steps": step.next_steps,
                "parallel_steps": step.parallel_steps,
                "max_retries": step.max_retries,
                "timeout": step.timeout
            }
            for step in request.steps
        ]
        
        workflow = await agent_manager.create_business_workflow(
            name=request.name,
            description=request.description,
            business_area=request.business_area,
            steps=steps,
            created_by=created_by,
            variables=request.variables
        )
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "business_area": workflow.business_area,
            "status": workflow.status.value,
            "created_by": workflow.created_by,
            "created_at": workflow.created_at.isoformat(),
            "steps_count": len(workflow.steps)
        }
        
    except Exception as e:
        logger.error(f"Workflow creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create workflow")

@router.post("/workflows/{workflow_id}/execute", response_model=Dict[str, Any])
async def execute_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks
):
    """Execute a workflow."""
    
    try:
        result = await agent_manager.execute_business_workflow(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "execution_results": result,
            "executed_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute workflow")

@router.post("/workflows/{workflow_id}/pause", response_model=Dict[str, Any])
async def pause_workflow(workflow_id: str):
    """Pause a running workflow."""
    
    try:
        success = await agent_manager.workflow_engine.pause_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {
            "workflow_id": workflow_id,
            "status": "paused",
            "paused_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Workflow pause failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to pause workflow")

@router.post("/workflows/{workflow_id}/resume", response_model=Dict[str, Any])
async def resume_workflow(workflow_id: str):
    """Resume a paused workflow."""
    
    try:
        result = await agent_manager.workflow_engine.resume_workflow(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "status": "resumed",
            "execution_results": result,
            "resumed_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow resume failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to resume workflow")

@router.delete("/workflows/{workflow_id}", response_model=Dict[str, Any])
async def delete_workflow(workflow_id: str):
    """Delete a workflow."""
    
    try:
        success = agent_manager.workflow_engine.delete_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {
            "workflow_id": workflow_id,
            "status": "deleted",
            "deleted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Workflow deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete workflow")

# Document Generation Endpoints

@router.post("/documents/generate", response_model=Dict[str, Any])
async def generate_document(
    request: DocumentRequestModel,
    created_by: str = Query(..., description="User creating the document")
):
    """Generate a business document."""
    
    try:
        result = await agent_manager.generate_business_document(
            document_type=request.document_type,
            title=request.title,
            description=request.description,
            business_area=request.business_area,
            created_by=created_by,
            variables=request.variables,
            format=request.format
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Document generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate document")

@router.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents(
    business_area: Optional[str] = Query(None, description="Filter by business area"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    created_by: Optional[str] = Query(None, description="Filter by creator")
):
    """List generated documents with optional filters."""
    
    documents = agent_manager.document_generator.list_documents(
        business_area=business_area,
        document_type=document_type,
        created_by=created_by
    )
    
    return [
        {
            "id": doc.id,
            "request_id": doc.request_id,
            "title": doc.title,
            "format": doc.format.value,
            "file_path": doc.file_path,
            "size_bytes": doc.size_bytes,
            "created_at": doc.created_at.isoformat(),
            "metadata": doc.metadata
        }
        for doc in documents
    ]

@router.get("/documents/{document_id}", response_model=Dict[str, Any])
async def get_document(document_id: str):
    """Get specific document details."""
    
    document = agent_manager.document_generator.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    return {
        "id": document.id,
        "request_id": document.request_id,
        "title": document.title,
        "content": document.content,
        "format": document.format.value,
        "file_path": document.file_path,
        "size_bytes": document.size_bytes,
        "created_at": document.created_at.isoformat(),
        "metadata": document.metadata
    }

@router.get("/documents/{document_id}/download")
async def download_document(document_id: str):
    """Download a generated document."""
    
    document = agent_manager.document_generator.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    if not document.file_path:
        raise HTTPException(status_code=404, detail="Document file not found")
    
    return FileResponse(
        path=document.file_path,
        filename=f"{document.title}.{document.format.value}",
        media_type="application/octet-stream"
    )

# Business Area and Template Endpoints

@router.get("/business-areas", response_model=List[Dict[str, Any]])
async def get_business_areas():
    """Get all available business areas."""
    
    business_areas = agent_manager.get_business_areas()
    
    return [
        {
            "value": area.value,
            "name": area.value.replace("_", " ").title(),
            "agents_count": len(agent_manager.get_agents_by_business_area(area))
        }
        for area in business_areas
    ]

@router.get("/workflow-templates", response_model=Dict[str, List[Dict[str, Any]]])
async def get_workflow_templates():
    """Get predefined workflow templates for each business area."""
    
    return agent_manager.get_workflow_templates()

@router.get("/agents/by-business-area/{business_area}", response_model=List[Dict[str, Any]])
async def get_agents_by_business_area(business_area: BusinessArea):
    """Get all agents for a specific business area."""
    
    agents = agent_manager.get_agents_by_business_area(business_area)
    
    return [
        {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "capabilities_count": len(agent.capabilities),
            "is_active": agent.is_active
        }
        for agent in agents
    ]

# Health Check Endpoint

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint for the business agents system."""
    
    try:
        # Check system components
        agents_count = len(agent_manager.list_agents())
        workflows_count = len(agent_manager.list_workflows())
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "agent_manager": "healthy",
                "workflow_engine": "healthy",
                "document_generator": "healthy"
            },
            "metrics": {
                "total_agents": agents_count,
                "total_workflows": workflows_count
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Export/Import Endpoints

@router.get("/workflows/{workflow_id}/export", response_model=Dict[str, Any])
async def export_workflow(workflow_id: str):
    """Export workflow as JSON."""
    
    try:
        export_data = agent_manager.workflow_engine.export_workflow(workflow_id)
        return export_data
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow export failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export workflow")

@router.post("/workflows/import", response_model=Dict[str, Any])
async def import_workflow(workflow_data: Dict[str, Any]):
    """Import workflow from JSON."""
    
    try:
        workflow = agent_manager.workflow_engine.import_workflow(workflow_data)
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "business_area": workflow.business_area,
            "status": workflow.status.value,
            "imported_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Workflow import failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to import workflow")

# NLP Integration Endpoints

class BusinessDocumentAnalysisRequest(BaseModel):
    """Request model for business document analysis."""
    text: str = Field(..., description="Document text to analyze", min_length=1, max_length=50000)
    document_type: DocumentType = Field(..., description="Type of business document")
    business_area: BusinessArea = Field(..., description="Business area context")
    language: Optional[str] = Field(default="en", description="Document language")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_entities: bool = Field(default=True, description="Include entity extraction")
    include_keywords: bool = Field(default=True, description="Include keyword extraction")
    include_topics: bool = Field(default=False, description="Include topic modeling")
    include_readability: bool = Field(default=True, description="Include readability analysis")

class BusinessDocumentAnalysisResponse(BaseModel):
    """Response model for business document analysis."""
    document_type: str
    business_area: str
    analysis: Dict[str, Any]
    recommendations: List[str]
    processing_time: float
    timestamp: datetime

@router.post("/analyze-document", response_model=BusinessDocumentAnalysisResponse)
async def analyze_business_document(request: BusinessDocumentAnalysisRequest):
    """Analyze business documents with NLP and provide business insights."""
    try:
        start_time = datetime.now()
        
        # Perform advanced NLP analysis
        analysis_result = await advanced_nlp_system.analyze_text_advanced(
            text=request.text,
            language=request.language or "en"
        )
        
        # Generate business recommendations based on analysis
        recommendations = await _generate_business_recommendations(
            analysis_result, request.document_type, request.business_area
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BusinessDocumentAnalysisResponse(
            document_type=request.document_type.value,
            business_area=request.business_area.value,
            analysis=analysis_result,
            recommendations=recommendations,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Business document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Business document analysis failed: {e}")

class ContentOptimizationRequest(BaseModel):
    """Request model for content optimization."""
    text: str = Field(..., description="Text to optimize", min_length=1, max_length=10000)
    target_audience: str = Field(..., description="Target audience description")
    business_goal: str = Field(..., description="Business goal for the content")
    language: Optional[str] = Field(default="en", description="Content language")

class ContentOptimizationResponse(BaseModel):
    """Response model for content optimization."""
    original_text: str
    optimized_text: str
    improvements: List[str]
    readability_before: float
    readability_after: float
    sentiment_analysis: Dict[str, Any]
    keyword_suggestions: List[str]
    processing_time: float
    timestamp: datetime

@router.post("/optimize-content", response_model=ContentOptimizationResponse)
async def optimize_business_content(request: ContentOptimizationRequest):
    """Optimize business content for better engagement and readability."""
    try:
        start_time = datetime.now()
        
        # Analyze original content
        original_analysis = await advanced_nlp_system.analyze_text_advanced(
            text=request.text,
            language=request.language or "en"
        )
        
        # Generate optimized content
        optimized_content = await _optimize_content_for_business(
            request.text, request.target_audience, request.business_goal
        )
        
        # Analyze optimized content
        optimized_analysis = await advanced_nlp_system.analyze_text_advanced(
            text=optimized_content,
            language=request.language or "en"
        )
        
        # Generate improvement suggestions
        improvements = await _generate_improvement_suggestions(
            original_analysis, optimized_analysis
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ContentOptimizationResponse(
            original_text=request.text,
            optimized_text=optimized_content,
            improvements=improvements,
            readability_before=original_analysis.get('readability', {}).get('average_score', 0),
            readability_after=optimized_analysis.get('readability', {}).get('average_score', 0),
            sentiment_analysis=optimized_analysis.get('sentiment', {}),
            keyword_suggestions=optimized_analysis.get('keywords', []),
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Content optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content optimization failed: {e}")

# Helper functions for business analysis

async def _generate_business_recommendations(analysis: Dict[str, Any], document_type: DocumentType, business_area: BusinessArea) -> List[str]:
    """Generate business recommendations based on NLP analysis."""
    recommendations = []
    
    try:
        # Sentiment-based recommendations
        sentiment = analysis.get('sentiment', {})
        if sentiment.get('ensemble', {}).get('sentiment') == 'negative':
            recommendations.append("Consider revising content to improve sentiment and tone")
        
        # Readability recommendations
        readability = analysis.get('readability', {})
        if readability.get('average_score', 0) < 50:
            recommendations.append("Simplify language to improve readability and accessibility")
        
        # Keyword recommendations
        keywords = analysis.get('keywords', [])
        if len(keywords) < 5:
            recommendations.append("Add more relevant keywords to improve SEO and discoverability")
        
        # Entity-based recommendations
        entities = analysis.get('entities', [])
        if not any(entity.get('label') == 'ORG' for entity in entities):
            recommendations.append("Consider mentioning relevant organizations or companies")
        
        # Business area specific recommendations
        if business_area == BusinessArea.MARKETING:
            recommendations.append("Optimize content for marketing campaigns and brand messaging")
        elif business_area == BusinessArea.SALES:
            recommendations.append("Focus on customer pain points and solution benefits")
        elif business_area == BusinessArea.FINANCE:
            recommendations.append("Include financial metrics and data-driven insights")
        
        return recommendations[:5]  # Return top 5 recommendations
        
    except Exception as e:
        logger.error(f"Failed to generate business recommendations: {e}")
        return ["Unable to generate recommendations at this time"]

async def _optimize_content_for_business(text: str, target_audience: str, business_goal: str) -> str:
    """Optimize content for business purposes."""
    try:
        # This is a simplified optimization - in practice, you'd use more sophisticated methods
        optimized_text = text
        
        # Basic optimizations
        if "very" in optimized_text.lower():
            optimized_text = optimized_text.replace(" very ", " ")
        
        # Add business-specific improvements
        if business_goal.lower() in ["increase sales", "generate leads"]:
            if not any(word in optimized_text.lower() for word in ["call", "contact", "buy", "purchase"]):
                optimized_text += " Contact us today to learn more."
        
        return optimized_text
        
    except Exception as e:
        logger.error(f"Content optimization failed: {e}")
        return text

async def _generate_improvement_suggestions(original: Dict[str, Any], optimized: Dict[str, Any]) -> List[str]:
    """Generate improvement suggestions based on analysis comparison."""
    suggestions = []
    
    try:
        # Readability improvements
        orig_readability = original.get('readability', {}).get('average_score', 0)
        opt_readability = optimized.get('readability', {}).get('average_score', 0)
        
        if opt_readability > orig_readability:
            suggestions.append(f"Improved readability score from {orig_readability:.1f} to {opt_readability:.1f}")
        
        # Keyword improvements
        orig_keywords = len(original.get('keywords', []))
        opt_keywords = len(optimized.get('keywords', []))
        
        if opt_keywords > orig_keywords:
            suggestions.append(f"Increased keyword density from {orig_keywords} to {opt_keywords} keywords")
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to generate improvement suggestions: {e}")
        return ["Analysis completed successfully"]















