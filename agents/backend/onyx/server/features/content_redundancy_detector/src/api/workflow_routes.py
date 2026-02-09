"""
Workflow Routes - Advanced workflow automation API
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field, validator
import json

from ..core.content_workflow_engine import (
    create_workflow,
    execute_workflow,
    get_execution_status,
    get_workflow_metrics,
    get_workflow_health,
    initialize_workflow_engine,
    shutdown_workflow_engine,
    TriggerType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workflows", tags=["Workflows"])


# Pydantic models for request/response validation
class WorkflowStepRequest(BaseModel):
    """Request model for workflow step"""
    step_id: str = Field(..., description="Unique identifier for the step")
    name: str = Field(..., description="Human-readable name for the step")
    step_type: str = Field(..., description="Type of step (content_analysis, content_optimization, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Step configuration")
    dependencies: List[str] = Field(default_factory=list, description="List of step IDs this step depends on")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retries")
    timeout: int = Field(default=300, ge=1, le=3600, description="Step timeout in seconds")
    
    @validator('step_type')
    def validate_step_type(cls, v):
        allowed_types = [
            "content_analysis", "content_optimization", "similarity_analysis",
            "notification", "data_transformation", "validation", "custom"
        ]
        if v not in allowed_types:
            raise ValueError(f'Step type must be one of: {allowed_types}')
        return v


class WorkflowTriggerRequest(BaseModel):
    """Request model for workflow trigger"""
    trigger_type: str = Field(..., description="Type of trigger")
    config: Dict[str, Any] = Field(default_factory=dict, description="Trigger configuration")
    
    @validator('trigger_type')
    def validate_trigger_type(cls, v):
        allowed_types = ["manual", "scheduled", "event", "webhook", "condition"]
        if v not in allowed_types:
            raise ValueError(f'Trigger type must be one of: {allowed_types}')
        return v


class CreateWorkflowRequest(BaseModel):
    """Request model for creating a workflow"""
    name: str = Field(..., min_length=1, max_length=100, description="Workflow name")
    description: str = Field(..., min_length=1, max_length=500, description="Workflow description")
    steps: List[WorkflowStepRequest] = Field(..., min_items=1, description="List of workflow steps")
    triggers: List[WorkflowTriggerRequest] = Field(default_factory=list, description="List of workflow triggers")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")
    
    @validator('steps')
    def validate_steps(cls, v):
        if not v:
            raise ValueError('At least one step is required')
        
        # Check for duplicate step IDs
        step_ids = [step.step_id for step in v]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError('Step IDs must be unique')
        
        # Validate dependencies
        for step in v:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(f'Dependency {dep} not found in steps')
        
        return v


class ExecuteWorkflowRequest(BaseModel):
    """Request model for executing a workflow"""
    workflow_id: str = Field(..., description="ID of the workflow to execute")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    trigger_type: str = Field(default="manual", description="Type of trigger")
    
    @validator('trigger_type')
    def validate_trigger_type(cls, v):
        allowed_types = ["manual", "scheduled", "event", "webhook", "condition"]
        if v not in allowed_types:
            raise ValueError(f'Trigger type must be one of: {allowed_types}')
        return v


# Response models
class WorkflowStepResponse(BaseModel):
    """Response model for workflow step"""
    step_id: str
    name: str
    step_type: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    retry_count: int
    result: Optional[Dict[str, Any]] = None


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution"""
    execution_id: str
    workflow_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    steps: List[WorkflowStepResponse]


class WorkflowDefinitionResponse(BaseModel):
    """Response model for workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str
    steps_count: int
    triggers_count: int
    is_active: bool
    created_at: str
    updated_at: str


# Dependency functions
async def get_current_user() -> Dict[str, str]:
    """Dependency to get current user (placeholder for auth)"""
    return {"user_id": "anonymous", "role": "user"}


async def validate_api_key(api_key: Optional[str] = Query(None)) -> bool:
    """Dependency to validate API key"""
    # Placeholder for API key validation
    return True


# Route handlers
@router.post("/create", response_model=Dict[str, str])
async def create_workflow_endpoint(
    request: CreateWorkflowRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Create a new workflow definition
    
    - **name**: Human-readable name for the workflow
    - **description**: Detailed description of the workflow
    - **steps**: List of workflow steps with their configurations
    - **triggers**: List of triggers that can start the workflow
    - **variables**: Workflow-level variables
    """
    
    try:
        # Convert request to workflow format
        steps_data = []
        for step in request.steps:
            steps_data.append({
                "step_id": step.step_id,
                "name": step.name,
                "step_type": step.step_type,
                "config": step.config,
                "dependencies": step.dependencies,
                "max_retries": step.max_retries,
                "timeout": step.timeout
            })
        
        triggers_data = []
        for trigger in request.triggers:
            triggers_data.append({
                "trigger_type": trigger.trigger_type,
                "config": trigger.config
            })
        
        # Create workflow
        workflow_id = await create_workflow(
            name=request.name,
            description=request.description,
            steps=steps_data,
            triggers=triggers_data,
            variables=request.variables
        )
        
        logger.info(f"Workflow created: {request.name} ({workflow_id})")
        
        return {
            "workflow_id": workflow_id,
            "name": request.name,
            "status": "created",
            "message": "Workflow created successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Validation error in workflow creation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during workflow creation")


@router.post("/execute", response_model=Dict[str, str])
async def execute_workflow_endpoint(
    request: ExecuteWorkflowRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Execute a workflow
    
    - **workflow_id**: ID of the workflow to execute
    - **context**: Execution context with input data
    - **trigger_type**: Type of trigger that initiated the execution
    """
    
    try:
        # Convert trigger type
        trigger_type = TriggerType(request.trigger_type)
        
        # Execute workflow
        execution_id = await execute_workflow(
            workflow_id=request.workflow_id,
            context=request.context,
            trigger_type=trigger_type
        )
        
        logger.info(f"Workflow execution started: {execution_id}")
        
        return {
            "execution_id": execution_id,
            "workflow_id": request.workflow_id,
            "status": "queued",
            "message": "Workflow execution queued successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Validation error in workflow execution: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during workflow execution")


@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(
    execution_id: str,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> WorkflowExecutionResponse:
    """
    Get workflow execution status and details
    
    - **execution_id**: ID of the workflow execution
    """
    
    try:
        # Get execution status
        execution_data = await get_execution_status(execution_id)
        
        if not execution_data:
            raise HTTPException(status_code=404, detail="Workflow execution not found")
        
        # Convert steps to response format
        steps_response = [
            WorkflowStepResponse(
                step_id=step["step_id"],
                name=step["name"],
                step_type="unknown",  # This would come from workflow definition
                status=step["status"],
                started_at=step["started_at"],
                completed_at=step["completed_at"],
                error=step["error"],
                retry_count=step["retry_count"],
                result=None  # This would be included in a more detailed response
            )
            for step in execution_data["steps"]
        ]
        
        return WorkflowExecutionResponse(
            execution_id=execution_data["execution_id"],
            workflow_id=execution_data["workflow_id"],
            status=execution_data["status"],
            created_at=execution_data["created_at"],
            started_at=execution_data["started_at"],
            completed_at=execution_data["completed_at"],
            error=execution_data["error"],
            steps=steps_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow execution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during execution retrieval")


@router.get("/executions")
async def list_workflow_executions(
    workflow_id: Optional[str] = Query(None, description="Filter by workflow ID"),
    status: Optional[str] = Query(None, description="Filter by execution status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of executions to return"),
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List workflow executions with optional filtering
    
    - **workflow_id**: Filter executions by workflow ID
    - **status**: Filter executions by status
    - **limit**: Maximum number of executions to return
    """
    
    try:
        # This would typically query a database or storage system
        # For now, return a placeholder response
        executions = []
        
        return {
            "executions": executions,
            "total_count": len(executions),
            "filters": {
                "workflow_id": workflow_id,
                "status": status
            },
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing workflow executions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during execution listing")


@router.get("/templates")
async def get_workflow_templates() -> Dict[str, Any]:
    """Get predefined workflow templates"""
    
    templates = {
        "content_analysis_pipeline": {
            "name": "Content Analysis Pipeline",
            "description": "Complete content analysis workflow with AI insights",
            "steps": [
                {
                    "step_id": "analyze_content",
                    "name": "Analyze Content",
                    "step_type": "content_analysis",
                    "config": {
                        "analysis_type": "comprehensive"
                    },
                    "dependencies": []
                },
                {
                    "step_id": "generate_insights",
                    "name": "Generate Insights",
                    "step_type": "content_analysis",
                    "config": {
                        "analysis_type": "insights"
                    },
                    "dependencies": ["analyze_content"]
                },
                {
                    "step_id": "notify_completion",
                    "name": "Notify Completion",
                    "step_type": "notification",
                    "config": {
                        "message": "Content analysis completed",
                        "type": "success"
                    },
                    "dependencies": ["generate_insights"]
                }
            ],
            "triggers": [
                {
                    "trigger_type": "manual",
                    "config": {}
                }
            ]
        },
        "content_optimization_workflow": {
            "name": "Content Optimization Workflow",
            "description": "Optimize content for readability, SEO, and engagement",
            "steps": [
                {
                    "step_id": "analyze_content",
                    "name": "Analyze Content",
                    "step_type": "content_analysis",
                    "config": {
                        "analysis_type": "comprehensive"
                    },
                    "dependencies": []
                },
                {
                    "step_id": "optimize_content",
                    "name": "Optimize Content",
                    "step_type": "content_optimization",
                    "config": {
                        "optimization_goals": ["readability", "seo", "engagement"]
                    },
                    "dependencies": ["analyze_content"]
                },
                {
                    "step_id": "notify_optimization",
                    "name": "Notify Optimization",
                    "step_type": "notification",
                    "config": {
                        "message": "Content optimization completed",
                        "type": "info"
                    },
                    "dependencies": ["optimize_content"]
                }
            ],
            "triggers": [
                {
                    "trigger_type": "manual",
                    "config": {}
                }
            ]
        },
        "redundancy_detection_workflow": {
            "name": "Redundancy Detection Workflow",
            "description": "Detect and analyze content redundancy across multiple items",
            "steps": [
                {
                    "step_id": "similarity_analysis",
                    "name": "Similarity Analysis",
                    "step_type": "similarity_analysis",
                    "config": {
                        "similarity_threshold": 0.7
                    },
                    "dependencies": []
                },
                {
                    "step_id": "generate_report",
                    "name": "Generate Report",
                    "step_type": "notification",
                    "config": {
                        "message": "Redundancy analysis completed",
                        "type": "report"
                    },
                    "dependencies": ["similarity_analysis"]
                }
            ],
            "triggers": [
                {
                    "trigger_type": "manual",
                    "config": {}
                }
            ]
        }
    }
    
    return {
        "templates": templates,
        "total_templates": len(templates),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/templates/{template_name}/create")
async def create_workflow_from_template(
    template_name: str,
    custom_name: Optional[str] = Query(None, description="Custom name for the workflow"),
    custom_description: Optional[str] = Query(None, description="Custom description for the workflow"),
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Create a workflow from a predefined template
    
    - **template_name**: Name of the template to use
    - **custom_name**: Optional custom name for the workflow
    - **custom_description**: Optional custom description for the workflow
    """
    
    try:
        # Get templates
        templates_response = await get_workflow_templates()
        templates = templates_response["templates"]
        
        if template_name not in templates:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        template = templates[template_name]
        
        # Use custom name/description if provided
        name = custom_name or template["name"]
        description = custom_description or template["description"]
        
        # Create workflow from template
        workflow_id = await create_workflow(
            name=name,
            description=description,
            steps=template["steps"],
            triggers=template["triggers"]
        )
        
        logger.info(f"Workflow created from template: {template_name} -> {workflow_id}")
        
        return {
            "workflow_id": workflow_id,
            "template_name": template_name,
            "name": name,
            "status": "created",
            "message": f"Workflow created successfully from template '{template_name}'",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow from template: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during template-based workflow creation")


@router.get("/metrics")
async def get_workflow_metrics_endpoint(
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get workflow engine metrics"""
    
    try:
        metrics = await get_workflow_metrics()
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during metrics retrieval")


@router.get("/health")
async def workflow_health_check() -> Dict[str, Any]:
    """Health check endpoint for workflow service"""
    
    try:
        health_status = await get_workflow_health()
        
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
            "service": "workflow-engine",
            "timestamp": datetime.now().isoformat(),
            "workflow_engine": health_status
        }
        
    except Exception as e:
        logger.error(f"Workflow health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "workflow-engine",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/step-types")
async def get_available_step_types() -> Dict[str, Any]:
    """Get available workflow step types and their descriptions"""
    
    step_types = {
        "content_analysis": {
            "description": "Analyze content using AI for sentiment, topics, and insights",
            "required_config": ["analysis_type"],
            "optional_config": ["confidence_threshold", "language"],
            "outputs": ["sentiment_score", "topic_classification", "key_phrases", "insights"]
        },
        "content_optimization": {
            "description": "Optimize content for readability, SEO, and engagement",
            "required_config": ["optimization_goals"],
            "optional_config": ["target_audience", "brand_voice"],
            "outputs": ["optimized_content", "suggestions", "improvement_score"]
        },
        "similarity_analysis": {
            "description": "Analyze similarity and redundancy between content items",
            "required_config": ["similarity_threshold"],
            "optional_config": ["analysis_type", "clustering_method"],
            "outputs": ["similarity_scores", "duplicate_groups", "recommendations"]
        },
        "notification": {
            "description": "Send notifications about workflow progress or completion",
            "required_config": ["message"],
            "optional_config": ["type", "recipients", "channels"],
            "outputs": ["notification_sent", "delivery_status"]
        },
        "data_transformation": {
            "description": "Transform data between different formats or structures",
            "required_config": ["transformation_type"],
            "optional_config": ["input_format", "output_format", "mapping"],
            "outputs": ["transformed_data", "transformation_log"]
        },
        "validation": {
            "description": "Validate data or content against specific rules or criteria",
            "required_config": ["validation_rules"],
            "optional_config": ["strict_mode", "error_handling"],
            "outputs": ["validation_result", "errors", "warnings"]
        },
        "custom": {
            "description": "Custom step type for specialized processing",
            "required_config": ["handler_class"],
            "optional_config": ["custom_config"],
            "outputs": ["custom_output"]
        }
    }
    
    return {
        "step_types": step_types,
        "total_types": len(step_types),
        "timestamp": datetime.now().isoformat()
    }


# Startup and shutdown handlers
@router.on_event("startup")
async def startup_workflow_service():
    """Initialize workflow service on startup"""
    try:
        await initialize_workflow_engine()
        logger.info("Workflow service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize workflow service: {e}")


@router.on_event("shutdown")
async def shutdown_workflow_service():
    """Shutdown workflow service on shutdown"""
    try:
        await shutdown_workflow_engine()
        logger.info("Workflow service shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown workflow service: {e}")




