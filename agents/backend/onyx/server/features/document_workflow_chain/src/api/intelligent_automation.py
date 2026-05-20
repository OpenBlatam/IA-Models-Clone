"""
Intelligent Automation API - Advanced Implementation
==================================================

Advanced intelligent automation API with AI-powered decision making and automation.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime

from ..services import intelligent_automation_service, AutomationTrigger, AutomationAction, AutomationStatus

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class AutomationCreateRequest(BaseModel):
    """Automation create request model"""
    name: str
    description: str
    trigger: Dict[str, Any]
    actions: List[Dict[str, Any]]
    ai_decision_making: bool = False
    learning_enabled: bool = False
    template_name: Optional[str] = None


class AutomationResponse(BaseModel):
    """Automation response model"""
    automation_id: str
    name: str
    status: str
    created_at: str
    message: str


class AutomationInfoResponse(BaseModel):
    """Automation info response model"""
    id: str
    name: str
    description: str
    trigger: Dict[str, Any]
    actions: List[Dict[str, Any]]
    ai_decision_making: bool
    learning_enabled: bool
    status: str
    created_at: str
    last_triggered: Optional[str]
    last_executed: Optional[str]
    execution_count: int
    success_count: int
    failure_count: int
    learning_data: Dict[str, Any]
    metadata: Dict[str, Any]


class AutomationTriggerRequest(BaseModel):
    """Automation trigger request model"""
    trigger_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class AutomationTemplateResponse(BaseModel):
    """Automation template response model"""
    name: str
    description: str
    trigger: Dict[str, Any]
    actions: List[Dict[str, Any]]
    ai_decision_making: bool
    learning_enabled: bool


class AutomationStatsResponse(BaseModel):
    """Automation statistics response model"""
    total_automations: int
    active_automations: int
    triggered_automations: int
    completed_automations: int
    failed_automations: int
    automations_by_trigger: Dict[str, int]
    automations_by_action: Dict[str, int]
    available_templates: int


# Automation creation endpoints
@router.post("/automations", response_model=AutomationResponse)
async def create_automation(request: AutomationCreateRequest):
    """Create intelligent automation"""
    try:
        automation_id = await intelligent_automation_service.create_automation(
            name=request.name,
            description=request.description,
            trigger=request.trigger,
            actions=request.actions,
            ai_decision_making=request.ai_decision_making,
            learning_enabled=request.learning_enabled,
            template_name=request.template_name
        )
        
        return AutomationResponse(
            automation_id=automation_id,
            name=request.name,
            status="active",
            created_at=datetime.utcnow().isoformat(),
            message="Automation created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create automation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create automation: {str(e)}"
        )


@router.post("/automations/{automation_id}/trigger")
async def trigger_automation(automation_id: str, request: AutomationTriggerRequest):
    """Trigger automation execution"""
    try:
        result = await intelligent_automation_service.trigger_automation(
            automation_id=automation_id,
            trigger_data=request.trigger_data,
            context=request.context
        )
        
        return {
            "automation_id": automation_id,
            "triggered": result["triggered"],
            "execution_result": result.get("execution_result"),
            "reason": result.get("reason"),
            "timestamp": result["timestamp"]
        }
    
    except Exception as e:
        logger.error(f"Failed to trigger automation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger automation: {str(e)}"
        )


# Automation management endpoints
@router.get("/automations/{automation_id}", response_model=AutomationInfoResponse)
async def get_automation(automation_id: str):
    """Get automation information"""
    try:
        automation = await intelligent_automation_service.get_automation(automation_id)
        if not automation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Automation not found"
            )
        
        return AutomationInfoResponse(**automation)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get automation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation: {str(e)}"
        )


@router.get("/automations", response_model=List[Dict[str, Any]])
async def list_automations(
    status: Optional[str] = None,
    trigger_type: Optional[str] = None,
    limit: int = 100
):
    """List automations with filtering"""
    try:
        # Convert string parameters to enums
        status_enum = AutomationStatus(status) if status else None
        trigger_enum = AutomationTrigger(trigger_type) if trigger_type else None
        
        automations = await intelligent_automation_service.list_automations(
            status=status_enum,
            trigger_type=trigger_enum,
            limit=limit
        )
        
        return automations
    
    except Exception as e:
        logger.error(f"Failed to list automations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list automations: {str(e)}"
        )


# Automation templates endpoints
@router.get("/automations/templates", response_model=Dict[str, Any])
async def get_automation_templates():
    """Get available automation templates"""
    try:
        templates = await intelligent_automation_service.get_automation_templates()
        return templates
    
    except Exception as e:
        logger.error(f"Failed to get automation templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation templates: {str(e)}"
        )


# Automation statistics endpoint
@router.get("/automations/stats", response_model=AutomationStatsResponse)
async def get_automation_stats():
    """Get automation service statistics"""
    try:
        stats = await intelligent_automation_service.get_automation_stats()
        return AutomationStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get automation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation stats: {str(e)}"
        )


# Health check endpoint
@router.get("/automations/health")
async def automation_health():
    """Automation service health check"""
    try:
        stats = await intelligent_automation_service.get_automation_stats()
        
        return {
            "service": "intelligent_automation_service",
            "status": "healthy",
            "total_automations": stats["total_automations"],
            "active_automations": stats["active_automations"],
            "available_templates": stats["available_templates"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Automation service health check failed: {e}")
        return {
            "service": "intelligent_automation_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

