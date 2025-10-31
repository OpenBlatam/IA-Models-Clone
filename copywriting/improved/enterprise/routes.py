"""
Enterprise API Routes
====================

Enterprise-level endpoints for workflow management, brand consistency, and compliance.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, Query, Body
from fastapi.responses import JSONResponse

from ..schemas import CopywritingRequest, CopywritingVariant, ErrorResponse
from ..services import get_copywriting_service, CopywritingService
from ..exceptions import WorkflowError, BrandViolationError, ComplianceError
from .workflow_engine import (
    workflow_engine,
    WorkflowStatus,
    WorkflowStepType,
    ApprovalLevel,
    WorkflowTemplate,
    WorkflowInstance
)
from .brand_manager import (
    brand_manager,
    BrandTone,
    BrandStyle,
    BrandVoice,
    BrandGuidelines
)
from .compliance_engine import (
    compliance_engine,
    ComplianceType,
    ComplianceLevel,
    IndustryType,
    ComplianceRule
)

logger = logging.getLogger(__name__)

# Create enterprise router
enterprise_router = APIRouter(
    prefix="/api/v2/copywriting/enterprise",
    tags=["enterprise"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)


# Workflow Management Routes
@enterprise_router.get(
    "/workflows/templates",
    summary="Get Workflow Templates",
    description="Get available workflow templates"
)
async def get_workflow_templates():
    """Get all available workflow templates"""
    try:
        templates = []
        for template in workflow_engine.templates.values():
            templates.append({
                "id": str(template.id),
                "name": template.name,
                "description": template.description,
                "steps_count": len(template.steps),
                "is_active": template.is_active,
                "created_by": template.created_by,
                "created_at": template.created_at.isoformat()
            })
        
        return {
            "templates": templates,
            "total_templates": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Failed to get workflow templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workflow templates"
        )


@enterprise_router.post(
    "/workflows/create",
    summary="Create Workflow Instance",
    description="Create a new workflow instance from template"
)
async def create_workflow_instance(
    template_id: UUID = Body(..., description="Template ID"),
    name: str = Body(..., description="Workflow name"),
    created_by: str = Body(..., description="Creator user ID"),
    priority: int = Body(default=1, ge=1, le=5, description="Priority level (1-5)"),
    due_date: Optional[str] = Body(default=None, description="Due date (ISO format)"),
    metadata: Optional[Dict[str, Any]] = Body(default=None, description="Additional metadata")
):
    """Create a new workflow instance"""
    try:
        from datetime import datetime
        
        due_date_obj = None
        if due_date:
            due_date_obj = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
        
        instance = await workflow_engine.create_workflow_instance(
            template_id=template_id,
            name=name,
            created_by=created_by,
            priority=priority,
            due_date=due_date_obj,
            metadata=metadata
        )
        
        return {
            "workflow_instance": {
                "id": str(instance.id),
                "template_id": str(instance.template_id),
                "name": instance.name,
                "status": instance.status.value,
                "created_by": instance.created_by,
                "priority": instance.priority,
                "due_date": instance.due_date.isoformat() if instance.due_date else None,
                "created_at": instance.created_at.isoformat()
            },
            "message": "Workflow instance created successfully"
        }
        
    except WorkflowError as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create workflow instance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow instance"
        )


@enterprise_router.post(
    "/workflows/{instance_id}/start",
    summary="Start Workflow",
    description="Start a workflow instance"
)
async def start_workflow(
    instance_id: UUID,
    started_by: str = Body(..., description="User ID who started the workflow")
):
    """Start a workflow instance"""
    try:
        instance = await workflow_engine.start_workflow(instance_id, started_by)
        
        return {
            "workflow_instance": {
                "id": str(instance.id),
                "status": instance.status.value,
                "current_step_id": str(instance.current_step_id) if instance.current_step_id else None,
                "assigned_to": instance.assigned_to,
                "updated_at": instance.updated_at.isoformat()
            },
            "message": "Workflow started successfully"
        }
        
    except WorkflowError as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start workflow"
        )


@enterprise_router.post(
    "/workflows/{instance_id}/execute",
    summary="Execute Workflow Step",
    description="Execute a workflow step"
)
async def execute_workflow_step(
    instance_id: UUID,
    step_id: UUID = Body(..., description="Step ID to execute"),
    executed_by: str = Body(..., description="User ID executing the step"),
    status: WorkflowStatus = Body(..., description="Step execution status"),
    comments: Optional[str] = Body(default=None, description="Execution comments"),
    data: Optional[Dict[str, Any]] = Body(default=None, description="Step execution data")
):
    """Execute a workflow step"""
    try:
        instance = await workflow_engine.execute_step(
            instance_id=instance_id,
            step_id=step_id,
            executed_by=executed_by,
            status=status,
            comments=comments,
            data=data
        )
        
        return {
            "workflow_instance": {
                "id": str(instance.id),
                "status": instance.status.value,
                "current_step_id": str(instance.current_step_id) if instance.current_step_id else None,
                "assigned_to": instance.assigned_to,
                "updated_at": instance.updated_at.isoformat(),
                "completed_at": instance.completed_at.isoformat() if instance.completed_at else None
            },
            "message": "Workflow step executed successfully"
        }
        
    except WorkflowError as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to execute workflow step: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute workflow step"
        )


@enterprise_router.get(
    "/workflows/{instance_id}/status",
    summary="Get Workflow Status",
    description="Get comprehensive workflow status"
)
async def get_workflow_status(instance_id: UUID):
    """Get workflow status"""
    try:
        status_info = await workflow_engine.get_workflow_status(instance_id)
        return status_info
        
    except WorkflowError as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow status"
        )


@enterprise_router.get(
    "/workflows/user/{user_id}",
    summary="Get User Workflows",
    description="Get workflows assigned to a user"
)
async def get_user_workflows(
    user_id: str,
    status_filter: Optional[WorkflowStatus] = Query(default=None, description="Filter by status")
):
    """Get workflows assigned to a user"""
    try:
        workflows = await workflow_engine.get_user_workflows(user_id, status_filter)
        return {
            "user_id": user_id,
            "workflows": workflows,
            "total_workflows": len(workflows)
        }
        
    except Exception as e:
        logger.error(f"Failed to get user workflows: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user workflows"
        )


@enterprise_router.get(
    "/workflows/analytics",
    summary="Get Workflow Analytics",
    description="Get workflow analytics and metrics"
)
async def get_workflow_analytics():
    """Get workflow analytics"""
    try:
        analytics = await workflow_engine.get_workflow_analytics()
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get workflow analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow analytics"
        )


# Brand Management Routes
@enterprise_router.get(
    "/brand/guidelines",
    summary="Get Brand Guidelines",
    description="Get all brand guidelines"
)
async def get_brand_guidelines():
    """Get all brand guidelines"""
    try:
        guidelines = []
        for guideline in brand_manager.brand_guidelines.values():
            guidelines.append({
                "id": str(guideline.id),
                "brand_name": guideline.brand_name,
                "primary_tone": guideline.primary_tone.value,
                "secondary_tones": [tone.value for tone in guideline.secondary_tones],
                "style": guideline.style.value,
                "voice_characteristics": [voice.value for voice in guideline.voice_characteristics],
                "is_active": guideline.is_active,
                "created_at": guideline.created_at.isoformat()
            })
        
        return {
            "guidelines": guidelines,
            "total_guidelines": len(guidelines)
        }
        
    except Exception as e:
        logger.error(f"Failed to get brand guidelines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve brand guidelines"
        )


@enterprise_router.post(
    "/brand/validate",
    summary="Validate Content Against Brand Guidelines",
    description="Validate content against brand guidelines"
)
async def validate_brand_compliance(
    content: str = Body(..., description="Content to validate"),
    brand_guidelines_id: UUID = Body(..., description="Brand guidelines ID"),
    content_id: Optional[UUID] = Body(default=None, description="Content ID")
):
    """Validate content against brand guidelines"""
    try:
        report = await brand_manager.validate_content(
            content=content,
            brand_guidelines_id=brand_guidelines_id,
            content_id=content_id
        )
        
        return {
            "compliance_report": {
                "content_id": str(report.content_id),
                "brand_guidelines_id": str(report.brand_guidelines_id),
                "overall_score": report.overall_score,
                "compliance_percentage": report.compliance_percentage,
                "violations": [
                    {
                        "id": str(violation.id),
                        "violation_type": violation.violation_type,
                        "severity": violation.severity,
                        "description": violation.description,
                        "suggested_fix": violation.suggested_fix,
                        "line_number": violation.line_number,
                        "detected_at": violation.detected_at.isoformat()
                    }
                    for violation in report.violations
                ],
                "recommendations": report.recommendations,
                "generated_at": report.generated_at.isoformat()
            }
        }
        
    except BrandViolationError as e:
        logger.error(f"Brand validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to validate brand compliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate brand compliance"
        )


@enterprise_router.post(
    "/brand/guidelines/create",
    summary="Create Brand Guidelines",
    description="Create new brand guidelines"
)
async def create_brand_guidelines(
    brand_name: str = Body(..., description="Brand name"),
    primary_tone: BrandTone = Body(..., description="Primary brand tone"),
    style: BrandStyle = Body(..., description="Brand style"),
    voice_characteristics: List[BrandVoice] = Body(..., description="Voice characteristics"),
    secondary_tones: Optional[List[BrandTone]] = Body(default=None, description="Secondary tones"),
    preferred_words: Optional[List[str]] = Body(default=None, description="Preferred words"),
    forbidden_words: Optional[List[str]] = Body(default=None, description="Forbidden words"),
    industry_terms: Optional[List[str]] = Body(default=None, description="Industry terms"),
    brand_terms: Optional[List[str]] = Body(default=None, description="Brand terms")
):
    """Create new brand guidelines"""
    try:
        guidelines = await brand_manager.create_brand_guidelines(
            brand_name=brand_name,
            primary_tone=primary_tone,
            style=style,
            voice_characteristics=voice_characteristics,
            secondary_tones=secondary_tones or [],
            preferred_words=preferred_words or [],
            forbidden_words=forbidden_words or [],
            industry_terms=industry_terms or [],
            brand_terms=brand_terms or []
        )
        
        return {
            "brand_guidelines": {
                "id": str(guidelines.id),
                "brand_name": guidelines.brand_name,
                "primary_tone": guidelines.primary_tone.value,
                "style": guidelines.style.value,
                "voice_characteristics": [voice.value for voice in guidelines.voice_characteristics],
                "created_at": guidelines.created_at.isoformat()
            },
            "message": "Brand guidelines created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create brand guidelines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create brand guidelines"
        )


@enterprise_router.get(
    "/brand/analytics",
    summary="Get Brand Analytics",
    description="Get brand compliance analytics"
)
async def get_brand_analytics():
    """Get brand analytics"""
    try:
        analytics = await brand_manager.get_brand_analytics()
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get brand analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get brand analytics"
        )


# Compliance Routes
@enterprise_router.post(
    "/compliance/check",
    summary="Check Content Compliance",
    description="Check content compliance against regulations"
)
async def check_compliance(
    content: str = Body(..., description="Content to check"),
    industry: IndustryType = Body(..., description="Industry type"),
    content_id: Optional[UUID] = Body(default=None, description="Content ID"),
    additional_rules: Optional[List[UUID]] = Body(default=None, description="Additional rule IDs")
):
    """Check content compliance"""
    try:
        report = await compliance_engine.check_compliance(
            content=content,
            industry=industry,
            content_id=content_id,
            additional_rules=additional_rules
        )
        
        return {
            "compliance_report": {
                "content_id": str(report.content_id),
                "industry": report.industry.value,
                "overall_compliance_score": report.overall_compliance_score,
                "violations": [
                    {
                        "id": str(violation.id),
                        "rule_id": str(violation.rule_id),
                        "violation_type": violation.violation_type.value,
                        "level": violation.level.value,
                        "description": violation.description,
                        "suggested_fix": violation.suggested_fix,
                        "line_number": violation.line_number,
                        "detected_at": violation.detected_at.isoformat()
                    }
                    for violation in report.violations
                ],
                "recommendations": report.recommendations,
                "compliance_summary": report.compliance_summary,
                "generated_at": report.generated_at.isoformat()
            }
        }
        
    except ComplianceError as e:
        logger.error(f"Compliance error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to check compliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check compliance"
        )


@enterprise_router.get(
    "/compliance/rules",
    summary="Get Compliance Rules",
    description="Get compliance rules for an industry"
)
async def get_compliance_rules(
    industry: Optional[IndustryType] = Query(default=None, description="Filter by industry")
):
    """Get compliance rules"""
    try:
        if industry:
            rules = await compliance_engine.get_industry_compliance_requirements(industry)
        else:
            rules = list(compliance_engine.compliance_rules.values())
        
        rules_data = []
        for rule in rules:
            rules_data.append({
                "id": str(rule.id),
                "name": rule.name,
                "description": rule.description,
                "compliance_type": rule.compliance_type.value,
                "industry": rule.industry.value if rule.industry else None,
                "level": rule.level.value,
                "is_active": rule.is_active,
                "created_at": rule.created_at.isoformat()
            })
        
        return {
            "rules": rules_data,
            "total_rules": len(rules_data),
            "industry": industry.value if industry else "all"
        }
        
    except Exception as e:
        logger.error(f"Failed to get compliance rules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get compliance rules"
        )


@enterprise_router.post(
    "/compliance/rules/create",
    summary="Create Compliance Rule",
    description="Create a custom compliance rule"
)
async def create_compliance_rule(
    name: str = Body(..., description="Rule name"),
    description: str = Body(..., description="Rule description"),
    compliance_type: ComplianceType = Body(..., description="Compliance type"),
    level: ComplianceLevel = Body(..., description="Compliance level"),
    industry: Optional[IndustryType] = Body(default=None, description="Industry"),
    keywords: Optional[List[str]] = Body(default=None, description="Keywords to check"),
    required_elements: Optional[List[str]] = Body(default=None, description="Required elements"),
    forbidden_elements: Optional[List[str]] = Body(default=None, description="Forbidden elements")
):
    """Create a custom compliance rule"""
    try:
        rule = await compliance_engine.create_custom_rule(
            name=name,
            description=description,
            compliance_type=compliance_type,
            level=level,
            industry=industry,
            keywords=keywords or [],
            required_elements=required_elements or [],
            forbidden_elements=forbidden_elements or []
        )
        
        return {
            "compliance_rule": {
                "id": str(rule.id),
                "name": rule.name,
                "description": rule.description,
                "compliance_type": rule.compliance_type.value,
                "industry": rule.industry.value if rule.industry else None,
                "level": rule.level.value,
                "is_active": rule.is_active,
                "created_at": rule.created_at.isoformat()
            },
            "message": "Compliance rule created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create compliance rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create compliance rule"
        )


@enterprise_router.get(
    "/compliance/analytics",
    summary="Get Compliance Analytics",
    description="Get compliance analytics and metrics"
)
async def get_compliance_analytics():
    """Get compliance analytics"""
    try:
        analytics = await compliance_engine.get_compliance_analytics()
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get compliance analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get compliance analytics"
        )


# Enterprise Dashboard
@enterprise_router.get(
    "/dashboard",
    summary="Get Enterprise Dashboard",
    description="Get comprehensive enterprise dashboard data"
)
async def get_enterprise_dashboard():
    """Get enterprise dashboard data"""
    try:
        # Get analytics from all enterprise modules
        workflow_analytics = await workflow_engine.get_workflow_analytics()
        brand_analytics = await brand_manager.get_brand_analytics()
        compliance_analytics = await compliance_engine.get_compliance_analytics()
        
        return {
            "dashboard": {
                "workflow_analytics": workflow_analytics,
                "brand_analytics": brand_analytics,
                "compliance_analytics": compliance_analytics,
                "generated_at": "2024-01-01T00:00:00Z"  # Would be datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get enterprise dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get enterprise dashboard"
        )






























