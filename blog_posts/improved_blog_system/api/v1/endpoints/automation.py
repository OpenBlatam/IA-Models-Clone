"""
Advanced Automation API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_automation_service import AdvancedAutomationService, AutomationType, TriggerType, ActionType, ExecutionStatus, ConditionOperator
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class CreateAutomationRuleRequest(BaseModel):
    """Request model for creating an automation rule."""
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    automation_type: str = Field(..., description="Automation type")
    triggers: List[Dict[str, Any]] = Field(..., description="Automation triggers")
    conditions: List[Dict[str, Any]] = Field(default=[], description="Automation conditions")
    actions: List[Dict[str, Any]] = Field(..., description="Automation actions")
    is_active: bool = Field(default=True, description="Is rule active")
    priority: int = Field(default=0, description="Rule priority")


class CreateAutomationWorkflowRequest(BaseModel):
    """Request model for creating an automation workflow."""
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    workflow_type: str = Field(..., description="Workflow type")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    variables: Optional[Dict[str, Any]] = Field(default=None, description="Workflow variables")
    is_active: bool = Field(default=True, description="Is workflow active")


class ExecuteAutomationRuleRequest(BaseModel):
    """Request model for executing an automation rule."""
    rule_id: str = Field(..., description="Rule ID")
    trigger_data: Dict[str, Any] = Field(..., description="Trigger data")


class ExecuteAutomationWorkflowRequest(BaseModel):
    """Request model for executing an automation workflow."""
    workflow_id: str = Field(..., description="Workflow ID")
    input_data: Dict[str, Any] = Field(..., description="Input data")


class ScheduleAutomationRequest(BaseModel):
    """Request model for scheduling automation."""
    rule_id: str = Field(..., description="Rule ID")
    schedule_config: Dict[str, Any] = Field(..., description="Schedule configuration")


class CreateWebhookHandlerRequest(BaseModel):
    """Request model for creating a webhook handler."""
    name: str = Field(..., description="Webhook name")
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to listen for")
    secret: Optional[str] = Field(default=None, description="Webhook secret")
    is_active: bool = Field(default=True, description="Is webhook active")


async def get_automation_service(session: DatabaseSessionDep) -> AdvancedAutomationService:
    """Get automation service instance."""
    return AdvancedAutomationService(session)


@router.post("/rules", response_model=Dict[str, Any])
async def create_automation_rule(
    request: CreateAutomationRuleRequest = Depends(),
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new automation rule."""
    try:
        # Convert automation type to enum
        try:
            automation_type_enum = AutomationType(request.automation_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid automation type: {request.automation_type}")
        
        result = await automation_service.create_automation_rule(
            name=request.name,
            description=request.description,
            automation_type=automation_type_enum,
            user_id=str(current_user.id),
            triggers=request.triggers,
            conditions=request.conditions,
            actions=request.actions,
            is_active=request.is_active,
            priority=request.priority
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Automation rule created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create automation rule"
        )


@router.post("/workflows", response_model=Dict[str, Any])
async def create_automation_workflow(
    request: CreateAutomationWorkflowRequest = Depends(),
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new automation workflow."""
    try:
        result = await automation_service.create_automation_workflow(
            name=request.name,
            description=request.description,
            workflow_type=request.workflow_type,
            user_id=str(current_user.id),
            steps=request.steps,
            variables=request.variables,
            is_active=request.is_active
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Automation workflow created successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create automation workflow"
        )


@router.post("/rules/execute", response_model=Dict[str, Any])
async def execute_automation_rule(
    request: ExecuteAutomationRuleRequest = Depends(),
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Execute an automation rule."""
    try:
        result = await automation_service.execute_automation_rule(
            rule_id=request.rule_id,
            trigger_data=request.trigger_data,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Automation rule executed successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute automation rule"
        )


@router.post("/workflows/execute", response_model=Dict[str, Any])
async def execute_automation_workflow(
    request: ExecuteAutomationWorkflowRequest = Depends(),
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Execute an automation workflow."""
    try:
        result = await automation_service.execute_automation_workflow(
            workflow_id=request.workflow_id,
            input_data=request.input_data,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Automation workflow executed successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute automation workflow"
        )


@router.post("/schedule", response_model=Dict[str, Any])
async def schedule_automation(
    request: ScheduleAutomationRequest = Depends(),
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Schedule an automation rule."""
    try:
        result = await automation_service.schedule_automation(
            rule_id=request.rule_id,
            schedule_config=request.schedule_config,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Automation scheduled successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule automation"
        )


@router.post("/webhooks", response_model=Dict[str, Any])
async def create_webhook_handler(
    request: CreateWebhookHandlerRequest = Depends(),
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a webhook handler."""
    try:
        result = await automation_service.create_webhook_handler(
            name=request.name,
            url=request.url,
            events=request.events,
            user_id=str(current_user.id),
            secret=request.secret,
            is_active=request.is_active
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Webhook handler created successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create webhook handler"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_automation_analytics(
    user_id: Optional[str] = Query(default=None, description="User ID"),
    time_period: str = Query(default="30_days", description="Time period"),
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get automation analytics."""
    try:
        result = await automation_service.get_automation_analytics(
            user_id=user_id,
            time_period=time_period
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Automation analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get automation analytics"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_automation_stats(
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get automation system statistics."""
    try:
        result = await automation_service.get_automation_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Automation statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get automation statistics"
        )


@router.get("/automation-types", response_model=Dict[str, Any])
async def get_automation_types():
    """Get available automation types."""
    automation_types = {
        "content": {
            "name": "Content Automation",
            "description": "Automate content creation, publishing, and management",
            "icon": "📝",
            "use_cases": ["Auto-publish content", "Content scheduling", "Content optimization"]
        },
        "social": {
            "name": "Social Automation",
            "description": "Automate social media posting and engagement",
            "icon": "📱",
            "use_cases": ["Social media posting", "Engagement tracking", "Social analytics"]
        },
        "ecommerce": {
            "name": "E-commerce Automation",
            "description": "Automate e-commerce operations and order processing",
            "icon": "🛒",
            "use_cases": ["Order processing", "Inventory management", "Customer notifications"]
        },
        "gamification": {
            "name": "Gamification Automation",
            "description": "Automate gamification features and user engagement",
            "icon": "🎮",
            "use_cases": ["Badge assignment", "Point updates", "Achievement tracking"]
        },
        "learning": {
            "name": "Learning Automation",
            "description": "Automate learning and education workflows",
            "icon": "🎓",
            "use_cases": ["Course enrollment", "Progress tracking", "Certificate generation"]
        },
        "ai": {
            "name": "AI Automation",
            "description": "Automate AI-powered tasks and workflows",
            "icon": "🤖",
            "use_cases": ["AI content generation", "AI analysis", "AI recommendations"]
        },
        "notification": {
            "name": "Notification Automation",
            "description": "Automate notification delivery and management",
            "icon": "🔔",
            "use_cases": ["Email campaigns", "Push notifications", "Alert management"]
        },
        "analytics": {
            "name": "Analytics Automation",
            "description": "Automate analytics and reporting workflows",
            "icon": "📊",
            "use_cases": ["Report generation", "Data analysis", "Performance tracking"]
        },
        "security": {
            "name": "Security Automation",
            "description": "Automate security monitoring and response",
            "icon": "🔒",
            "use_cases": ["Threat detection", "Access control", "Security audits"]
        },
        "system": {
            "name": "System Automation",
            "description": "Automate system operations and maintenance",
            "icon": "⚙️",
            "use_cases": ["Backup automation", "Log cleanup", "System monitoring"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "automation_types": automation_types,
            "total_types": len(automation_types)
        },
        "message": "Automation types retrieved successfully"
    }


@router.get("/trigger-types", response_model=Dict[str, Any])
async def get_trigger_types():
    """Get available trigger types."""
    trigger_types = {
        "time_based": {
            "name": "Time-based Trigger",
            "description": "Trigger based on time schedules (cron, intervals)",
            "icon": "⏰",
            "config_fields": ["cron", "timezone", "interval"]
        },
        "event_based": {
            "name": "Event-based Trigger",
            "description": "Trigger based on system or user events",
            "icon": "📡",
            "config_fields": ["event", "filters", "conditions"]
        },
        "condition_based": {
            "name": "Condition-based Trigger",
            "description": "Trigger based on data conditions",
            "icon": "🔍",
            "config_fields": ["field", "operator", "value"]
        },
        "webhook": {
            "name": "Webhook Trigger",
            "description": "Trigger based on incoming webhook calls",
            "icon": "🔗",
            "config_fields": ["url", "secret", "events"]
        },
        "api_call": {
            "name": "API Call Trigger",
            "description": "Trigger based on API calls",
            "icon": "🌐",
            "config_fields": ["endpoint", "method", "headers"]
        },
        "user_action": {
            "name": "User Action Trigger",
            "description": "Trigger based on user actions",
            "icon": "👤",
            "config_fields": ["action", "user_id", "context"]
        },
        "system_event": {
            "name": "System Event Trigger",
            "description": "Trigger based on system events",
            "icon": "⚙️",
            "config_fields": ["event", "severity", "source"]
        },
        "external_service": {
            "name": "External Service Trigger",
            "description": "Trigger based on external service events",
            "icon": "🔌",
            "config_fields": ["service", "event", "credentials"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "trigger_types": trigger_types,
            "total_types": len(trigger_types)
        },
        "message": "Trigger types retrieved successfully"
    }


@router.get("/action-types", response_model=Dict[str, Any])
async def get_action_types():
    """Get available action types."""
    action_types = {
        "create_content": {
            "name": "Create Content",
            "description": "Create new content (posts, articles, etc.)",
            "icon": "📝",
            "category": "content"
        },
        "update_content": {
            "name": "Update Content",
            "description": "Update existing content",
            "icon": "✏️",
            "category": "content"
        },
        "delete_content": {
            "name": "Delete Content",
            "description": "Delete content",
            "icon": "🗑️",
            "category": "content"
        },
        "publish_content": {
            "name": "Publish Content",
            "description": "Publish content to make it live",
            "icon": "📢",
            "category": "content"
        },
        "send_notification": {
            "name": "Send Notification",
            "description": "Send notification to users",
            "icon": "🔔",
            "category": "notification"
        },
        "create_user": {
            "name": "Create User",
            "description": "Create new user account",
            "icon": "👤",
            "category": "user"
        },
        "update_user": {
            "name": "Update User",
            "description": "Update user information",
            "icon": "👥",
            "category": "user"
        },
        "send_email": {
            "name": "Send Email",
            "description": "Send email to users",
            "icon": "📧",
            "category": "notification"
        },
        "post_social": {
            "name": "Post to Social Media",
            "description": "Post content to social media platforms",
            "icon": "📱",
            "category": "social"
        },
        "create_order": {
            "name": "Create Order",
            "description": "Create new e-commerce order",
            "icon": "🛒",
            "category": "ecommerce"
        },
        "update_inventory": {
            "name": "Update Inventory",
            "description": "Update product inventory levels",
            "icon": "📦",
            "category": "ecommerce"
        },
        "assign_badge": {
            "name": "Assign Badge",
            "description": "Assign achievement badge to user",
            "icon": "🏆",
            "category": "gamification"
        },
        "update_points": {
            "name": "Update Points",
            "description": "Update user points or XP",
            "icon": "⭐",
            "category": "gamification"
        },
        "create_course": {
            "name": "Create Course",
            "description": "Create new learning course",
            "icon": "🎓",
            "category": "learning"
        },
        "enroll_user": {
            "name": "Enroll User",
            "description": "Enroll user in course",
            "icon": "📚",
            "category": "learning"
        },
        "execute_ai_task": {
            "name": "Execute AI Task",
            "description": "Execute AI-powered task",
            "icon": "🤖",
            "category": "ai"
        },
        "generate_report": {
            "name": "Generate Report",
            "description": "Generate analytics or system report",
            "icon": "📊",
            "category": "analytics"
        },
        "backup_data": {
            "name": "Backup Data",
            "description": "Backup system data",
            "icon": "💾",
            "category": "system"
        },
        "cleanup_logs": {
            "name": "Cleanup Logs",
            "description": "Clean up old log files",
            "icon": "🧹",
            "category": "system"
        },
        "custom_script": {
            "name": "Custom Script",
            "description": "Execute custom automation script",
            "icon": "⚙️",
            "category": "system"
        }
    }
    
    return {
        "success": True,
        "data": {
            "action_types": action_types,
            "total_types": len(action_types)
        },
        "message": "Action types retrieved successfully"
    }


@router.get("/condition-operators", response_model=Dict[str, Any])
async def get_condition_operators():
    """Get available condition operators."""
    condition_operators = {
        "equals": {
            "name": "Equals",
            "description": "Field value equals specified value",
            "icon": "=",
            "data_types": ["string", "number", "boolean"]
        },
        "not_equals": {
            "name": "Not Equals",
            "description": "Field value does not equal specified value",
            "icon": "≠",
            "data_types": ["string", "number", "boolean"]
        },
        "greater_than": {
            "name": "Greater Than",
            "description": "Field value is greater than specified value",
            "icon": ">",
            "data_types": ["number", "date"]
        },
        "less_than": {
            "name": "Less Than",
            "description": "Field value is less than specified value",
            "icon": "<",
            "data_types": ["number", "date"]
        },
        "greater_equal": {
            "name": "Greater or Equal",
            "description": "Field value is greater than or equal to specified value",
            "icon": "≥",
            "data_types": ["number", "date"]
        },
        "less_equal": {
            "name": "Less or Equal",
            "description": "Field value is less than or equal to specified value",
            "icon": "≤",
            "data_types": ["number", "date"]
        },
        "contains": {
            "name": "Contains",
            "description": "Field value contains specified text",
            "icon": "⊃",
            "data_types": ["string"]
        },
        "not_contains": {
            "name": "Not Contains",
            "description": "Field value does not contain specified text",
            "icon": "⊅",
            "data_types": ["string"]
        },
        "starts_with": {
            "name": "Starts With",
            "description": "Field value starts with specified text",
            "icon": "→",
            "data_types": ["string"]
        },
        "ends_with": {
            "name": "Ends With",
            "description": "Field value ends with specified text",
            "icon": "←",
            "data_types": ["string"]
        },
        "regex": {
            "name": "Regular Expression",
            "description": "Field value matches regular expression",
            "icon": ".*",
            "data_types": ["string"]
        },
        "in": {
            "name": "In List",
            "description": "Field value is in specified list",
            "icon": "∈",
            "data_types": ["string", "number"]
        },
        "not_in": {
            "name": "Not In List",
            "description": "Field value is not in specified list",
            "icon": "∉",
            "data_types": ["string", "number"]
        },
        "is_null": {
            "name": "Is Null",
            "description": "Field value is null or empty",
            "icon": "∅",
            "data_types": ["any"]
        },
        "is_not_null": {
            "name": "Is Not Null",
            "description": "Field value is not null or empty",
            "icon": "∅̸",
            "data_types": ["any"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "condition_operators": condition_operators,
            "total_operators": len(condition_operators)
        },
        "message": "Condition operators retrieved successfully"
    }


@router.get("/execution-statuses", response_model=Dict[str, Any])
async def get_execution_statuses():
    """Get available execution statuses."""
    execution_statuses = {
        "pending": {
            "name": "Pending",
            "description": "Automation is waiting to be executed",
            "icon": "⏳",
            "color": "yellow"
        },
        "running": {
            "name": "Running",
            "description": "Automation is currently executing",
            "icon": "🔄",
            "color": "blue"
        },
        "completed": {
            "name": "Completed",
            "description": "Automation completed successfully",
            "icon": "✅",
            "color": "green"
        },
        "failed": {
            "name": "Failed",
            "description": "Automation execution failed",
            "icon": "❌",
            "color": "red"
        },
        "cancelled": {
            "name": "Cancelled",
            "description": "Automation was cancelled",
            "icon": "⏹️",
            "color": "gray"
        },
        "skipped": {
            "name": "Skipped",
            "description": "Automation was skipped due to conditions",
            "icon": "⏭️",
            "color": "orange"
        },
        "retrying": {
            "name": "Retrying",
            "description": "Automation is retrying after failure",
            "icon": "🔄",
            "color": "purple"
        }
    }
    
    return {
        "success": True,
        "data": {
            "execution_statuses": execution_statuses,
            "total_statuses": len(execution_statuses)
        },
        "message": "Execution statuses retrieved successfully"
    }


@router.get("/templates", response_model=Dict[str, Any])
async def get_automation_templates():
    """Get available automation templates."""
    templates = {
        "content_publishing": {
            "name": "Content Publishing Automation",
            "description": "Automatically publish content based on schedule",
            "icon": "📝",
            "category": "content",
            "complexity": "simple",
            "estimated_setup_time": "5 minutes"
        },
        "user_onboarding": {
            "name": "User Onboarding Automation",
            "description": "Automated user onboarding workflow",
            "icon": "👤",
            "category": "system",
            "complexity": "medium",
            "estimated_setup_time": "15 minutes"
        },
        "social_engagement": {
            "name": "Social Engagement Automation",
            "description": "Automated social media posting and engagement",
            "icon": "📱",
            "category": "social",
            "complexity": "medium",
            "estimated_setup_time": "20 minutes"
        },
        "ecommerce_fulfillment": {
            "name": "E-commerce Fulfillment Automation",
            "description": "Automated order processing and fulfillment",
            "icon": "🛒",
            "category": "ecommerce",
            "complexity": "complex",
            "estimated_setup_time": "30 minutes"
        },
        "learning_progress": {
            "name": "Learning Progress Automation",
            "description": "Automated learning progress tracking and notifications",
            "icon": "🎓",
            "category": "learning",
            "complexity": "medium",
            "estimated_setup_time": "15 minutes"
        },
        "ai_content_generation": {
            "name": "AI Content Generation Automation",
            "description": "Automated AI-powered content generation",
            "icon": "🤖",
            "category": "ai",
            "complexity": "complex",
            "estimated_setup_time": "25 minutes"
        }
    }
    
    return {
        "success": True,
        "data": {
            "templates": templates,
            "total_templates": len(templates)
        },
        "message": "Automation templates retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_automation_health(
    automation_service: AdvancedAutomationService = Depends(get_automation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get automation system health status."""
    try:
        # Get automation stats
        stats = await automation_service.get_automation_stats()
        
        # Calculate health metrics
        total_rules = stats["data"].get("total_rules", 0)
        total_workflows = stats["data"].get("total_workflows", 0)
        total_executions = stats["data"].get("total_executions", 0)
        total_schedules = stats["data"].get("total_schedules", 0)
        total_webhooks = stats["data"].get("total_webhooks", 0)
        active_rules = stats["data"].get("active_rules", 0)
        inactive_rules = stats["data"].get("inactive_rules", 0)
        rules_by_type = stats["data"].get("rules_by_type", {})
        available_templates = stats["data"].get("available_templates", 0)
        
        # Calculate health score
        health_score = 100
        
        # Check rule distribution
        if total_rules > 0:
            active_ratio = active_rules / total_rules
            if active_ratio < 0.5:
                health_score -= 20
            elif active_ratio > 0.9:
                health_score -= 10
        
        # Check automation diversity
        if len(rules_by_type) < 3:
            health_score -= 15
        elif len(rules_by_type) > 8:
            health_score -= 5
        
        # Check execution activity
        if total_rules > 0:
            executions_per_rule = total_executions / total_rules
            if executions_per_rule < 1:
                health_score -= 25
            elif executions_per_rule > 100:
                health_score -= 10
        
        # Check schedule coverage
        if total_rules > 0:
            schedule_ratio = total_schedules / total_rules
            if schedule_ratio < 0.2:
                health_score -= 15
        
        # Check webhook integration
        if total_webhooks < 1:
            health_score -= 10
        
        # Check template availability
        if available_templates < 5:
            health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_rules": total_rules,
                "total_workflows": total_workflows,
                "total_executions": total_executions,
                "total_schedules": total_schedules,
                "total_webhooks": total_webhooks,
                "active_rules": active_rules,
                "inactive_rules": inactive_rules,
                "active_ratio": active_ratio if total_rules > 0 else 0,
                "automation_diversity": len(rules_by_type),
                "executions_per_rule": executions_per_rule if total_rules > 0 else 0,
                "schedule_ratio": schedule_ratio if total_rules > 0 else 0,
                "rules_by_type": rules_by_type,
                "available_templates": available_templates,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Automation health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get automation health status"
        )
























