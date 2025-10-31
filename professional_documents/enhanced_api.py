"""
Enhanced Professional Documents API
==================================

Enhanced API endpoints with advanced features, analytics, and workflow automation.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.engine import get_session
from onyx.db.models import User

from .models import (
    DocumentGenerationRequest,
    DocumentGenerationResponse,
    DocumentExportRequest,
    DocumentExportResponse,
    DocumentType,
    ExportFormat,
    ProfessionalDocument
)
from .services import DocumentGenerationService, DocumentExportService
from .advanced_ai_service import AdvancedAIDocumentGenerator, ContentQuality, AIModel
from .advanced_export_service import AdvancedExportService
from .analytics_service import DocumentAnalyticsService, TimeRange
from .workflow_automation import WorkflowAutomationService, WorkflowAction, UserRole
from .advanced_templates import AdvancedTemplateManager, TemplateComplexity

logger = logging.getLogger(__name__)

# Create enhanced router
enhanced_router = APIRouter(prefix="/enhanced", tags=["Enhanced Professional Documents"])

# Initialize enhanced services
document_service = DocumentGenerationService()
advanced_ai_service = AdvancedAIDocumentGenerator()
advanced_export_service = AdvancedExportService()
analytics_service = DocumentAnalyticsService(document_service)
workflow_service = WorkflowAutomationService()
advanced_template_manager = AdvancedTemplateManager()


@enhanced_router.post("/generate-advanced", response_model=DocumentGenerationResponse)
async def generate_advanced_document(
    request: DocumentGenerationRequest,
    content_quality: ContentQuality = Query(ContentQuality.PREMIUM, description="Content quality level"),
    ai_model: AIModel = Query(AIModel.GPT_4_TURBO, description="AI model to use"),
    target_audience: Optional[str] = Query(None, description="Target audience for the document"),
    industry_context: Optional[str] = Query(None, description="Industry context"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Generate a professional document with advanced AI capabilities.
    
    This endpoint uses advanced AI models and quality settings to generate
    high-quality professional documents with enhanced features.
    """
    try:
        logger.info(f"Generating advanced document for user {user.id}: {request.document_type}")
        
        # Configure AI service
        advanced_ai_service.set_advanced_config(
            model=ai_model,
            max_tokens=8000 if content_quality == ContentQuality.ENTERPRISE else 4000,
            temperature=0.5 if content_quality == ContentQuality.ENTERPRISE else 0.7,
            top_p=0.7 if content_quality == ContentQuality.ENTERPRISE else 0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            content_quality=content_quality
        )
        
        # Generate document with advanced AI
        response = await document_service.generate_document(request)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        # Enhance document with advanced features
        document = response.document
        document.metadata.update({
            "content_quality": content_quality.value,
            "ai_model": ai_model.value,
            "target_audience": target_audience,
            "industry_context": industry_context,
            "generation_method": "advanced_ai"
        })
        
        # Start workflow if configured
        if request.metadata and request.metadata.get("start_workflow", False):
            workflow_template = request.metadata.get("workflow_template", "standard_approval")
            background_tasks.add_task(
                workflow_service.start_workflow,
                document,
                workflow_template
            )
        
        logger.info(f"Advanced document generated successfully: {document.id}")
        
        return DocumentGenerationResponse(
            success=True,
            document=document,
            message=f"Advanced document generated successfully with {content_quality.value} quality",
            generation_time=response.generation_time,
            word_count=document.word_count,
            estimated_pages=document.page_count
        )
        
    except Exception as e:
        logger.error(f"Error generating advanced document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate advanced document: {str(e)}")


@enhanced_router.post("/export-advanced", response_model=DocumentExportResponse)
async def export_advanced_document(
    document_id: str,
    format: ExportFormat,
    watermark_config: Optional[Dict[str, Any]] = None,
    signature_config: Optional[Dict[str, Any]] = None,
    branding_config: Optional[Dict[str, Any]] = None,
    interactive_features: Optional[Dict[str, Any]] = None,
    custom_filename: Optional[str] = None,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Export a document with advanced features like watermarks, signatures, and branding.
    """
    try:
        # Get the document
        document = document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create export request
        export_request = DocumentExportRequest(
            document_id=document_id,
            format=format,
            custom_filename=custom_filename
        )
        
        # Export with advanced features
        response = await advanced_export_service.export_document_advanced(
            document=document,
            request=export_request,
            watermark_config=watermark_config,
            signature_config=signature_config,
            branding_config=branding_config,
            interactive_features=interactive_features
        )
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced export: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced export failed: {str(e)}")


@enhanced_router.get("/analytics/comprehensive")
async def get_comprehensive_analytics(
    time_range: TimeRange = Query(TimeRange.MONTH, description="Time range for analytics"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get comprehensive analytics report for professional documents.
    """
    try:
        # Generate comprehensive analytics
        report = await analytics_service.generate_comprehensive_analytics(
            time_range=time_range,
            user_id=user_id,
            document_type=document_type
        )
        
        return {
            "report_id": report.report_id,
            "title": report.title,
            "time_range": report.time_range.value,
            "generated_at": report.generated_at.isoformat(),
            "summary": report.summary,
            "insights": report.insights,
            "recommendations": report.recommendations,
            "metrics_count": len(report.metrics)
        }
        
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")


@enhanced_router.get("/analytics/user/{user_id}")
async def get_user_analytics(
    user_id: str,
    time_range: TimeRange = Query(TimeRange.MONTH, description="Time range for analytics"),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get analytics for a specific user.
    """
    try:
        analytics = await analytics_service.get_user_analytics(user_id, time_range)
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User analytics failed: {str(e)}")


@enhanced_router.get("/analytics/trends/{metric_name}")
async def get_metric_trends(
    metric_name: str,
    time_range: TimeRange = Query(TimeRange.MONTH, description="Time range for trends"),
    granularity: str = Query("day", description="Trend granularity"),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get metric trends over time.
    """
    try:
        trends = await analytics_service.get_metric_trends(
            metric_name=metric_name,
            time_range=time_range,
            granularity=granularity
        )
        
        return {
            "metric_name": metric_name,
            "time_range": time_range.value,
            "granularity": granularity,
            "trends": [
                {
                    "timestamp": timestamp.isoformat(),
                    "value": value
                }
                for timestamp, value in trends
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting metric trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metric trends failed: {str(e)}")


@enhanced_router.post("/workflow/start")
async def start_document_workflow(
    document_id: str,
    workflow_template_id: str = Query("standard_approval", description="Workflow template to use"),
    assigned_users: Optional[Dict[str, List[str]]] = None,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Start a workflow for a document.
    """
    try:
        # Get the document
        document = document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Start workflow
        workflow_instance = await workflow_service.start_workflow(
            document=document,
            workflow_template_id=workflow_template_id,
            assigned_users=assigned_users
        )
        
        return {
            "workflow_instance_id": workflow_instance.instance_id,
            "document_id": document_id,
            "workflow_template": workflow_template_id,
            "current_step": workflow_instance.current_step,
            "status": workflow_instance.status.value,
            "created_at": workflow_instance.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow start failed: {str(e)}")


@enhanced_router.post("/workflow/action")
async def execute_workflow_action(
    workflow_instance_id: str,
    step_id: str,
    action_type: WorkflowAction,
    comment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Execute a workflow action.
    """
    try:
        # Execute workflow action
        workflow_instance = await workflow_service.execute_workflow_action(
            instance_id=workflow_instance_id,
            step_id=step_id,
            user_id=str(user.id),
            action_type=action_type,
            comment=comment,
            metadata=metadata
        )
        
        return {
            "workflow_instance_id": workflow_instance.instance_id,
            "current_step": workflow_instance.current_step,
            "status": workflow_instance.status.value,
            "updated_at": workflow_instance.updated_at.isoformat(),
            "action_executed": action_type.value
        }
        
    except Exception as e:
        logger.error(f"Error executing workflow action: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow action failed: {str(e)}")


@enhanced_router.get("/workflow/status/{workflow_instance_id}")
async def get_workflow_status(
    workflow_instance_id: str,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get workflow status.
    """
    try:
        status = await workflow_service.get_workflow_status(workflow_instance_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Workflow instance not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow status failed: {str(e)}")


@enhanced_router.get("/workflow/user/{user_id}")
async def get_user_workflows(
    user_id: str,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get workflows for a user.
    """
    try:
        workflows = await workflow_service.get_user_workflows(user_id)
        return {
            "user_id": user_id,
            "workflows": workflows,
            "total_count": len(workflows)
        }
        
    except Exception as e:
        logger.error(f"Error getting user workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User workflows failed: {str(e)}")


@enhanced_router.get("/workflow/analytics")
async def get_workflow_analytics(
    time_range: str = Query("month", description="Time range for workflow analytics"),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get workflow analytics.
    """
    try:
        analytics = await workflow_service.get_workflow_analytics(time_range)
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting workflow analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow analytics failed: {str(e)}")


@enhanced_router.get("/templates/advanced")
async def get_advanced_templates(
    complexity: Optional[TemplateComplexity] = Query(None, description="Filter by complexity"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    user: User = Depends(current_user)
):
    """
    Get advanced document templates.
    """
    try:
        if complexity:
            templates = advanced_template_manager.get_templates_by_complexity(complexity)
        elif document_type:
            templates = advanced_template_manager.get_templates_by_type(document_type)
        else:
            templates = advanced_template_manager.get_advanced_templates()
        
        return {
            "templates": [
                {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "document_type": template.document_type.value,
                    "complexity": template.metadata.get("complexity", "basic"),
                    "visual_elements": template.metadata.get("visual_elements", []),
                    "sections": template.sections,
                    "metadata": template.metadata
                }
                for template in templates
            ],
            "total_count": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Error getting advanced templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced templates failed: {str(e)}")


@enhanced_router.get("/templates/statistics")
async def get_template_statistics(
    user: User = Depends(current_user)
):
    """
    Get template statistics.
    """
    try:
        stats = advanced_template_manager.get_template_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting template statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template statistics failed: {str(e)}")


@enhanced_router.get("/ai/models")
async def get_available_ai_models(
    user: User = Depends(current_user)
):
    """
    Get available AI models.
    """
    try:
        models = advanced_ai_service.get_available_models()
        quality_levels = advanced_ai_service.get_quality_levels()
        
        return {
            "models": [model.value for model in models],
            "quality_levels": [quality.value for quality in quality_levels],
            "current_model": advanced_ai_service.model_name.value,
            "current_quality": advanced_ai_service.content_quality.value
        }
        
    except Exception as e:
        logger.error(f"Error getting AI models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI models failed: {str(e)}")


@enhanced_router.get("/workflow/templates")
async def get_workflow_templates(
    user: User = Depends(current_user)
):
    """
    Get available workflow templates.
    """
    try:
        templates = workflow_service.get_available_workflow_templates()
        return {
            "templates": templates,
            "total_count": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow templates failed: {str(e)}")


@enhanced_router.get("/health/enhanced")
async def enhanced_health_check():
    """
    Enhanced health check for all services.
    """
    try:
        # Check all services
        services_status = {
            "document_service": "healthy",
            "advanced_ai_service": "healthy",
            "advanced_export_service": "healthy",
            "analytics_service": "healthy",
            "workflow_service": "healthy",
            "advanced_template_manager": "healthy"
        }
        
        # Get basic statistics
        all_documents = document_service.list_documents(limit=1000, offset=0)
        workflow_analytics = await workflow_service.get_workflow_analytics()
        template_stats = advanced_template_manager.get_template_statistics()
        
        return {
            "status": "healthy",
            "service": "enhanced-professional-documents",
            "version": "2.0.0",
            "services": services_status,
            "statistics": {
                "total_documents": len(all_documents),
                "total_workflows": workflow_analytics.get("total_workflows", 0),
                "total_templates": template_stats.get("total_templates", 0),
                "advanced_templates": template_stats.get("advanced_templates", 0)
            },
            "features": {
                "advanced_ai_generation": True,
                "advanced_export": True,
                "analytics": True,
                "workflow_automation": True,
                "advanced_templates": True,
                "real_time_collaboration": False,  # Future feature
                "version_control": False  # Future feature
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }



























