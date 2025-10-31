"""
Enhanced API v2
==============

Advanced API endpoints for the enhanced professional documents system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from .models import (
    DocumentGenerationRequest, DocumentGenerationResponse,
    DocumentExportRequest, DocumentListResponse, DocumentInfo
)
from .real_time_collaboration import RealTimeCollaborationService, CollaborationAction
from .version_control import VersionControlService, VersionType
from .ai_insights_service import AIInsightsService, InsightType
from .document_comparison import DocumentComparisonService, ComparisonType
from .smart_templates import SmartTemplatesService, TemplateType
from .document_security import DocumentSecurityService, SecurityLevel, AccessType

logger = logging.getLogger(__name__)

# Initialize services
collaboration_service = RealTimeCollaborationService()
version_control_service = VersionControlService()
ai_insights_service = AIInsightsService()
comparison_service = DocumentComparisonService()
smart_templates_service = SmartTemplatesService()
security_service = DocumentSecurityService()

router = APIRouter()


# Pydantic models for new endpoints
class CollaborationJoinRequest(BaseModel):
    user_id: str
    username: str
    email: str
    role: str = "collaborator"


class CollaborationEventRequest(BaseModel):
    action: str
    data: Dict[str, Any]


class VersionCreateRequest(BaseModel):
    title: str
    content: str
    metadata: Dict[str, Any] = {}
    change_summary: str
    version_type: VersionType = VersionType.MINOR
    tags: List[str] = []


class VersionRestoreRequest(BaseModel):
    version_id: str
    reason: str


class DocumentAnalysisRequest(BaseModel):
    content: str
    title: str = ""
    document_type: str = "general"


class DocumentComparisonRequest(BaseModel):
    document1_id: str
    document2_id: str
    document1_content: str
    document2_content: str
    document1_metadata: Dict[str, Any] = {}
    document2_metadata: Dict[str, Any] = {}
    comparison_type: ComparisonType = ComparisonType.COMPREHENSIVE


class TemplateSuggestionRequest(BaseModel):
    content_preview: str
    user_preferences: Dict[str, Any] = {}


class SecurityPolicyRequest(BaseModel):
    document_id: str
    policy_id: str
    content: str
    metadata: Dict[str, Any] = {}


class AccessGrantRequest(BaseModel):
    user_id: str
    access_type: AccessType
    expires_at: Optional[datetime] = None
    reason: str = ""


# Real-time Collaboration Endpoints
@router.websocket("/collaboration/{document_id}/ws")
async def collaboration_websocket(websocket: WebSocket, document_id: str):
    """WebSocket endpoint for real-time collaboration."""
    
    await websocket.accept()
    user_id = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "join":
                join_data = message["data"]
                user = await collaboration_service.join_document_session(
                    document_id=document_id,
                    user_id=join_data["user_id"],
                    username=join_data["username"],
                    email=join_data["email"],
                    role=join_data.get("role", "collaborator"),
                    websocket=websocket
                )
                user_id = join_data["user_id"]
                
            elif message["type"] == "collaboration_event":
                if user_id:
                    event_data = message["data"]
                    await collaboration_service.handle_collaboration_event(
                        document_id=document_id,
                        user_id=user_id,
                        action=CollaborationAction(event_data["action"]),
                        data=event_data["data"]
                    )
                    
            elif message["type"] == "leave":
                if user_id:
                    await collaboration_service.leave_document_session(document_id, user_id)
                    break
                    
    except WebSocketDisconnect:
        if user_id:
            await collaboration_service.leave_document_session(document_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()


@router.get("/collaboration/{document_id}/users")
async def get_collaborators(document_id: str):
    """Get list of collaborators for a document."""
    
    try:
        collaborators = await collaboration_service.get_document_collaborators(document_id)
        return {"collaborators": collaborators}
    except Exception as e:
        logger.error(f"Error getting collaborators: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collaboration/{document_id}/comments")
async def get_document_comments(document_id: str):
    """Get comments for a document."""
    
    try:
        comments = await collaboration_service.get_document_comments(document_id)
        return {"comments": comments}
    except Exception as e:
        logger.error(f"Error getting comments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collaboration/{document_id}/comments/{comment_id}/resolve")
async def resolve_comment(document_id: str, comment_id: str, user_id: str = "system"):
    """Resolve a comment."""
    
    try:
        success = await collaboration_service.resolve_comment(document_id, comment_id, user_id)
        if success:
            return {"message": "Comment resolved successfully"}
        else:
            raise HTTPException(status_code=404, detail="Comment not found")
    except Exception as e:
        logger.error(f"Error resolving comment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collaboration/{document_id}/analytics")
async def get_collaboration_analytics(document_id: str):
    """Get collaboration analytics for a document."""
    
    try:
        analytics = await collaboration_service.get_collaboration_analytics(document_id)
        return analytics
    except Exception as e:
        logger.error(f"Error getting collaboration analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Version Control Endpoints
@router.post("/documents/{document_id}/versions")
async def create_document_version(
    document_id: str,
    request: VersionCreateRequest,
    created_by: str = "system"
):
    """Create a new version of a document."""
    
    try:
        # Check if this is the first version
        current_version = await version_control_service.get_current_version(document_id)
        
        if not current_version:
            # Create initial version
            version = await version_control_service.create_initial_version(
                document_id=document_id,
                title=request.title,
                content=request.content,
                metadata=request.metadata,
                created_by=created_by,
                version_type=request.version_type
            )
        else:
            # Create new version
            version = await version_control_service.create_new_version(
                document_id=document_id,
                title=request.title,
                content=request.content,
                metadata=request.metadata,
                created_by=created_by,
                change_summary=request.change_summary,
                version_type=request.version_type,
                tags=request.tags
            )
        
        return {
            "version_id": version.version_id,
            "version_number": version.version_number,
            "created_at": version.created_at.isoformat(),
            "created_by": version.created_by
        }
    except Exception as e:
        logger.error(f"Error creating document version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/versions")
async def get_document_versions(
    document_id: str,
    limit: int = 50,
    offset: int = 0,
    version_type: Optional[VersionType] = None
):
    """Get versions of a document."""
    
    try:
        versions = await version_control_service.get_document_versions(
            document_id=document_id,
            limit=limit,
            offset=offset,
            version_type=version_type
        )
        
        return {
            "versions": [
                {
                    "version_id": v.version_id,
                    "version_number": v.version_number,
                    "version_type": v.version_type.value,
                    "title": v.title,
                    "created_at": v.created_at.isoformat(),
                    "created_by": v.created_by,
                    "change_summary": v.change_summary,
                    "is_current": v.is_current,
                    "tags": v.tags
                }
                for v in versions
            ]
        }
    except Exception as e:
        logger.error(f"Error getting document versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/versions/{version_id}")
async def get_version_by_id(document_id: str, version_id: str):
    """Get specific version by ID."""
    
    try:
        version = await version_control_service.get_version_by_id(document_id, version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {
            "version_id": version.version_id,
            "version_number": version.version_number,
            "version_type": version.version_type.value,
            "title": version.title,
            "content": version.content,
            "metadata": version.metadata,
            "created_at": version.created_at.isoformat(),
            "created_by": version.created_by,
            "change_summary": version.change_summary,
            "is_current": version.is_current,
            "tags": version.tags
        }
    except Exception as e:
        logger.error(f"Error getting version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/{document_id}/versions/{version_id}/restore")
async def restore_version(
    document_id: str,
    version_id: str,
    request: VersionRestoreRequest,
    restored_by: str = "system"
):
    """Restore a previous version."""
    
    try:
        new_version = await version_control_service.restore_version(
            document_id=document_id,
            version_id=version_id,
            restored_by=restored_by,
            reason=request.reason
        )
        
        return {
            "version_id": new_version.version_id,
            "version_number": new_version.version_number,
            "created_at": new_version.created_at.isoformat(),
            "message": f"Restored from version {version_id}"
        }
    except Exception as e:
        logger.error(f"Error restoring version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/versions/{version1_id}/compare/{version2_id}")
async def compare_versions(document_id: str, version1_id: str, version2_id: str):
    """Compare two versions of a document."""
    
    try:
        comparison = await version_control_service.compare_versions(
            document_id=document_id,
            version1_id=version1_id,
            version2_id=version2_id
        )
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparing versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/versions/analytics")
async def get_version_analytics(document_id: str):
    """Get version analytics for a document."""
    
    try:
        analytics = await version_control_service.get_version_analytics(document_id)
        return analytics
    except Exception as e:
        logger.error(f"Error getting version analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# AI Insights Endpoints
@router.post("/documents/analyze")
async def analyze_document(request: DocumentAnalysisRequest):
    """Analyze document content with AI insights."""
    
    try:
        analysis = await ai_insights_service.analyze_document(
            document_id=str(datetime.now().timestamp()),  # Generate temporary ID
            content=request.content,
            title=request.title,
            document_type=request.document_type
        )
        
        return {
            "analysis_id": analysis.analysis_id,
            "overall_score": analysis.overall_score,
            "overall_quality": analysis.overall_quality.value,
            "summary": analysis.summary,
            "insights": [
                {
                    "insight_type": insight.insight_type.value,
                    "title": insight.title,
                    "description": insight.description,
                    "score": insight.score,
                    "quality_level": insight.quality_level.value,
                    "recommendations": insight.recommendations,
                    "confidence": insight.confidence
                }
                for insight in analysis.insights
            ],
            "processing_time": analysis.processing_time,
            "created_at": analysis.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/trends")
async def get_insight_trends(
    insight_type: InsightType,
    days: int = 30,
    document_ids: List[str] = []
):
    """Get insight trends across multiple documents."""
    
    try:
        trends = await ai_insights_service.get_insight_trends(
            document_ids=document_ids,
            insight_type=insight_type,
            days=days
        )
        
        return trends
    except Exception as e:
        logger.error(f"Error getting insight trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Document Comparison Endpoints
@router.post("/documents/compare")
async def compare_documents(request: DocumentComparisonRequest):
    """Compare two documents."""
    
    try:
        comparison = await comparison_service.compare_documents(
            document1_id=request.document1_id,
            document2_id=request.document2_id,
            document1_content=request.document1_content,
            document2_content=request.document2_content,
            document1_metadata=request.document1_metadata,
            document2_metadata=request.document2_metadata,
            comparison_type=request.comparison_type
        )
        
        return {
            "comparison_id": comparison.comparison_id,
            "similarity_score": comparison.similarity_score,
            "summary": comparison.summary,
            "changes": [
                {
                    "change_type": change.change_type.value,
                    "field_name": change.field_name,
                    "description": change.description,
                    "confidence": change.confidence
                }
                for change in comparison.changes
            ],
            "processing_time": comparison.processing_time,
            "created_at": comparison.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/compare/{comparison_id}/report")
async def get_comparison_report(
    comparison_id: str,
    format: str = "json"
):
    """Get comparison report in specified format."""
    
    try:
        # This would typically retrieve from database
        # For now, return mock data
        mock_comparison = type('MockComparison', (), {
            'comparison_id': comparison_id,
            'document1_id': 'doc1',
            'document2_id': 'doc2',
            'comparison_type': ComparisonType.COMPREHENSIVE,
            'changes': [],
            'similarity_score': 85.5,
            'summary': {'total_changes': 5, 'change_severity': 'minor'},
            'created_at': datetime.now(),
            'processing_time': 1.2
        })()
        
        report = await comparison_service.generate_diff_report(mock_comparison, format)
        
        if format == "html":
            return StreamingResponse(
                iter([report]),
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename=comparison_{comparison_id}.html"}
            )
        else:
            return {"report": report}
    except Exception as e:
        logger.error(f"Error generating comparison report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Smart Templates Endpoints
@router.post("/templates/suggest")
async def get_template_suggestions(request: TemplateSuggestionRequest):
    """Get template suggestions based on content preview."""
    
    try:
        suggestions = await smart_templates_service.get_template_suggestions(
            content_preview=request.content_preview,
            user_preferences=request.user_preferences
        )
        
        return {
            "suggestions": [
                {
                    "template_id": suggestion.template.template_id,
                    "template_name": suggestion.template.name,
                    "template_type": suggestion.template.template_type.value,
                    "content_category": suggestion.template.content_category.value,
                    "match_score": suggestion.match_score,
                    "confidence": suggestion.confidence,
                    "reasoning": suggestion.reasoning,
                    "adaptations": suggestion.adaptations
                }
                for suggestion in suggestions
            ]
        }
    except Exception as e:
        logger.error(f"Error getting template suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/{template_id}/apply")
async def apply_template(
    template_id: str,
    content: str,
    context: Dict[str, Any] = {},
    adaptations: List[Dict[str, Any]] = []
):
    """Apply template to content."""
    
    try:
        if template_id not in smart_templates_service.templates:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template = smart_templates_service.templates[template_id]
        result = await smart_templates_service.apply_template(
            template=template,
            content=content,
            adaptations=adaptations,
            context=context
        )
        
        return {"applied_template": result}
    except Exception as e:
        logger.error(f"Error applying template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_available_templates():
    """Get all available templates."""
    
    try:
        templates = []
        for template in smart_templates_service.templates.values():
            if template.is_active:
                templates.append({
                    "template_id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "template_type": template.template_type.value,
                    "content_category": template.content_category.value,
                    "version": template.version,
                    "created_at": template.created_at.isoformat()
                })
        
        return {"templates": templates}
    except Exception as e:
        logger.error(f"Error getting templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Document Security Endpoints
@router.post("/documents/{document_id}/security/apply")
async def apply_security_policy(request: SecurityPolicyRequest):
    """Apply security policy to document."""
    
    try:
        security_result = await security_service.apply_security_policy(
            document_id=request.document_id,
            policy_id=request.policy_id,
            content=request.content,
            metadata=request.metadata
        )
        
        return security_result
    except Exception as e:
        logger.error(f"Error applying security policy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/{document_id}/security/access/grant")
async def grant_document_access(
    document_id: str,
    request: AccessGrantRequest,
    granted_by: str = "system"
):
    """Grant access to document."""
    
    try:
        access = await security_service.grant_access(
            document_id=document_id,
            user_id=request.user_id,
            access_type=request.access_type,
            granted_by=granted_by,
            expires_at=request.expires_at,
            reason=request.reason
        )
        
        return {
            "access_id": access.access_id,
            "granted_at": access.granted_at.isoformat(),
            "expires_at": access.expires_at.isoformat() if access.expires_at else None,
            "message": "Access granted successfully"
        }
    except Exception as e:
        logger.error(f"Error granting access: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}/security/access/revoke")
async def revoke_document_access(
    document_id: str,
    user_id: str,
    access_type: AccessType,
    revoked_by: str = "system",
    reason: str = ""
):
    """Revoke access to document."""
    
    try:
        success = await security_service.revoke_access(
            document_id=document_id,
            user_id=user_id,
            access_type=access_type,
            revoked_by=revoked_by,
            reason=reason
        )
        
        if success:
            return {"message": "Access revoked successfully"}
        else:
            raise HTTPException(status_code=404, detail="Access record not found")
    except Exception as e:
        logger.error(f"Error revoking access: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/security/access/check")
async def check_document_access(
    document_id: str,
    user_id: str,
    access_type: AccessType
):
    """Check if user has access to document."""
    
    try:
        has_access = await security_service.check_access(
            document_id=document_id,
            user_id=user_id,
            access_type=access_type
        )
        
        return {"has_access": has_access}
    except Exception as e:
        logger.error(f"Error checking access: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/security/audit")
async def get_security_audit_log(
    document_id: str,
    limit: int = 100,
    offset: int = 0
):
    """Get security audit log for document."""
    
    try:
        audit_log = await security_service.get_access_log(
            document_id=document_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "audit_events": [
                {
                    "audit_id": audit.audit_id,
                    "user_id": audit.user_id,
                    "action": audit.action,
                    "timestamp": audit.timestamp.isoformat(),
                    "ip_address": audit.ip_address,
                    "success": audit.success,
                    "details": audit.details
                }
                for audit in audit_log
            ]
        }
    except Exception as e:
        logger.error(f"Error getting audit log: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/security/summary")
async def get_security_summary(document_id: str):
    """Get security summary for document."""
    
    try:
        summary = await security_service.get_security_summary(document_id)
        return summary
    except Exception as e:
        logger.error(f"Error getting security summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/policies")
async def get_security_policies():
    """Get all security policies."""
    
    try:
        policies = await security_service.get_security_policies()
        
        return {
            "policies": [
                {
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "description": policy.description,
                    "security_level": policy.security_level.value,
                    "encryption_required": policy.encryption_required,
                    "audit_required": policy.audit_required,
                    "retention_period": policy.retention_period,
                    "watermark_required": policy.watermark_required,
                    "digital_signature_required": policy.digital_signature_required,
                    "created_at": policy.created_at.isoformat()
                }
                for policy in policies
            ]
        }
    except Exception as e:
        logger.error(f"Error getting security policies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/analytics")
async def get_security_analytics():
    """Get security analytics."""
    
    try:
        analytics = await security_service.get_security_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Error getting security analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "collaboration": "active",
            "version_control": "active",
            "ai_insights": "active",
            "document_comparison": "active",
            "smart_templates": "active",
            "document_security": "active"
        }
    }



























