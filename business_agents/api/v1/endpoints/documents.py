"""
Advanced Document Generation API Endpoints
=========================================

Comprehensive API endpoints for document generation, template management, and report creation.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
import redis
from io import BytesIO

from ...schemas import (
    BusinessAgent, AgentRequest, AgentResponse, AgentAnalytics,
    AgentWorkflow, AgentCollaboration, AgentSettings,
    ErrorResponse, SuccessResponse
)
from ...exceptions import (
    DocumentGenerationError, TemplateNotFoundError, DocumentValidationError,
    DocumentExportError, DocumentSystemError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ...document_generator import (
    DocumentGenerator, DocumentType, DocumentFormat, TemplateType,
    DocumentTemplate, DocumentRequest, GeneratedDocument
)
from ...models import db_manager
from ...config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Generation"])


# Dependency injection
async def get_document_generator(
    db: AsyncSession = Depends(db_manager.get_session),
    redis_client: redis.Redis = Depends(lambda: redis.Redis())
) -> DocumentGenerator:
    """Get document generator instance"""
    return DocumentGenerator(db, redis_client)


# Document Generation Endpoints
@router.post("/generate", response_model=Dict[str, Any])
async def generate_document(
    template_id: str = Query(..., description="Template ID for document generation"),
    format: DocumentFormat = Query(DocumentFormat.PDF, description="Output document format"),
    data: Dict[str, Any] = None,
    options: Dict[str, Any] = None,
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Generate document from template
    
    - **template_id**: Template ID for document generation
    - **format**: Output document format (PDF, DOCX, PPTX, XLSX, HTML, Markdown)
    - **data**: Data to populate template variables
    - **options**: Additional generation options
    - **user_id**: User ID for audit logging
    """
    try:
        # Generate document
        document = await document_generator.generate_document(
            template_id=template_id,
            data=data or {},
            format=format,
            options=options or {},
            user_id=user_id
        )
        
        return {
            "success": True,
            "message": "Document generated successfully",
            "data": {
                "document_id": document.document_id,
                "name": document.name,
                "document_type": document.document_type.value,
                "format": document.format.value,
                "file_size": document.file_size,
                "generated_at": document.generated_at.isoformat(),
                "metadata": document.metadata
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, template_id=template_id, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.post("/agent-report", response_model=Dict[str, Any])
async def generate_agent_report(
    agent_id: str = Query(..., description="Agent ID for report generation"),
    report_period: str = Query("30d", description="Report period (e.g., 30d, 7d, 1y)"),
    format: DocumentFormat = Query(DocumentFormat.PDF, description="Output document format"),
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Generate agent performance report
    
    - **agent_id**: Agent ID for report generation
    - **report_period**: Report period (e.g., 30d, 7d, 1y)
    - **format**: Output document format
    - **user_id**: User ID for audit logging
    """
    try:
        # Generate agent report
        document = await document_generator.generate_agent_report(
            agent_id=agent_id,
            report_period=report_period,
            format=format,
            user_id=user_id
        )
        
        return {
            "success": True,
            "message": "Agent report generated successfully",
            "data": {
                "document_id": document.document_id,
                "name": document.name,
                "document_type": document.document_type.value,
                "format": document.format.value,
                "file_size": document.file_size,
                "generated_at": document.generated_at.isoformat(),
                "metadata": document.metadata
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, agent_id=agent_id, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.post("/workflow-docs", response_model=Dict[str, Any])
async def generate_workflow_documentation(
    workflow_id: str = Query(..., description="Workflow ID for documentation generation"),
    format: DocumentFormat = Query(DocumentFormat.DOCX, description="Output document format"),
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Generate workflow documentation
    
    - **workflow_id**: Workflow ID for documentation generation
    - **format**: Output document format
    - **user_id**: User ID for audit logging
    """
    try:
        # Generate workflow documentation
        document = await document_generator.generate_workflow_documentation(
            workflow_id=workflow_id,
            format=format,
            user_id=user_id
        )
        
        return {
            "success": True,
            "message": "Workflow documentation generated successfully",
            "data": {
                "document_id": document.document_id,
                "name": document.name,
                "document_type": document.document_type.value,
                "format": document.format.value,
                "file_size": document.file_size,
                "generated_at": document.generated_at.isoformat(),
                "metadata": document.metadata
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, workflow_id=workflow_id, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.post("/analytics-dashboard", response_model=Dict[str, Any])
async def generate_analytics_dashboard(
    dashboard_type: str = Query(..., description="Dashboard type (system, agents, workflows)"),
    time_period: str = Query("30d", description="Time period for analytics"),
    format: DocumentFormat = Query(DocumentFormat.HTML, description="Output document format"),
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Generate analytics dashboard
    
    - **dashboard_type**: Dashboard type (system, agents, workflows)
    - **time_period**: Time period for analytics
    - **format**: Output document format
    - **user_id**: User ID for audit logging
    """
    try:
        # Generate analytics dashboard
        document = await document_generator.generate_analytics_dashboard(
            dashboard_type=dashboard_type,
            time_period=time_period,
            format=format,
            user_id=user_id
        )
        
        return {
            "success": True,
            "message": "Analytics dashboard generated successfully",
            "data": {
                "document_id": document.document_id,
                "name": document.name,
                "document_type": document.document_type.value,
                "format": document.format.value,
                "file_size": document.file_size,
                "generated_at": document.generated_at.isoformat(),
                "metadata": document.metadata
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, dashboard_type=dashboard_type, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Template Management Endpoints
@router.post("/templates", response_model=Dict[str, Any])
async def create_custom_template(
    name: str = Query(..., description="Template name"),
    description: str = Query(..., description="Template description"),
    document_type: DocumentType = Query(..., description="Document type"),
    format: DocumentFormat = Query(..., description="Document format"),
    content: str = Query(..., description="Template content"),
    variables: Dict[str, Any] = None,
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Create custom document template
    
    - **name**: Template name
    - **description**: Template description
    - **document_type**: Document type
    - **format**: Document format
    - **content**: Template content
    - **variables**: Template variables
    - **user_id**: User ID for audit logging
    """
    try:
        # Create custom template
        template = await document_generator.create_custom_template(
            name=name,
            description=description,
            document_type=document_type,
            format=format,
            content=content,
            variables=variables or {},
            user_id=user_id
        )
        
        return {
            "success": True,
            "message": "Custom template created successfully",
            "data": {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "document_type": template.document_type.value,
                "template_type": template.template_type.value,
                "format": template.format.value,
                "variables": template.variables,
                "created_by": template.created_by,
                "created_at": template.created_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/templates", response_model=Dict[str, Any])
async def list_templates(
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    template_type: Optional[TemplateType] = Query(None, description="Filter by template type"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    List available templates
    
    - **document_type**: Filter by document type
    - **template_type**: Filter by template type
    """
    try:
        # List templates
        templates = await document_generator.list_templates(
            document_type=document_type,
            template_type=template_type
        )
        
        return {
            "success": True,
            "message": "Templates retrieved successfully",
            "data": {
                "templates": [
                    {
                        "template_id": template.template_id,
                        "name": template.name,
                        "description": template.description,
                        "document_type": template.document_type.value,
                        "template_type": template.template_type.value,
                        "format": template.format.value,
                        "variables": template.variables,
                        "created_by": template.created_by,
                        "created_at": template.created_at.isoformat(),
                        "updated_at": template.updated_at.isoformat()
                    }
                    for template in templates
                ],
                "total_count": len(templates)
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/templates/{template_id}", response_model=Dict[str, Any])
async def get_template(
    template_id: str = Path(..., description="Template ID"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Get specific template
    
    - **template_id**: Template ID
    """
    try:
        # Get template
        template = await document_generator.get_template(template_id)
        
        if not template:
            raise TemplateNotFoundError(
                "template_not_found",
                f"Template {template_id} not found",
                {"template_id": template_id}
            )
        
        return {
            "success": True,
            "message": "Template retrieved successfully",
            "data": {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "document_type": template.document_type.value,
                "template_type": template.template_type.value,
                "format": template.format.value,
                "content": template.content,
                "variables": template.variables,
                "metadata": template.metadata,
                "created_by": template.created_by,
                "created_at": template.created_at.isoformat(),
                "updated_at": template.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, template_id=template_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.put("/templates/{template_id}", response_model=Dict[str, Any])
async def update_template(
    template_id: str = Path(..., description="Template ID"),
    name: Optional[str] = Query(None, description="Template name"),
    description: Optional[str] = Query(None, description="Template description"),
    content: Optional[str] = Query(None, description="Template content"),
    variables: Optional[Dict[str, Any]] = None,
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Update template
    
    - **template_id**: Template ID
    - **name**: Template name
    - **description**: Template description
    - **content**: Template content
    - **variables**: Template variables
    - **user_id**: User ID for audit logging
    """
    try:
        # Get existing template
        template = await document_generator.get_template(template_id)
        
        if not template:
            raise TemplateNotFoundError(
                "template_not_found",
                f"Template {template_id} not found",
                {"template_id": template_id}
            )
        
        # Update template fields
        if name is not None:
            template.name = name
        if description is not None:
            template.description = description
        if content is not None:
            template.content = content
        if variables is not None:
            template.variables = variables
        
        template.updated_at = datetime.utcnow()
        
        # Store updated template
        await document_generator._store_template(template)
        
        return {
            "success": True,
            "message": "Template updated successfully",
            "data": {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "document_type": template.document_type.value,
                "template_type": template.template_type.value,
                "format": template.format.value,
                "variables": template.variables,
                "created_by": template.created_by,
                "created_at": template.created_at.isoformat(),
                "updated_at": template.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, template_id=template_id, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.delete("/templates/{template_id}", response_model=Dict[str, Any])
async def delete_template(
    template_id: str = Path(..., description="Template ID"),
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Delete template
    
    - **template_id**: Template ID
    - **user_id**: User ID for audit logging
    """
    try:
        # Get existing template
        template = await document_generator.get_template(template_id)
        
        if not template:
            raise TemplateNotFoundError(
                "template_not_found",
                f"Template {template_id} not found",
                {"template_id": template_id}
            )
        
        # Delete template from cache
        if template_id in document_generator.templates:
            del document_generator.templates[template_id]
        
        # TODO: Delete from database
        
        return {
            "success": True,
            "message": "Template deleted successfully",
            "data": {
                "template_id": template_id,
                "deleted_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, template_id=template_id, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Document Download Endpoints
@router.get("/{document_id}/download")
async def download_document(
    document_id: str = Path(..., description="Document ID"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> StreamingResponse:
    """
    Download generated document
    
    - **document_id**: Document ID
    """
    try:
        # TODO: Retrieve document from storage
        # For now, return a placeholder response
        
        # This would typically retrieve the document from storage
        # and return it as a streaming response
        
        raise DocumentSystemError(
            "download_not_implemented",
            "Document download not yet implemented",
            {"document_id": document_id}
        )
        
    except Exception as e:
        error = handle_agent_error(e, document_id=document_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Document Management Endpoints
@router.get("/", response_model=Dict[str, Any])
async def list_documents(
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    format: Optional[DocumentFormat] = Query(None, description="Filter by format"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(50, description="Number of documents to return"),
    offset: int = Query(0, description="Number of documents to skip"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    List generated documents
    
    - **document_type**: Filter by document type
    - **format**: Filter by format
    - **user_id**: Filter by user ID
    - **limit**: Number of documents to return
    - **offset**: Number of documents to skip
    """
    try:
        # TODO: Implement document listing from database
        # For now, return empty list
        
        return {
            "success": True,
            "message": "Documents retrieved successfully",
            "data": {
                "documents": [],
                "total_count": 0,
                "limit": limit,
                "offset": offset
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/{document_id}", response_model=Dict[str, Any])
async def get_document(
    document_id: str = Path(..., description="Document ID"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Get document details
    
    - **document_id**: Document ID
    """
    try:
        # TODO: Implement document retrieval from database
        # For now, return not found
        
        raise DocumentSystemError(
            "document_not_found",
            f"Document {document_id} not found",
            {"document_id": document_id}
        )
        
    except Exception as e:
        error = handle_agent_error(e, document_id=document_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Batch Operations
@router.post("/batch-generate", response_model=Dict[str, Any])
async def batch_generate_documents(
    requests: List[Dict[str, Any]] = None,
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Batch generate multiple documents
    
    - **requests**: List of document generation requests
    - **user_id**: User ID for audit logging
    """
    try:
        if not requests:
            raise DocumentValidationError(
                "invalid_request",
                "No requests provided",
                {"requests": requests}
            )
        
        results = []
        errors = []
        
        for i, request in enumerate(requests):
            try:
                document = await document_generator.generate_document(
                    template_id=request.get("template_id"),
                    data=request.get("data", {}),
                    format=DocumentFormat(request.get("format", "pdf")),
                    options=request.get("options", {}),
                    user_id=user_id
                )
                
                results.append({
                    "index": i,
                    "success": True,
                    "document_id": document.document_id,
                    "name": document.name
                })
                
            except Exception as e:
                errors.append({
                    "index": i,
                    "error": str(e),
                    "request": request
                })
        
        return {
            "success": True,
            "message": f"Batch generation completed: {len(results)} successful, {len(errors)} failed",
            "data": {
                "results": results,
                "errors": errors,
                "total_requests": len(requests),
                "successful": len(results),
                "failed": len(errors)
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Template Upload
@router.post("/templates/upload", response_model=Dict[str, Any])
async def upload_template(
    file: UploadFile = File(..., description="Template file"),
    name: str = Query(..., description="Template name"),
    description: str = Query(..., description="Template description"),
    document_type: DocumentType = Query(..., description="Document type"),
    format: DocumentFormat = Query(..., description="Document format"),
    user_id: str = Query(None, description="User ID for audit logging"),
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Upload template file
    
    - **file**: Template file
    - **name**: Template name
    - **description**: Template description
    - **document_type**: Document type
    - **format**: Document format
    - **user_id**: User ID for audit logging
    """
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Create template
        template = await document_generator.create_custom_template(
            name=name,
            description=description,
            document_type=document_type,
            format=format,
            content=content_str,
            user_id=user_id
        )
        
        return {
            "success": True,
            "message": "Template uploaded successfully",
            "data": {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "document_type": template.document_type.value,
                "template_type": template.template_type.value,
                "format": template.format.value,
                "file_size": len(content),
                "created_by": template.created_by,
                "created_at": template.created_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Health Check
@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    document_generator: DocumentGenerator = Depends(get_document_generator)
) -> Dict[str, Any]:
    """
    Document generation service health check
    """
    try:
        # Check template availability
        templates = await document_generator.list_templates()
        
        return {
            "success": True,
            "message": "Document generation service is healthy",
            "data": {
                "status": "healthy",
                "templates_available": len(templates),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )





























