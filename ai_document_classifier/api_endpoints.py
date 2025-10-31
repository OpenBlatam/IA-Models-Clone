"""
API Endpoints for AI Document Classifier
========================================

FastAPI endpoints for document type classification and template export functionality.
"""

from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import tempfile
import os

from .document_classifier_engine import (
    DocumentClassifierEngine, 
    DocumentType, 
    ClassificationResult,
    TemplateDesign
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/ai-document-classifier", tags=["AI Document Classifier"])

# Initialize the classifier engine
classifier_engine = DocumentClassifierEngine()

# Pydantic models for API
class ClassificationRequest(BaseModel):
    """Request model for document classification"""
    query: str = Field(..., description="Text query describing the document to classify")
    use_ai: bool = Field(True, description="Whether to use AI for classification")
    
class ClassificationResponse(BaseModel):
    """Response model for document classification"""
    document_type: str
    confidence: float
    keywords: List[str]
    reasoning: str
    template_suggestions: List[str]

class TemplateExportRequest(BaseModel):
    """Request model for template export"""
    document_type: str
    template_name: Optional[str] = None
    format: str = Field("json", description="Export format: json, yaml, or markdown")

class TemplateListResponse(BaseModel):
    """Response model for template listing"""
    document_type: str
    templates: List[Dict[str, Any]]

class TemplateResponse(BaseModel):
    """Response model for individual template"""
    name: str
    document_type: str
    sections: List[Dict[str, Any]]
    formatting: Dict[str, Any]
    metadata: Dict[str, Any]

@router.post("/classify", response_model=ClassificationResponse)
async def classify_document(request: ClassificationRequest):
    """
    Classify a document type from a text query.
    
    Args:
        request: ClassificationRequest with query text and options
        
    Returns:
        ClassificationResponse with document type and confidence
    """
    try:
        logger.info(f"Classifying document: {request.query[:100]}...")
        
        result = classifier_engine.classify_document(
            query=request.query,
            use_ai=request.use_ai
        )
        
        return ClassificationResponse(
            document_type=result.document_type.value,
            confidence=result.confidence,
            keywords=result.keywords,
            reasoning=result.reasoning,
            template_suggestions=result.template_suggestions
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.get("/document-types", response_model=List[str])
async def get_document_types():
    """
    Get list of supported document types.
    
    Returns:
        List of supported document type names
    """
    return [doc_type.value for doc_type in DocumentType if doc_type != DocumentType.UNKNOWN]

@router.get("/templates/{document_type}", response_model=TemplateListResponse)
async def get_templates(
    document_type: str = Path(..., description="Document type to get templates for")
):
    """
    Get all templates for a specific document type.
    
    Args:
        document_type: Document type name
        
    Returns:
        TemplateListResponse with list of templates
    """
    try:
        # Validate document type
        try:
            doc_type_enum = DocumentType(document_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid document type: {document_type}"
            )
        
        templates = classifier_engine.get_templates(doc_type_enum)
        
        template_data = []
        for template in templates:
            template_data.append({
                "name": template.name,
                "sections": template.sections,
                "formatting": template.formatting,
                "metadata": template.metadata
            })
        
        return TemplateListResponse(
            document_type=document_type,
            templates=template_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.get("/template/{document_type}/{template_name}", response_model=TemplateResponse)
async def get_template(
    document_type: str = Path(..., description="Document type"),
    template_name: str = Path(..., description="Template name")
):
    """
    Get a specific template by document type and name.
    
    Args:
        document_type: Document type name
        template_name: Template name
        
    Returns:
        TemplateResponse with template details
    """
    try:
        # Validate document type
        try:
            doc_type_enum = DocumentType(document_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid document type: {document_type}"
            )
        
        templates = classifier_engine.get_templates(doc_type_enum)
        
        # Find template by name
        template = None
        for t in templates:
            if t.name.lower() == template_name.lower():
                template = t
                break
        
        if not template:
            raise HTTPException(
                status_code=404, 
                detail=f"Template '{template_name}' not found for document type '{document_type}'"
            )
        
        return TemplateResponse(
            name=template.name,
            document_type=template.document_type.value,
            sections=template.sections,
            formatting=template.formatting,
            metadata=template.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")

@router.post("/export-template")
async def export_template(request: TemplateExportRequest):
    """
    Export a template in the specified format.
    
    Args:
        request: TemplateExportRequest with export parameters
        
    Returns:
        Exported template content or file download
    """
    try:
        # Validate document type
        try:
            doc_type_enum = DocumentType(request.document_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid document type: {request.document_type}"
            )
        
        # Validate format
        if request.format not in ["json", "yaml", "markdown"]:
            raise HTTPException(
                status_code=400, 
                detail="Format must be one of: json, yaml, markdown"
            )
        
        templates = classifier_engine.get_templates(doc_type_enum)
        
        # Find template by name or use first one
        template = None
        if request.template_name:
            for t in templates:
                if t.name.lower() == request.template_name.lower():
                    template = t
                    break
            if not template:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Template '{request.template_name}' not found"
                )
        else:
            if not templates:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No templates found for document type '{request.document_type}'"
                )
            template = templates[0]
        
        # Export template
        exported_content = classifier_engine.export_template(template, request.format)
        
        # Return as file download for certain formats
        if request.format in ["yaml", "markdown"]:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'.{request.format}', 
                delete=False,
                encoding='utf-8'
            ) as temp_file:
                temp_file.write(exported_content)
                temp_file_path = temp_file.name
            
            filename = f"{template.name.lower().replace(' ', '_')}.{request.format}"
            
            return FileResponse(
                path=temp_file_path,
                filename=filename,
                media_type='application/octet-stream'
            )
        else:
            # Return as JSON response
            return JSONResponse(
                content={
                    "template_name": template.name,
                    "document_type": template.document_type.value,
                    "format": request.format,
                    "content": exported_content
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export template: {str(e)}")

@router.post("/classify-and-export")
async def classify_and_export(
    query: str = Query(..., description="Document description query"),
    format: str = Query("json", description="Export format"),
    use_ai: bool = Query(True, description="Use AI for classification")
):
    """
    Classify a document and export the most appropriate template in one call.
    
    Args:
        query: Document description
        format: Export format (json, yaml, markdown)
        use_ai: Whether to use AI classification
        
    Returns:
        Classification result and exported template
    """
    try:
        # Classify document
        classification_result = classifier_engine.classify_document(query, use_ai)
        
        # Get templates for classified type
        templates = classifier_engine.get_templates(classification_result.document_type)
        
        if not templates:
            return JSONResponse(
                content={
                    "classification": {
                        "document_type": classification_result.document_type.value,
                        "confidence": classification_result.confidence,
                        "keywords": classification_result.keywords,
                        "reasoning": classification_result.reasoning
                    },
                    "template": None,
                    "message": "No templates available for this document type"
                }
            )
        
        # Export first template
        template = templates[0]
        exported_content = classifier_engine.export_template(template, format)
        
        return JSONResponse(
            content={
                "classification": {
                    "document_type": classification_result.document_type.value,
                    "confidence": classification_result.confidence,
                    "keywords": classification_result.keywords,
                    "reasoning": classification_result.reasoning
                },
                "template": {
                    "name": template.name,
                    "format": format,
                    "content": exported_content
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in classify and export: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to classify and export: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the AI document classifier service.
    
    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "AI Document Classifier",
        "supported_types": len([dt for dt in DocumentType if dt != DocumentType.UNKNOWN]),
        "available_templates": sum(len(templates) for templates in classifier_engine.templates.values())
    }

# Error handlers
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )



























