"""
Professional Documents Integration
=================================

Integration layer for the Professional Documents feature with the existing
Blatam Academy API architecture.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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
from .services import DocumentGenerationService, DocumentExportService, TemplateService
from ..integrated.api import IntegratedRequest, IntegratedResponse, DocumentRequest

logger = logging.getLogger(__name__)

# Create integration router
integration_router = APIRouter(prefix="/integrated", tags=["Professional Documents Integration"])

# Initialize services
document_service = DocumentGenerationService()
export_service = DocumentExportService()
template_service = TemplateService()


class ProfessionalDocumentsIntegration:
    """Integration class for Professional Documents with existing API."""
    
    def __init__(self):
        self.document_service = document_service
        self.export_service = export_service
        self.template_service = template_service
    
    async def process_document_request(
        self, 
        request: DocumentRequest,
        user: User,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document request using the professional documents system."""
        
        try:
            # Convert DocumentRequest to DocumentGenerationRequest
            doc_gen_request = self._convert_to_generation_request(
                request, 
                user, 
                additional_context
            )
            
            # Generate the document
            response = await self.document_service.generate_document(doc_gen_request)
            
            if response.success:
                return {
                    "status": "success",
                    "document": response.document.dict(),
                    "generation_time": response.generation_time,
                    "word_count": response.word_count,
                    "estimated_pages": response.estimated_pages
                }
            else:
                return {
                    "status": "error",
                    "message": response.message,
                    "generation_time": response.generation_time
                }
                
        except Exception as e:
            logger.error(f"Error processing document request: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to process document request: {str(e)}"
            }
    
    def _convert_to_generation_request(
        self,
        request: DocumentRequest,
        user: User,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> DocumentGenerationRequest:
        """Convert DocumentRequest to DocumentGenerationRequest."""
        
        # Determine document type based on content or metadata
        document_type = self._determine_document_type(request, additional_context)
        
        # Extract title from metadata or generate from content
        title = self._extract_title(request, additional_context)
        
        # Create generation request
        return DocumentGenerationRequest(
            query=request.document_content or f"Process document: {request.document_url}",
            document_type=document_type,
            title=title,
            author=user.email if user else "System",
            language=request.language,
            tone="professional",
            length="medium",
            additional_requirements=additional_context.get("requirements") if additional_context else None
        )
    
    def _determine_document_type(
        self, 
        request: DocumentRequest, 
        context: Optional[Dict[str, Any]] = None
    ) -> DocumentType:
        """Determine the appropriate document type based on request content."""
        
        # Check if type is specified in context
        if context and "document_type" in context:
            try:
                return DocumentType(context["document_type"])
            except ValueError:
                pass
        
        # Analyze content to determine type
        content = (request.document_content or "").lower()
        
        if any(keyword in content for keyword in ["proposal", "suggest", "recommend"]):
            return DocumentType.PROPOSAL
        elif any(keyword in content for keyword in ["report", "analysis", "findings"]):
            return DocumentType.REPORT
        elif any(keyword in content for keyword in ["manual", "guide", "instructions"]):
            return DocumentType.MANUAL
        elif any(keyword in content for keyword in ["technical", "api", "code", "system"]):
            return DocumentType.TECHNICAL_DOCUMENT
        elif any(keyword in content for keyword in ["business plan", "startup", "investment"]):
            return DocumentType.BUSINESS_PLAN
        elif any(keyword in content for keyword in ["academic", "research", "study"]):
            return DocumentType.ACADEMIC_PAPER
        else:
            return DocumentType.REPORT  # Default
    
    def _extract_title(
        self, 
        request: DocumentRequest, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Extract or generate a title for the document."""
        
        # Check if title is provided in context
        if context and "title" in context:
            return context["title"]
        
        # Generate title from content
        content = request.document_content or ""
        if len(content) > 50:
            # Take first few words as title
            words = content.split()[:6]
            return " ".join(words).title()
        else:
            return f"Document - {datetime.now().strftime('%Y-%m-%d')}"
    
    async def export_document_integration(
        self,
        document_id: str,
        export_format: str,
        user: User,
        custom_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export a document using the professional documents system."""
        
        try:
            # Get the document
            document = self.document_service.get_document(document_id)
            
            if not document:
                return {
                    "status": "error",
                    "message": "Document not found"
                }
            
            # Create export request
            export_request = DocumentExportRequest(
                document_id=document_id,
                format=ExportFormat(export_format),
                custom_filename=custom_filename
            )
            
            # Export the document
            response = await self.export_service.export_document(document, export_request)
            
            if response.success:
                return {
                    "status": "success",
                    "file_path": response.file_path,
                    "file_size": response.file_size,
                    "download_url": response.download_url,
                    "export_time": response.export_time
                }
            else:
                return {
                    "status": "error",
                    "message": response.message,
                    "export_time": response.export_time
                }
                
        except Exception as e:
            logger.error(f"Error exporting document: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to export document: {str(e)}"
            }


# Initialize integration
integration = ProfessionalDocumentsIntegration()


@integration_router.post("/professional-documents/process")
async def process_document_with_professional_system(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Process a document request using the professional documents system.
    
    This endpoint integrates the existing document processing with the new
    professional document generation capabilities.
    """
    try:
        # Process the document
        result = await integration.process_document_request(request, user)
        
        return IntegratedResponse(
            request_id=str(uuid4()),
            status=result["status"],
            result=result,
            metadata={
                "user_id": str(user.id) if user else None,
                "timestamp": datetime.utcnow().isoformat(),
                "feature": "professional_documents"
            },
            performance_metrics={
                "generation_time": result.get("generation_time", 0),
                "word_count": result.get("word_count", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in professional documents integration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration error: {str(e)}")


@integration_router.post("/professional-documents/export")
async def export_document_integration(
    document_id: str,
    export_format: str,
    custom_filename: Optional[str] = None,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Export a document using the professional documents system.
    """
    try:
        # Export the document
        result = await integration.export_document_integration(
            document_id, export_format, user, custom_filename
        )
        
        return IntegratedResponse(
            request_id=str(uuid4()),
            status=result["status"],
            result=result,
            metadata={
                "user_id": str(user.id) if user else None,
                "timestamp": datetime.utcnow().isoformat(),
                "feature": "professional_documents_export"
            },
            performance_metrics={
                "export_time": result.get("export_time", 0),
                "file_size": result.get("file_size", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in document export integration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export integration error: {str(e)}")


@integration_router.get("/professional-documents/templates")
async def get_available_templates(
    document_type: Optional[str] = None,
    user: User = Depends(current_user)
):
    """
    Get available professional document templates.
    """
    try:
        if document_type:
            templates = template_service.get_templates_by_type(DocumentType(document_type))
        else:
            templates = template_service.get_all_templates()
        
        return IntegratedResponse(
            request_id=str(uuid4()),
            status="success",
            result={
                "templates": [template.dict() for template in templates],
                "total_count": len(templates)
            },
            metadata={
                "user_id": str(user.id) if user else None,
                "timestamp": datetime.utcnow().isoformat(),
                "feature": "professional_documents_templates"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template error: {str(e)}")


@integration_router.get("/professional-documents/stats")
async def get_document_stats_integration(
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get professional document statistics.
    """
    try:
        # Get all documents
        all_documents = document_service.list_documents(limit=1000, offset=0)
        
        # Calculate stats
        total_documents = len(all_documents)
        documents_by_type = {}
        total_word_count = 0
        
        for doc in all_documents:
            doc_type = doc.document_type.value
            documents_by_type[doc_type] = documents_by_type.get(doc_type, 0) + 1
            total_word_count += doc.word_count
        
        average_document_length = total_word_count / total_documents if total_documents > 0 else 0
        
        return IntegratedResponse(
            request_id=str(uuid4()),
            status="success",
            result={
                "total_documents": total_documents,
                "documents_by_type": documents_by_type,
                "total_word_count": total_word_count,
                "average_document_length": average_document_length
            },
            metadata={
                "user_id": str(user.id) if user else None,
                "timestamp": datetime.utcnow().isoformat(),
                "feature": "professional_documents_stats"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


# Utility functions for integration
class ProfessionalDocumentsUtils:
    """Utility functions for professional documents integration."""
    
    @staticmethod
    def create_document_from_content(
        content: str,
        document_type: DocumentType = DocumentType.REPORT,
        title: Optional[str] = None,
        author: Optional[str] = None
    ) -> DocumentGenerationRequest:
        """Create a document generation request from content."""
        
        return DocumentGenerationRequest(
            query=content,
            document_type=document_type,
            title=title or f"Generated Document - {datetime.now().strftime('%Y-%m-%d')}",
            author=author,
            tone="professional",
            length="medium"
        )
    
    @staticmethod
    def get_document_type_from_keywords(content: str) -> DocumentType:
        """Determine document type from content keywords."""
        
        content_lower = content.lower()
        
        # Define keyword mappings
        keyword_mappings = {
            DocumentType.PROPOSAL: ["proposal", "suggest", "recommend", "propose"],
            DocumentType.REPORT: ["report", "analysis", "findings", "summary"],
            DocumentType.MANUAL: ["manual", "guide", "instructions", "how to"],
            DocumentType.TECHNICAL_DOCUMENT: ["technical", "api", "code", "system", "implementation"],
            DocumentType.BUSINESS_PLAN: ["business plan", "startup", "investment", "funding"],
            DocumentType.ACADEMIC_PAPER: ["academic", "research", "study", "thesis"],
            DocumentType.WHITEPAPER: ["whitepaper", "white paper", "industry", "trends"],
            DocumentType.NEWSLETTER: ["newsletter", "news", "update", "announcement"],
            DocumentType.BROCHURE: ["brochure", "marketing", "promotional", "advertisement"],
            DocumentType.GUIDE: ["guide", "tutorial", "step by step", "walkthrough"],
            DocumentType.CATALOG: ["catalog", "products", "services", "listing"],
            DocumentType.PRESENTATION: ["presentation", "slides", "pitch", "demo"]
        }
        
        # Find the best match
        best_match = DocumentType.REPORT  # Default
        max_matches = 0
        
        for doc_type, keywords in keyword_mappings.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if matches > max_matches:
                max_matches = matches
                best_match = doc_type
        
        return best_match
    
    @staticmethod
    def format_document_metadata(document: ProfessionalDocument) -> Dict[str, Any]:
        """Format document metadata for API responses."""
        
        return {
            "id": document.id,
            "title": document.title,
            "subtitle": document.subtitle,
            "document_type": document.document_type.value,
            "author": document.author,
            "company": document.company,
            "date_created": document.date_created.isoformat(),
            "date_modified": document.date_modified.isoformat(),
            "word_count": document.word_count,
            "page_count": document.page_count,
            "status": document.status,
            "sections_count": len(document.sections)
        }




























