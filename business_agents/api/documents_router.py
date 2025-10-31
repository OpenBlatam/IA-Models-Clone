"""
Documents API Router
===================

FastAPI router for document generation and management.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import os

from ..business_agents import BusinessAgentManager
from ..document_generator import DocumentType, DocumentFormat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Dependency to get agent manager
def get_agent_manager() -> BusinessAgentManager:
    """Get the global agent manager instance."""
    from ..main import app
    return app.state.agent_manager

# Request/Response Models
class DocumentGenerationRequest(BaseModel):
    document_type: DocumentType = Field(..., description="Type of document to generate")
    title: str = Field(..., description="Document title")
    description: str = Field(..., description="Document description")
    business_area: str = Field(..., description="Business area")
    variables: Optional[Dict[str, Any]] = Field(None, description="Document variables")
    format: DocumentFormat = Field(DocumentFormat.MARKDOWN, description="Output format")

class DocumentGenerationResponse(BaseModel):
    document_id: str
    request_id: str
    title: str
    file_path: str
    format: str
    size_bytes: int
    created_at: str
    status: str = "generated"

class DocumentListResponse(BaseModel):
    documents: List[DocumentGenerationResponse]
    total: int

class DocumentRequest(BaseModel):
    document_type: DocumentType
    title: str
    description: str
    business_area: str
    created_by: str
    variables: Optional[Dict[str, Any]] = None
    format: DocumentFormat = DocumentFormat.MARKDOWN

# Endpoints
@router.post("/generate", response_model=DocumentGenerationResponse)
async def generate_document(
    request: DocumentGenerationRequest,
    created_by: str = "system",  # This would come from authentication
    background_tasks: BackgroundTasks = None,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
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
        
        return DocumentGenerationResponse(
            document_id=result["document_id"],
            request_id=result["request_id"],
            title=result["title"],
            file_path=result["file_path"],
            format=result["format"],
            size_bytes=result["size_bytes"],
            created_at=result["created_at"],
            status="generated"
        )
        
    except Exception as e:
        logger.error(f"Failed to generate document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate document: {str(e)}")

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    business_area: Optional[str] = None,
    document_type: Optional[DocumentType] = None,
    created_by: Optional[str] = None,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """List generated documents with optional filtering."""
    
    try:
        # This would integrate with document storage
        # For now, return empty list
        documents = []
        
        return DocumentListResponse(
            documents=documents,
            total=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/{document_id}", response_model=DocumentGenerationResponse)
async def get_document(
    document_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get document information by ID."""
    
    try:
        # This would retrieve from document storage
        # For now, return a mock response
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Download a generated document."""
    
    try:
        # This would retrieve the actual file path
        # For now, return a mock response
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download document: {str(e)}")

@router.get("/{document_id}/content")
async def get_document_content(
    document_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get document content as text."""
    
    try:
        # This would retrieve the document content
        # For now, return a mock response
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document content {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document content: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Delete a document."""
    
    try:
        # This would delete from document storage
        # For now, return success
        return {"message": f"Document {document_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/types/", response_model=List[str])
async def get_document_types():
    """Get all available document types."""
    
    try:
        return [doc_type.value for doc_type in DocumentType]
        
    except Exception as e:
        logger.error(f"Failed to get document types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document types: {str(e)}")

@router.get("/formats/", response_model=List[str])
async def get_document_formats():
    """Get all available document formats."""
    
    try:
        return [format_type.value for format_type in DocumentFormat]
        
    except Exception as e:
        logger.error(f"Failed to get document formats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document formats: {str(e)}")

@router.get("/templates/", response_model=Dict[str, List[Dict[str, Any]]])
async def get_document_templates():
    """Get document templates by business area."""
    
    try:
        # This would return actual document templates
        templates = {
            "marketing": [
                {
                    "name": "Marketing Strategy Document",
                    "description": "Comprehensive marketing strategy template",
                    "document_type": "strategy",
                    "sections": ["executive_summary", "market_analysis", "strategy", "implementation"]
                },
                {
                    "name": "Campaign Brief",
                    "description": "Marketing campaign brief template",
                    "document_type": "campaign",
                    "sections": ["objectives", "target_audience", "channels", "budget", "timeline"]
                }
            ],
            "sales": [
                {
                    "name": "Sales Proposal",
                    "description": "Professional sales proposal template",
                    "document_type": "proposal",
                    "sections": ["executive_summary", "solution", "pricing", "implementation", "next_steps"]
                },
                {
                    "name": "Sales Report",
                    "description": "Sales performance report template",
                    "document_type": "report",
                    "sections": ["performance_summary", "metrics", "analysis", "recommendations"]
                }
            ],
            "operations": [
                {
                    "name": "Process Documentation",
                    "description": "Business process documentation template",
                    "document_type": "process",
                    "sections": ["overview", "steps", "roles", "tools", "metrics"]
                },
                {
                    "name": "Standard Operating Procedure",
                    "description": "SOP template for operational procedures",
                    "document_type": "sop",
                    "sections": ["purpose", "scope", "procedure", "responsibilities", "references"]
                }
            ]
        }
        
        return templates
        
    except Exception as e:
        logger.error(f"Failed to get document templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document templates: {str(e)}")


