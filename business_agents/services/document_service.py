"""
Document Service
================

Service layer for document generation operations.
"""

from typing import Dict, List, Any, Optional
import logging

from ..business_agents import BusinessAgentManager
from ..document_generator import DocumentType, DocumentFormat

logger = logging.getLogger(__name__)

class DocumentService:
    """Service for document generation operations."""
    
    def __init__(self, agent_manager: BusinessAgentManager):
        self.agent_manager = agent_manager
    
    async def generate_document(
        self,
        document_type: DocumentType,
        title: str,
        description: str,
        business_area: str,
        created_by: str,
        variables: Dict[str, Any] = None,
        format: DocumentFormat = DocumentFormat.MARKDOWN
    ) -> Dict[str, Any]:
        """Generate a business document."""
        
        try:
            result = await self.agent_manager.generate_business_document(
                document_type=document_type,
                title=title,
                description=description,
                business_area=business_area,
                created_by=created_by,
                variables=variables,
                format=format
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Document generation failed: {str(e)}")
            raise Exception("Failed to generate document")
    
    async def list_documents(
        self,
        business_area: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        created_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List generated documents with optional filters."""
        
        try:
            documents = self.agent_manager.document_generator.list_documents(
                business_area=business_area,
                document_type=document_type,
                created_by=created_by
            )
            
            return [
                {
                    "id": doc.id,
                    "request_id": doc.request_id,
                    "title": doc.title,
                    "format": doc.format.value,
                    "file_path": doc.file_path,
                    "size_bytes": doc.size_bytes,
                    "created_at": doc.created_at.isoformat(),
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            
        except Exception as e:
            logger.error(f"Document listing failed: {str(e)}")
            raise Exception("Failed to list documents")
    
    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get specific document details."""
        
        try:
            document = self.agent_manager.document_generator.get_document(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            return {
                "id": document.id,
                "request_id": document.request_id,
                "title": document.title,
                "content": document.content,
                "format": document.format.value,
                "file_path": document.file_path,
                "size_bytes": document.size_bytes,
                "created_at": document.created_at.isoformat(),
                "metadata": document.metadata
            }
            
        except ValueError as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in document retrieval: {str(e)}")
            raise Exception("Failed to get document")
    
    async def get_document_file_path(self, document_id: str) -> str:
        """Get document file path for download."""
        
        try:
            document = self.agent_manager.document_generator.get_document(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            if not document.file_path:
                raise ValueError("Document file not found")
            
            return document.file_path
            
        except ValueError as e:
            logger.error(f"Document file path retrieval failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in document file path retrieval: {str(e)}")
            raise Exception("Failed to get document file path")
