"""Document service for managing documents"""

from typing import List, Optional
from datetime import datetime
import uuid
import logging
import os

from models.schemas import (
    DocumentRequest,
    DocumentResponse,
)

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing documents"""
    
    def __init__(self, upload_dir: str = "./uploads"):
        """Initialize document service"""
        self.documents = {}  # In-memory storage
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
    
    async def upload_document(
        self,
        request: DocumentRequest,
        file_content: bytes,
        mime_type: str
    ) -> DocumentResponse:
        """Upload a document"""
        try:
            document_id = f"DOC{str(uuid.uuid4())[:8].upper()}"
            
            # Save file
            file_path = os.path.join(self.upload_dir, f"{document_id}_{request.file_name}")
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            file_size = len(file_content)
            file_url = f"/documents/{document_id}/download"
            
            document = DocumentResponse(
                document_id=document_id,
                shipment_id=request.shipment_id,
                document_type=request.document_type,
                file_name=request.file_name,
                file_url=file_url,
                file_size=file_size,
                mime_type=mime_type,
                description=request.description,
                uploaded_at=datetime.now()
            )
            
            self.documents[document_id] = document
            
            logger.info(f"Document uploaded: {document_id}")
            return document
            
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[DocumentResponse]:
        """Get document by ID"""
        return self.documents.get(document_id)
    
    async def get_documents_by_shipment(
        self,
        shipment_id: str
    ) -> List[DocumentResponse]:
        """Get documents for a shipment"""
        return [
            doc for doc in self.documents.values()
            if doc.shipment_id == shipment_id
        ]
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        document = await self.get_document(document_id)
        if not document:
            return False
        
        # Delete file
        file_path = os.path.join(
            self.upload_dir,
            f"{document_id}_{document.file_name}"
        )
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from storage
        del self.documents[document_id]
        
        logger.info(f"Document deleted: {document_id}")
        return True













