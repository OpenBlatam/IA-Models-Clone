"""
Document Module - Presentation Layer
Document controllers and presenters
"""

from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import Request

from .domain import DocumentEntity
from .application import (
    UploadDocumentUseCase,
    GetDocumentUseCase,
    ListDocumentsUseCase,
    DeleteDocumentUseCase
)


class DocumentPresenter:
    """Presenter for document entities"""
    
    @staticmethod
    def to_dict(document: DocumentEntity) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            "id": document.id,
            "user_id": document.user_id,
            "filename": document.filename,
            "file_size": document.file_size,
            "content_type": document.content_type,
            "status": document.status,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            "metadata": document.metadata or {}
        }
    
    @staticmethod
    def to_list(documents: list[DocumentEntity]) -> Dict[str, Any]:
        """Convert document list to response format"""
        return {
            "items": [DocumentPresenter.to_dict(doc) for doc in documents],
            "count": len(documents)
        }


class DocumentController:
    """Controller for document operations"""
    
    def __init__(
        self,
        upload_use_case: UploadDocumentUseCase,
        get_use_case: GetDocumentUseCase,
        list_use_case: ListDocumentsUseCase,
        delete_use_case: DeleteDocumentUseCase
    ):
        self.upload_use_case = upload_use_case
        self.get_use_case = get_use_case
        self.list_use_case = list_use_case
        self.delete_use_case = delete_use_case
        self.presenter = DocumentPresenter()
    
    async def upload(
        self,
        request: Request,
        command
    ) -> Dict[str, Any]:
        """Handle upload request"""
        try:
            document = await self.upload_use_case.execute(command)
            return {
                "success": True,
                "data": self.presenter.to_dict(document),
                "request_id": getattr(request.state, 'request_id', None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": getattr(request.state, 'request_id', None)
            }
    
    async def get(
        self,
        request: Request,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle get request"""
        try:
            from .application import GetDocumentQuery
            query = GetDocumentQuery(document_id=document_id, user_id=user_id)
            document = await self.get_use_case.execute(query)
            
            if not document:
                return {
                    "success": False,
                    "error": "Document not found",
                    "request_id": getattr(request.state, 'request_id', None)
                }
            
            return {
                "success": True,
                "data": self.presenter.to_dict(document),
                "request_id": getattr(request.state, 'request_id', None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": getattr(request.state, 'request_id', None)
            }
    
    async def list(
        self,
        request: Request,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle list request"""
        try:
            from .application import ListDocumentsQuery
            query = ListDocumentsQuery(
                user_id=user_id,
                limit=limit,
                offset=offset,
                search=search
            )
            documents = await self.list_use_case.execute(query)
            
            return {
                "success": True,
                "data": self.presenter.to_list(documents),
                "request_id": getattr(request.state, 'request_id', None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": getattr(request.state, 'request_id', None)
            }
    
    async def delete(
        self,
        request: Request,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle delete request"""
        try:
            from .application import DeleteDocumentCommand
            command = DeleteDocumentCommand(document_id=document_id, user_id=user_id)
            success = await self.delete_use_case.execute(command)
            
            return {
                "success": success,
                "message": "Document deleted" if success else "Document not found",
                "request_id": getattr(request.state, 'request_id', None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": getattr(request.state, 'request_id', None)
            }






