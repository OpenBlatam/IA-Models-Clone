"""
Presentation Layer - Controllers
HTTP request handlers that delegate to use cases
"""

from typing import Any, Dict, Optional
from fastapi import Request, Depends

from ..application.use_cases import (
    UploadDocumentUseCase,
    GetDocumentUseCase,
    ListDocumentsUseCase,
    GenerateVariantsUseCase,
    ExtractTopicsUseCase
)
from ..application.use_cases import (
    UploadDocumentCommand,
    GetDocumentQuery,
    ListDocumentsQuery,
    GenerateVariantsCommand,
    ExtractTopicsCommand
)
from .presenters import DocumentPresenter, ErrorPresenter
from ..layers import Controller


class DocumentController(Controller):
    """Document controller"""
    
    def __init__(
        self,
        upload_use_case: UploadDocumentUseCase,
        get_use_case: GetDocumentUseCase,
        list_use_case: ListDocumentsUseCase
    ):
        self.upload_use_case = upload_use_case
        self.get_use_case = get_use_case
        self.list_use_case = list_use_case
        self.presenter = DocumentPresenter()
        self.error_presenter = ErrorPresenter()
    
    async def handle_upload(
        self,
        request: Request,
        user_id: str,
        filename: str,
        file_content: bytes,
        auto_process: bool = True
    ) -> Dict[str, Any]:
        """Handle document upload"""
        try:
            command = UploadDocumentCommand(
                user_id=user_id,
                filename=filename,
                file_content=file_content,
                auto_process=auto_process
            )
            
            document = await self.upload_use_case.execute(command)
            return self.presenter.present(document)
        
        except Exception as e:
            return self.error_presenter.present(e, status_code=500)
    
    async def handle_get(
        self,
        request: Request,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle get document"""
        try:
            query = GetDocumentQuery(document_id=document_id, user_id=user_id)
            document = await self.get_use_case.execute(query)
            
            if not document:
                return self.error_presenter.present(
                    Exception("Document not found"),
                    status_code=404
                )
            
            return self.presenter.present(document)
        
        except Exception as e:
            return self.error_presenter.present(e, status_code=500)
    
    async def handle_list(
        self,
        request: Request,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle list documents"""
        try:
            query = ListDocumentsQuery(
                user_id=user_id,
                limit=limit,
                offset=offset,
                search=search
            )
            
            documents = await self.list_use_case.execute(query)
            return {
                "success": True,
                "data": [self.presenter.present(doc) for doc in documents],
                "total": len(documents)
            }
        
        except Exception as e:
            return self.error_presenter.present(e, status_code=500)


class VariantController(Controller):
    """Variant controller"""
    
    def __init__(self, generate_use_case: GenerateVariantsUseCase):
        self.generate_use_case = generate_use_case
        self.error_presenter = ErrorPresenter()
    
    async def handle_generate(
        self,
        request: Request,
        document_id: str,
        user_id: str,
        variant_count: int = 10
    ) -> Dict[str, Any]:
        """Handle variant generation"""
        try:
            command = GenerateVariantsCommand(
                document_id=document_id,
                user_id=user_id,
                variant_count=variant_count
            )
            
            variants = await self.generate_use_case.execute(command)
            return {
                "success": True,
                "data": variants,
                "count": len(variants)
            }
        
        except Exception as e:
            return self.error_presenter.present(e, status_code=500)


class TopicController(Controller):
    """Topic controller"""
    
    def __init__(self, extract_use_case: ExtractTopicsUseCase):
        self.extract_use_case = extract_use_case
        self.error_presenter = ErrorPresenter()
    
    async def handle_extract(
        self,
        request: Request,
        document_id: str,
        user_id: str,
        min_relevance: float = 0.5
    ) -> Dict[str, Any]:
        """Handle topic extraction"""
        try:
            command = ExtractTopicsCommand(
                document_id=document_id,
                user_id=user_id,
                min_relevance=min_relevance
            )
            
            topics = await self.extract_use_case.execute(command)
            return {
                "success": True,
                "data": topics,
                "count": len(topics)
            }
        
        except Exception as e:
            return self.error_presenter.present(e, status_code=500)






