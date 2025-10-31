"""
Presentation Layer - Presenters
Format domain entities for API responses
"""

from typing import Any, Dict, Optional
from datetime import datetime

from ..domain.entities import DocumentEntity, VariantEntity, TopicEntity
from ..layers import Presenter


class DocumentPresenter(Presenter):
    """Presenter for document entities"""
    
    def present(self, document: DocumentEntity) -> Dict[str, Any]:
        """Format document for API response"""
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


class VariantPresenter(Presenter):
    """Presenter for variant entities"""
    
    def present(self, variant: VariantEntity) -> Dict[str, Any]:
        """Format variant for API response"""
        return {
            "id": variant.id,
            "document_id": variant.document_id,
            "variant_type": variant.variant_type,
            "similarity_score": variant.similarity_score,
            "status": variant.status,
            "created_at": variant.created_at.isoformat(),
            "metadata": variant.metadata or {}
        }


class TopicPresenter(Presenter):
    """Presenter for topic entities"""
    
    def present(self, topic: TopicEntity) -> Dict[str, Any]:
        """Format topic for API response"""
        return {
            "id": topic.id,
            "document_id": topic.document_id,
            "topic": topic.topic,
            "relevance_score": topic.relevance_score,
            "category": topic.category,
            "created_at": topic.created_at.isoformat()
        }


class ErrorPresenter(Presenter):
    """Presenter for errors"""
    
    def present(
        self,
        error: Exception,
        status_code: int = 500,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format error for API response"""
        return {
            "success": False,
            "error": {
                "message": str(error),
                "code": error.__class__.__name__,
                "status_code": status_code
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }






