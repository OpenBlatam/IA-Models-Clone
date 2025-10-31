"""
Topic Module - Presentation Layer
"""

from typing import Dict, Any, List, Optional
from fastapi import Request

from .domain import TopicEntity
from .application import (
    ExtractTopicsUseCase,
    GetTopicUseCase,
    ListTopicsUseCase
)


class TopicPresenter:
    """Presenter for topic entities"""
    
    @staticmethod
    def to_dict(topic: TopicEntity) -> Dict[str, Any]:
        """Convert topic to dictionary"""
        return {
            "id": topic.id,
            "document_id": topic.document_id,
            "topic": topic.topic,
            "relevance_score": topic.relevance_score,
            "category": topic.category,
            "created_at": topic.created_at.isoformat()
        }
    
    @staticmethod
    def to_list(topics: List[TopicEntity]) -> Dict[str, Any]:
        """Convert topic list to response format"""
        return {
            "items": [TopicPresenter.to_dict(t) for t in topics],
            "count": len(topics),
            "relevant_count": sum(1 for t in topics if t.is_relevant())
        }


class TopicController:
    """Controller for topic operations"""
    
    def __init__(
        self,
        extract_use_case: ExtractTopicsUseCase,
        get_use_case: GetTopicUseCase,
        list_use_case: ListTopicsUseCase
    ):
        self.extract_use_case = extract_use_case
        self.get_use_case = get_use_case
        self.list_use_case = list_use_case
        self.presenter = TopicPresenter()
    
    async def extract(
        self,
        request: Request,
        command
    ) -> Dict[str, Any]:
        """Handle extract request"""
        try:
            topics = await self.extract_use_case.execute(command)
            return {
                "success": True,
                "data": self.presenter.to_list(topics),
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
        topic_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle get request"""
        try:
            from .application import GetTopicQuery
            query = GetTopicQuery(topic_id=topic_id, user_id=user_id)
            topic = await self.get_use_case.execute(query)
            
            if not topic:
                return {
                    "success": False,
                    "error": "Topic not found",
                    "request_id": getattr(request.state, 'request_id', None)
                }
            
            return {
                "success": True,
                "data": self.presenter.to_dict(topic),
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
        document_id: str,
        user_id: str,
        min_relevance: float = 0.5
    ) -> Dict[str, Any]:
        """Handle list request"""
        try:
            from .application import ListTopicsQuery
            query = ListTopicsQuery(
                document_id=document_id,
                user_id=user_id,
                min_relevance=min_relevance
            )
            topics = await self.list_use_case.execute(query)
            
            return {
                "success": True,
                "data": self.presenter.to_list(topics),
                "request_id": getattr(request.state, 'request_id', None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": getattr(request.state, 'request_id', None)
            }






