"""
Topic Module - Infrastructure Layer
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .domain import TopicEntity


class TopicRepository(ABC):
    """Topic repository interface"""
    
    @abstractmethod
    async def get_by_id(self, topic_id: str) -> Optional[TopicEntity]:
        """Get topic by ID"""
        pass
    
    @abstractmethod
    async def save(self, topic: TopicEntity) -> TopicEntity:
        """Save topic"""
        pass
    
    @abstractmethod
    async def save_batch(self, topics: List[TopicEntity]) -> List[TopicEntity]:
        """Save multiple topics"""
        pass
    
    @abstractmethod
    async def delete(self, topic_id: str) -> bool:
        """Delete topic"""
        pass
    
    @abstractmethod
    async def find_by_document(
        self,
        document_id: str,
        min_relevance: float = 0.5
    ) -> List[TopicEntity]:
        """Find topics by document"""
        pass






