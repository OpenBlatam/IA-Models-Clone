"""
Topic Module - Application Layer
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from .domain import TopicEntity


@dataclass
class ExtractTopicsCommand:
    """Command to extract topics"""
    document_id: str
    user_id: str
    min_relevance: float = 0.5


@dataclass
class GetTopicQuery:
    """Query to get topic"""
    topic_id: str
    user_id: str


@dataclass
class ListTopicsQuery:
    """Query to list topics"""
    document_id: str
    user_id: str
    min_relevance: float = 0.5


class ExtractTopicsUseCase(ABC):
    """Extract topics use case"""
    
    @abstractmethod
    async def execute(self, command: ExtractTopicsCommand) -> List[TopicEntity]:
        """Execute extraction"""
        pass


class GetTopicUseCase(ABC):
    """Get topic use case"""
    
    @abstractmethod
    async def execute(self, query: GetTopicQuery) -> Optional[TopicEntity]:
        """Execute get"""
        pass


class ListTopicsUseCase(ABC):
    """List topics use case"""
    
    @abstractmethod
    async def execute(self, query: ListTopicsQuery) -> List[TopicEntity]:
        """Execute list"""
        pass






