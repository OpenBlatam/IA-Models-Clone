"""
Topic Module - Domain Layer
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List
from uuid import uuid4

from ...architecture.domain.value_objects import RelevanceScore


@dataclass
class TopicEntity:
    """Topic domain entity"""
    id: str
    document_id: str
    topic: str
    relevance_score: float
    category: str
    created_at: datetime
    
    def is_relevant(self, min_score: float = 0.5) -> bool:
        """Check if topic is relevant"""
        return self.relevance_score >= min_score


class TopicFactory:
    """Factory for creating topics"""
    
    @staticmethod
    def create(
        document_id: str,
        topic: str,
        relevance_score: float,
        category: str = "general"
    ) -> TopicEntity:
        """Create single topic"""
        relevance_vo = RelevanceScore(relevance_score)
        
        return TopicEntity(
            id=str(uuid4()),
            document_id=document_id,
            topic=topic,
            relevance_score=float(relevance_vo),
            category=category,
            created_at=datetime.utcnow()
        )
    
    @staticmethod
    def create_batch(
        document_id: str,
        topics: List[str],
        relevance_scores: List[float],
        category: str = "general"
    ) -> List[TopicEntity]:
        """Create multiple topics"""
        return [
            TopicFactory.create(document_id, topic, score, category)
            for topic, score in zip(topics, relevance_scores)
        ]






