"""
Domain Layer - Domain Services
Domain-specific business logic that doesn't fit in entities
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .entities import DocumentEntity, VariantEntity, TopicEntity
from .value_objects import RelevanceScore, SimilarityScore


class DocumentDomainService(ABC):
    """Domain service for document operations"""
    
    @abstractmethod
    def can_user_access_document(
        self,
        document: DocumentEntity,
        user_id: str
    ) -> bool:
        """Check if user can access document"""
        pass
    
    @abstractmethod
    def validate_document_for_processing(
        self,
        document: DocumentEntity
    ) -> tuple[bool, Optional[str]]:
        """Validate document can be processed"""
        pass


class VariantDomainService(ABC):
    """Domain service for variant operations"""
    
    @abstractmethod
    def calculate_similarity(
        self,
        original: str,
        variant: str
    ) -> SimilarityScore:
        """Calculate similarity between original and variant"""
        pass
    
    @abstractmethod
    def filter_variants_by_quality(
        self,
        variants: List[VariantEntity],
        min_similarity: float = 0.7
    ) -> List[VariantEntity]:
        """Filter variants by quality threshold"""
        pass


class TopicDomainService(ABC):
    """Domain service for topic operations"""
    
    @abstractmethod
    def calculate_relevance(
        self,
        topic: str,
        document_content: str
    ) -> RelevanceScore:
        """Calculate topic relevance to document"""
        pass
    
    @abstractmethod
    def rank_topics_by_relevance(
        self,
        topics: List[TopicEntity]
    ) -> List[TopicEntity]:
        """Rank topics by relevance score"""
        pass


class DocumentAccessService(DocumentDomainService):
    """Concrete implementation of document access service"""
    
    def can_user_access_document(
        self,
        document: DocumentEntity,
        user_id: str
    ) -> bool:
        """Check if user can access document"""
        # Owner always has access
        if document.is_owned_by(user_id):
            return True
        
        # Future: Check shared permissions, roles, etc.
        return False
    
    def validate_document_for_processing(
        self,
        document: DocumentEntity
    ) -> tuple[bool, Optional[str]]:
        """Validate document can be processed"""
        if document.status == "processing":
            return False, "Document is already being processed"
        
        if document.status == "error":
            return False, "Document has errors and cannot be processed"
        
        if document.status == "deleted":
            return False, "Document has been deleted"
        
        return True, None


class VariantQualityService(VariantDomainService):
    """Concrete implementation of variant quality service"""
    
    def calculate_similarity(
        self,
        original: str,
        variant: str
    ) -> SimilarityScore:
        """Calculate similarity using simple string comparison"""
        # Simplified - in production would use proper similarity algorithms
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original, variant).ratio()
        return SimilarityScore(similarity)
    
    def filter_variants_by_quality(
        self,
        variants: List[VariantEntity],
        min_similarity: float = 0.7
    ) -> List[VariantEntity]:
        """Filter variants above quality threshold"""
        return [
            v for v in variants
            if v.similarity_score >= min_similarity
        ]


class TopicRelevanceService(TopicDomainService):
    """Concrete implementation of topic relevance service"""
    
    def calculate_relevance(
        self,
        topic: str,
        document_content: str
    ) -> RelevanceScore:
        """Calculate topic relevance"""
        # Simplified - in production would use NLP/AI
        topic_lower = topic.lower()
        content_lower = document_content.lower()
        
        # Count occurrences
        occurrences = content_lower.count(topic_lower)
        word_count = len(document_content.split())
        
        # Simple relevance calculation
        if word_count == 0:
            relevance = 0.0
        else:
            relevance = min(1.0, (occurrences * len(topic.split())) / word_count * 10)
        
        return RelevanceScore(relevance)
    
    def rank_topics_by_relevance(
        self,
        topics: List[TopicEntity]
    ) -> List[TopicEntity]:
        """Sort topics by relevance (highest first)"""
        return sorted(topics, key=lambda t: t.relevance_score, reverse=True)






