"""
Application Layer - Specifications Pattern
Encapsulate business rules using specification pattern
"""

from abc import ABC, abstractmethod
from typing import List

from ..domain.entities import DocumentEntity, VariantEntity, TopicEntity


class Specification(ABC):
    """Base specification interface"""
    
    @abstractmethod
    def is_satisfied_by(self, candidate: any) -> bool:
        """Check if candidate satisfies specification"""
        pass
    
    def __and__(self, other: 'Specification') -> 'AndSpecification':
        """Combine with AND"""
        return AndSpecification(self, other)
    
    def __or__(self, other: 'Specification') -> 'OrSpecification':
        """Combine with OR"""
        return OrSpecification(self, other)
    
    def __invert__(self) -> 'NotSpecification':
        """Negate specification"""
        return NotSpecification(self)


class AndSpecification(Specification):
    """AND combination of specifications"""
    
    def __init__(self, spec1: Specification, spec2: Specification):
        self.spec1 = spec1
        self.spec2 = spec2
    
    def is_satisfied_by(self, candidate: any) -> bool:
        """Both specifications must be satisfied"""
        return self.spec1.is_satisfied_by(candidate) and self.spec2.is_satisfied_by(candidate)


class OrSpecification(Specification):
    """OR combination of specifications"""
    
    def __init__(self, spec1: Specification, spec2: Specification):
        self.spec1 = spec1
        self.spec2 = spec2
    
    def is_satisfied_by(self, candidate: any) -> bool:
        """At least one specification must be satisfied"""
        return self.spec1.is_satisfied_by(candidate) or self.spec2.is_satisfied_by(candidate)


class NotSpecification(Specification):
    """NOT negation of specification"""
    
    def __init__(self, spec: Specification):
        self.spec = spec
    
    def is_satisfied_by(self, candidate: any) -> bool:
        """Specification must not be satisfied"""
        return not self.spec.is_satisfied_by(candidate)


# Document specifications
class DocumentOwnedByUserSpecification(Specification):
    """Specification: Document is owned by user"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
    
    def is_satisfied_by(self, document: DocumentEntity) -> bool:
        """Check if document is owned by user"""
        return document.is_owned_by(self.user_id)


class DocumentIsProcessedSpecification(Specification):
    """Specification: Document is processed"""
    
    def is_satisfied_by(self, document: DocumentEntity) -> bool:
        """Check if document is processed"""
        return document.status == "processed"


class DocumentIsReadySpecification(Specification):
    """Specification: Document is ready for operations"""
    
    def is_satisfied_by(self, document: DocumentEntity) -> bool:
        """Check if document is ready"""
        return document.status in ["ready", "processed"]


# Variant specifications
class VariantHasHighQualitySpecification(Specification):
    """Specification: Variant has high quality (similarity > threshold)"""
    
    def __init__(self, min_similarity: float = 0.7):
        self.min_similarity = min_similarity
    
    def is_satisfied_by(self, variant: VariantEntity) -> bool:
        """Check if variant has high quality"""
        return variant.similarity_score >= self.min_similarity


class VariantIsCompletedSpecification(Specification):
    """Specification: Variant generation is completed"""
    
    def is_satisfied_by(self, variant: VariantEntity) -> bool:
        """Check if variant is completed"""
        return variant.status == "completed"


# Topic specifications
class TopicIsRelevantSpecification(Specification):
    """Specification: Topic is relevant (score > threshold)"""
    
    def __init__(self, min_relevance: float = 0.5):
        self.min_relevance = min_relevance
    
    def is_satisfied_by(self, topic: TopicEntity) -> bool:
        """Check if topic is relevant"""
        return topic.is_relevant(self.min_relevance)


# Composite specifications
class UserCanAccessDocumentSpecification(Specification):
    """Composite: User can access document"""
    
    def __init__(self, user_id: str):
        self.owned_by = DocumentOwnedByUserSpecification(user_id)
        # Future: Add shared documents, role-based access, etc.
    
    def is_satisfied_by(self, document: DocumentEntity) -> bool:
        """Check if user can access"""
        return self.owned_by.is_satisfied_by(document)


class DocumentReadyForVariantGenerationSpecification(Specification):
    """Composite: Document is ready for variant generation"""
    
    def __init__(self):
        self.is_ready = DocumentIsReadySpecification()
    
    def is_satisfied_by(self, document: DocumentEntity) -> bool:
        """Check if document is ready for variant generation"""
        return self.is_ready.is_satisfied_by(document)


# Filter helpers using specifications
def filter_by_specification(
    items: List[any],
    specification: Specification
) -> List[any]:
    """Filter items by specification"""
    return [item for item in items if specification.is_satisfied_by(item)]






