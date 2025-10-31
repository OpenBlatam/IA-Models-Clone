"""
Variant Module - Domain Layer
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
from uuid import uuid4

from ...architecture.domain.value_objects import SimilarityScore


@dataclass
class VariantEntity:
    """Variant domain entity"""
    id: str
    document_id: str
    variant_type: str
    content: str
    similarity_score: float
    status: str
    created_at: datetime
    metadata: Optional[Dict] = None
    
    def is_valid(self) -> bool:
        """Validate variant"""
        return (
            self.document_id is not None and
            self.variant_type is not None and
            len(self.content) > 0 and
            0.0 <= self.similarity_score <= 1.0
        )
    
    def has_high_quality(self, threshold: float = 0.7) -> bool:
        """Check quality"""
        return self.similarity_score >= threshold


class VariantFactory:
    """Factory for creating variants"""
    
    @staticmethod
    def create(
        document_id: str,
        variant_type: str,
        content: str,
        similarity_score: float
    ) -> VariantEntity:
        """Create single variant"""
        similarity_vo = SimilarityScore(similarity_score)
        
        return VariantEntity(
            id=str(uuid4()),
            document_id=document_id,
            variant_type=variant_type,
            content=content,
            similarity_score=float(similarity_vo),
            status="completed",
            created_at=datetime.utcnow()
        )
    
    @staticmethod
    def create_batch(
        document_id: str,
        variant_type: str,
        contents: List[str],
        similarity_scores: List[float]
    ) -> List[VariantEntity]:
        """Create multiple variants"""
        return [
            VariantFactory.create(document_id, variant_type, content, score)
            for content, score in zip(contents, similarity_scores)
        ]






