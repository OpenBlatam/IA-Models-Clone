"""
Application Layer - Factories
Factory patterns for creating complex objects
"""

from typing import Optional
from datetime import datetime
from uuid import uuid4

from ..domain.entities import DocumentEntity, VariantEntity, TopicEntity
from ..domain.value_objects import DocumentId, UserId, Filename, FileSize, RelevanceScore, SimilarityScore


class DocumentFactory:
    """Factory for creating document entities"""
    
    @staticmethod
    def create(
        user_id: str,
        filename: str,
        file_content: bytes,
        file_path: Optional[str] = None
    ) -> DocumentEntity:
        """Create new document entity"""
        filename_vo = Filename(filename)
        file_size_vo = FileSize(len(file_content))
        user_id_vo = UserId(user_id)
        
        if not file_path:
            file_path = f"uploads/{uuid4()}.pdf"
        
        return DocumentEntity(
            id=str(uuid4()),
            user_id=user_id_vo.value,
            filename=filename_vo.value,
            file_path=file_path,
            file_size=file_size_vo.bytes,
            content_type="application/pdf",
            status="uploaded",
            created_at=datetime.utcnow(),
            metadata={}
        )
    
    @staticmethod
    def from_dict(data: dict) -> DocumentEntity:
        """Create document from dictionary"""
        return DocumentEntity(
            id=data["id"],
            user_id=data["user_id"],
            filename=data["filename"],
            file_path=data["file_path"],
            file_size=data["file_size"],
            content_type=data.get("content_type", "application/pdf"),
            status=data.get("status", "uploaded"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") and isinstance(data["updated_at"], str) else data.get("updated_at"),
            metadata=data.get("metadata", {})
        )


class VariantFactory:
    """Factory for creating variant entities"""
    
    @staticmethod
    def create(
        document_id: str,
        variant_type: str,
        content: str,
        similarity_score: float
    ) -> VariantEntity:
        """Create new variant entity"""
        similarity_vo = SimilarityScore(similarity_score)
        
        return VariantEntity(
            id=str(uuid4()),
            document_id=document_id,
            variant_type=variant_type,
            content=content,
            similarity_score=float(similarity_vo),
            status="completed",
            created_at=datetime.utcnow(),
            metadata={}
        )
    
    @staticmethod
    def create_batch(
        document_id: str,
        variant_type: str,
        contents: list[str],
        similarity_scores: list[float]
    ) -> list[VariantEntity]:
        """Create multiple variants"""
        return [
            VariantFactory.create(document_id, variant_type, content, score)
            for content, score in zip(contents, similarity_scores)
        ]


class TopicFactory:
    """Factory for creating topic entities"""
    
    @staticmethod
    def create(
        document_id: str,
        topic: str,
        relevance_score: float,
        category: str = "general"
    ) -> TopicEntity:
        """Create new topic entity"""
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
        topics: list[str],
        relevance_scores: list[float],
        category: str = "general"
    ) -> list[TopicEntity]:
        """Create multiple topics"""
        return [
            TopicFactory.create(document_id, topic, score, category)
            for topic, score in zip(topics, relevance_scores)
        ]






