"""
Document Module - Domain Layer
Document-specific domain logic
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
from uuid import uuid4

from ...architecture.domain.value_objects import DocumentId, UserId, Filename, FileSize


@dataclass
class DocumentEntity:
    """Document domain entity"""
    id: str
    user_id: str
    filename: str
    file_path: str
    file_size: int
    content_type: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict] = None
    
    @classmethod
    def create(
        cls,
        user_id: str,
        filename: str,
        file_size: int,
        file_path: Optional[str] = None
    ) -> 'DocumentEntity':
        """Factory method to create document"""
        filename_vo = Filename(filename)
        file_size_vo = FileSize(file_size)
        user_id_vo = UserId(user_id)
        
        if not file_path:
            file_path = f"uploads/{uuid4()}.pdf"
        
        return cls(
            id=str(uuid4()),
            user_id=user_id_vo.value,
            filename=filename_vo.value,
            file_path=file_path,
            file_size=file_size_vo.bytes,
            content_type="application/pdf",
            status="uploaded",
            created_at=datetime.utcnow()
        )
    
    def is_owned_by(self, user_id: str) -> bool:
        """Check ownership"""
        return self.user_id == user_id
    
    def can_be_accessed_by(self, user_id: str) -> bool:
        """Check access permissions"""
        return self.is_owned_by(user_id)
    
    def mark_as_processed(self):
        """Mark as processed"""
        self.status = "processed"
        self.updated_at = datetime.utcnow()
    
    def can_be_deleted_by(self, user_id: str) -> bool:
        """Check if can be deleted"""
        return self.is_owned_by(user_id) and self.status != "deleted"


class DocumentFactory:
    """Factory for creating documents"""
    
    @staticmethod
    def create_document(
        user_id: str,
        filename: str,
        file_content: bytes,
        file_path: Optional[str] = None
    ) -> DocumentEntity:
        """Create new document"""
        return DocumentEntity.create(
            user_id=user_id,
            filename=filename,
            file_size=len(file_content),
            file_path=file_path
        )
    
    @staticmethod
    def from_dict(data: Dict) -> DocumentEntity:
        """Create from dictionary"""
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






