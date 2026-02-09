"""
Document Module - Infrastructure Layer
Document repository implementation
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .domain import DocumentEntity


class DocumentRepository(ABC):
    """Document repository interface"""
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[DocumentEntity]:
        """Get document by ID"""
        pass
    
    @abstractmethod
    async def save(self, document: DocumentEntity) -> DocumentEntity:
        """Save document"""
        pass
    
    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete document"""
        pass
    
    @abstractmethod
    async def find_by_user(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[DocumentEntity]:
        """Find documents by user"""
        pass
    
    @abstractmethod
    async def count_by_user(self, user_id: str) -> int:
        """Count documents by user"""
        pass


# Concrete implementations can be added here
# e.g., DatabaseDocumentRepository, InMemoryDocumentRepository, etc.






